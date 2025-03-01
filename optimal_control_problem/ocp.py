import time
import numpy as np
import casadi as ca
import osqp
from scipy import sparse

from utils.helpers import *
from dynamics import DynamicsRNEA


class OCP:
    def __init__(self, robot, nodes):
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.gait_sequence = robot.gait_sequence
        self.ee_ids = robot.ee_ids
        self.arm_ee_id = robot.arm_ee_id
        self.n_feet = len(self.ee_ids)

        self.nq = robot.nq
        self.nv = robot.nv
        self.nf = robot.nf
        self.nj = robot.nj

        self.nodes = nodes
        self.mass = self.data.mass[0]
        self.opti = ca.Opti()

    def setup_problem(self):
        self.setup_variables()
        self.setup_parameters()
        self.setup_targets()
        self.setup_constraints()
        obj = self.setup_objective()
        self.opti.minimize(obj)

    def setup_variables(self):
        """Initialize decision variables."""
        pass

    def setup_parameters(self):
        """
        Parameters that are the same for all OCPs.
        """
        self.x_init = self.opti.parameter(self.nx)  # initial state
        self.dt_min = self.opti.parameter(1)  # first time step size (used for sim)
        self.dt_max = self.opti.parameter(1)  # last time step size
        self.contact_schedule = self.opti.parameter(self.n_feet, self.nodes) # in_contact: 0 or 1
        self.swing_schedule = self.opti.parameter(self.n_feet, self.nodes) # swing_phase: from 0 to 1
        self.n_contacts = self.opti.parameter(1)  # number of contact feet
        self.swing_period = self.opti.parameter(1)  # swing period (in seconds)
        self.swing_height = self.opti.parameter(1)  # max swing height
        self.swing_vel_limits = self.opti.parameter(2)  # start and end swing velocities

        self.Q_diag = self.opti.parameter(self.ndx_opt)  # state weights
        self.R_diag = self.opti.parameter(self.nu_opt[0])  # input weights

        self.base_vel_des = self.opti.parameter(6)  # linear + angular velocity
        self.arm_f_des = self.opti.parameter(3)  # force at end-effector
        self.arm_vel_des = self.opti.parameter(3)  # linear velocity at end-effector

        # Increasing time step sizes
        ratio = self.dt_max / self.dt_min
        gamma = ratio ** (1 / (self.nodes - 1))  # growth factor
        self.dts = [self.dt_min * gamma**i for i in range(self.nodes)]

    def setup_targets(self):
        """Determine desired state and input."""
        pass

    def setup_objective(self):
        """
        Default objective with state and input weight matrices Q and R.
        """
        obj = 0
        Q = ca.diag(self.Q_diag)
        R = ca.diag(self.R_diag)
        for i in range(self.nodes):
            # Track desired state and input
            dx = self.DX_opt[i]
            u = self.U_opt[i]
            err_dx = dx - self.dx_des
            err_u = u - self.u_des
            obj += err_dx.T @ Q @ err_dx
            obj += err_u.T @ R @ err_u

        # Final state
        dx = self.DX_opt[self.nodes]
        err_dx = dx - self.dx_des
        obj += err_dx.T @ Q @ err_dx

        return obj

    def setup_constraints(self, mu=0.7):
        """
        Constraints that are the same for all OCPs.
        The dynamics constraints are implemented in the subclasses.
        """
        # Initial state
        self.opti.subject_to(self.DX_opt[0] == [0] * self.ndx_opt)

        for i in range(self.nodes):
            # Gather state and input info
            q = self.get_q(i)
            v = self.get_v(i)
            forces = self.get_forces(i)

            # Dynamics constraints
            self.setup_dynamics_constraints(i)

            # Contact and swing constraints
            for idx, frame_id in enumerate(self.ee_ids):
                f_e = forces[idx * 3 : (idx + 1) * 3]

                # Get contact and swing info
                in_contact = self.contact_schedule[idx, i]
                swing_phase = self.swing_schedule[idx, i]

                # Contact: Friction cone
                f_normal = f_e[2]
                f_tangent_square = f_e[0]**2 + f_e[1]**2
                self.opti.subject_to(in_contact * f_normal >= 0)
                self.opti.subject_to(in_contact * mu**2 * f_normal**2 >= in_contact * f_tangent_square)

                # Swing: Zero force
                self.opti.subject_to((1 - in_contact) * f_e == [0] * 3)

                if i == 0 and type(self.dyn) == DynamicsRNEA:
                    # RNEA: No state constraints at first time step
                    # Centroidal: Inputs control velocity, so keep constraints
                    continue

                # Contact: Zero xy-velocity
                vel = self.dyn.get_frame_velocity(frame_id)(q, v)
                vel_xy = vel[:2]
                self.opti.subject_to(in_contact * vel_xy == [0] * 2)

                # Contact: Zero z-velocity / Swing: Spline z-velocity
                vel_z = vel[2]
                vel_z_des = get_spline_vel_z(
                    swing_phase,
                    swing_period=self.swing_period,
                    h_max=self.swing_height,
                    v_liftoff=self.swing_vel_limits[0],
                    v_touchdown=self.swing_vel_limits[1]    
                )
                vel_diff = vel_z - vel_z_des
                self.opti.subject_to(in_contact * vel_z + (1 - in_contact) * vel_diff == 0)

            # Warm start: Use n_contacts from gait sequence for u_des
            self.opti.set_value(self.n_contacts, self.gait_sequence.n_contacts)
            self.opti.set_initial(self.DX_opt[i], np.zeros(self.ndx_opt))
            u_warm = self.opti.value(self.u_des)[:self.nu_opt[i]]
            self.opti.set_initial(self.U_opt[i], u_warm)

            if i == 0 and type(self.dyn) == DynamicsRNEA:
                # RNEA: No state constraints at first time step
                # Centroidal: Inputs control velocity, so keep constraints
                continue

            # Arm task
            if self.arm_ee_id:
                # Zero end-effector velocity
                vel = self.dyn.get_frame_velocity(self.arm_ee_id)(q, v)
                vel_lin = vel[:3]
                vel_diff = vel_lin - self.arm_vel_des
                self.opti.subject_to(vel_diff == [0] * 3)

                # Force at end-effector (after all feet)
                f_e = forces[3*self.n_feet:]
                self.opti.subject_to(f_e == self.arm_f_des)

            # Joint limits
            pos_min = self.robot.joint_pos_min
            pos_max = self.robot.joint_pos_max
            vel_min = -self.robot.joint_vel_max
            vel_max = self.robot.joint_vel_max
            q_j = q[7:]  # skip base quaternion
            v_j = v[6:]  # skip base angular velocity
            self.opti.subject_to(self.opti.bounded(pos_min, q_j, pos_max))
            self.opti.subject_to(self.opti.bounded(vel_min, v_j, vel_max))

        # Warm start
        self.opti.set_initial(self.DX_opt[self.nodes], np.zeros(self.ndx_opt))

        # TODO: Clean up how solutions are stored
        self.DX_prev = None
        self.U_prev = None
        self.lam_g = None

    def setup_dynamics_constraints(self, i):
        pass

    def get_q(self, i):
        pass

    def get_v(self, i):
        pass

    def get_forces(self, i):
        pass

    def set_time_params(self, dt_min, dt_max):
        self.opti.set_value(self.dt_min, dt_min)
        self.opti.set_value(self.dt_max, dt_max)

    def set_swing_params(self, swing_height, swing_vel_limits):
        self.opti.set_value(self.swing_height, swing_height)
        self.opti.set_value(self.swing_vel_limits, swing_vel_limits)

    def set_tracking_target(self, base_vel_des, arm_f_des, arm_vel_des):
        self.opti.set_value(self.base_vel_des, base_vel_des)
        if self.arm_ee_id:
            self.opti.set_value(self.arm_f_des, arm_f_des)
            self.opti.set_value(self.arm_vel_des, arm_vel_des)

    def set_weights(self, Q_diag, R_diag):
        self.opti.set_value(self.Q_diag, Q_diag)
        self.opti.set_value(self.R_diag, R_diag)

    def update_initial_state(self, x_init):
        self.opti.set_value(self.x_init, x_init)

    def update_gait_sequence(self, t_current):
        dts = [self.opti.value(self.dts[i]) for i in range(self.nodes)]
        contact_schedule, swing_schedule = self.gait_sequence.get_gait_schedule(t_current, dts, self.nodes)
        n_contacts = self.gait_sequence.n_contacts
        swing_period = self.gait_sequence.swing_period
        self.opti.set_value(self.contact_schedule, contact_schedule)
        self.opti.set_value(self.swing_schedule, swing_schedule)
        self.opti.set_value(self.n_contacts, n_contacts)
        self.opti.set_value(self.swing_period, swing_period)

    def warm_start(self):
        pass

    def init_solver(self, solver="fatrop", warm_start=False):
        self.solver_type = solver

        if solver == "fatrop":
            opts = {
                "expand": True,
                "structure_detection": "auto",
                "debug": True,
            }
            opts["fatrop"] = {
                "print_level": 1,
                "max_iter": 10,
                "tol": 1e-3,
                "mu_init": 1e-4,
                "warm_start_init_point": True,
                "warm_start_mult_bound_push": 1e-7,
                "bound_push": 1e-7,
            }
            self.opti.solver(solver, opts)

            # Params (as list for code generation)
            self.solver_params = [self.x_init, self.dt_min, self.dt_max, self.contact_schedule, self.swing_schedule,
                                  self.n_contacts, self.swing_period, self.swing_height, self.swing_vel_limits,
                                  self.Q_diag, self.R_diag, self.base_vel_des]
            if self.arm_ee_id:
                self.solver_params += [self.arm_f_des, self.arm_vel_des]
            if warm_start:
                self.solver_params += [self.opti.x]

        elif solver == "osqp":
            # Get all info from self.opti
            x = self.opti.x
            p = self.opti.p
            f = self.opti.f
            g = self.opti.g
            lbg = self.opti.lbg
            ubg = self.opti.ubg

            J_g = ca.jacobian(g, x)
            hess_f, grad_f = ca.hessian(f, x)

            # Store data functions
            # TODO: Code generation
            self.sqp_data = ca.Function("sqp_data", [x, p], [hess_f, grad_f, J_g, g, lbg, ubg])
            self.f_data = ca.Function("f_data", [x, p], [f, grad_f])
            self.g_data = ca.Function("g_data", [x, p], [g, lbg, ubg])

            # OSQP options
            self.osqp_opts = {
                "max_iter": 100,
                "alpha": 1.4,
                "rho": 2e-2,
            }

            # Params (as stacked vector for data functions)
            self.solver_params = ca.vertcat(
                self.opti.value(self.x_init),
                self.opti.value(self.dt_min),
                self.opti.value(self.dt_max),
                ca.vec(self.opti.value(self.contact_schedule)),  # flattened
                ca.vec(self.opti.value(self.swing_schedule)),  # flattened
                self.opti.value(self.n_contacts),
                self.opti.value(self.swing_period),
                self.opti.value(self.swing_height),
                self.opti.value(self.swing_vel_limits),
                self.opti.value(self.Q_diag),
                self.opti.value(self.R_diag),
                self.opti.value(self.base_vel_des),
            )
            if self.arm_ee_id:
                self.solver_params = ca.vertcat(
                    self.solver_params,
                    self.opti.value(self.arm_f_des),
                    self.opti.value(self.arm_vel_des),
                )

        else:
            raise ValueError(f"Solver {solver} not supported")
        
    def compile_solver(self):
        self.solver_function = self.opti.to_function(
            "compiled_solver",
            self.solver_params,
            [self.opti.x] # , self.opti.lam_g]  # output
        )
        self.solver_function.generate("compiled_solver.c")

    def solve(self, retract_all=True):
        if self.solver_type == "fatrop":
            try:
                self.sol = self.opti.solve()
            except:  # noqa: E722
                self.sol = self.opti.debug

            self.solve_time = self.sol.stats()["t_wall_total"]

            # TODO: Check solution status
            self.retract_opti_sol(retract_all)

            # Store dual variables
            self.lam_g = self.sol.value(self.opti.lam_g)

        elif self.solver_type == "osqp":
            # Initial guess
            current_x = self.opti.value(self.opti.x, self.opti.initial())
            start_time = time.time()

            # TODO: Termination condition
            for _ in range(3):
                # OSQP
                hess_f, grad_f, J_g, g, lbg, ubg = self.sqp_data(current_x, self.solver_params)
                P = sparse.csc_matrix(hess_f)
                A = sparse.csc_matrix(J_g)
                q = np.array(grad_f)
                l = np.array(lbg - g)
                u = np.array(ubg - g)

                # TODO: Possibly use update
                self.osqp_prob = osqp.OSQP()
                self.osqp_prob.setup(P, q, A, l, u, **self.osqp_opts)
                sol_dx = self.osqp_prob.solve().x

                # Armijo line search
                current_x = self._armijo_line_search(
                    current_x=current_x,
                    dx=sol_dx,
                )

            end_time = time.time()
            self.solve_time = end_time - start_time

            self.retract_stacked_sol(current_x, retract_all)

    def retract_opti_sol(self, retract_all=True):
        pass

    def retract_stacked_sol(self, sol_x, retract_all=True):
        pass

    def _armijo_line_search(self, current_x, dx):
        # Params
        armijo_factor = 1e-4
        a = 1.0
        a_min = 1e-4
        a_decay = 0.5
        g_max = 1e-3
        g_min = 1e-5
        gamma = 1e-5

        # Get current data
        f, grad_f = self.f_data(current_x, self.solver_params)
        g, lbg, ubg = self.g_data(current_x, self.solver_params)

        g_metric = self._constraint_metric(g, lbg, ubg)
        armijo_metric = grad_f.T @ dx
        accepted = False

        while not accepted and a > a_min:
            # Evaluate new solution
            new_x = current_x + a * dx
            new_f, _ = self.f_data(new_x, self.solver_params)
            new_g, lbg, ubg = self.g_data(new_x, self.solver_params)

            new_g_metric = self._constraint_metric(new_g, lbg, ubg)
            if new_g_metric > g_max:
                if new_g_metric < (1 - gamma) * g_metric:
                    print("Line search: g metric high, but improving")
                    accepted = True
    
            elif max(new_g_metric, g_metric) < g_min and armijo_metric < 0:
                if new_f <= f + armijo_factor * armijo_metric:
                    print("Line search: g metric low, f improving")
                    accepted = True

            elif new_f <= f - gamma * new_g_metric or new_g_metric < (1 - gamma) * g_metric:
                print("Line search: f improving or g metric improving")
                accepted = True

            a *= a_decay  # Reduce step size
            f = new_f
            g_metric = new_g_metric

        if accepted:
            # Print info (need to adjust a)
            print(f"a: {a / a_decay}, f: {f}, g metric: {g_metric}")
            return new_x

        else:
            print("Line search: Didn't converge!")
            return current_x

    def _constraint_metric(self, g, lbg, ubg):
        lb_violations = np.maximum(0, lbg - g)
        ub_violations = np.maximum(0, g - ubg)
        violations = np.concatenate((lb_violations, ub_violations))

        metric = np.linalg.norm(violations)
        return metric
