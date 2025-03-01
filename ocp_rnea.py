import time
import numpy as np
import casadi as ca
import pinocchio as pin
from scipy import sparse

from helpers import *
from dynamics_rnea import DynamicsRNEA


class OCP_RNEA:
    def __init__(
            self,
            robot,
            nodes,
            tau_nodes=2,
            mu=0.7,
        ):
        self.opti = ca.Opti()
        self.model = robot.model
        self.data = robot.data
        self.gait_sequence = robot.gait_sequence
        self.nq = robot.nq
        self.nv = robot.nv
        self.nf = robot.nf
        self.nj = robot.nj

        self.nodes = nodes
        self.tau_nodes = tau_nodes
        self.mass = self.data.mass[0]
        self.ee_ids = robot.ee_ids
        self.arm_ee_id = robot.arm_ee_id
        self.n_feet = len(self.ee_ids)

        # State and inputs to optimize
        self.nx = self.nq + self.nv  # positions + velocities
        self.ndx_opt = self.nv * 2  # position deltas + velocities
        self.nu_opt = (
            [self.nv + self.nf + self.nj] * self.tau_nodes +  # accelerations + forces + torques
            [self.nv + self.nf] * (self.nodes - self.tau_nodes)  # accelerations + forces
        )

        # Dynamics
        self.dyn = DynamicsRNEA(self.model, self.mass, self.ee_ids)

        # Decision variables
        self.DX_opt = []
        self.U_opt = []
        for i in range(self.nodes):
            self.DX_opt.append(self.opti.variable(self.ndx_opt))
            self.U_opt.append(self.opti.variable(self.nu_opt[i]))
        self.DX_opt.append(self.opti.variable(self.ndx_opt))

        # Parameters
        self.x_init = self.opti.parameter(self.nx)  # initial state
        self.tau_prev = self.opti.parameter(self.nj)  # previous joint torques

        self.dt_min = self.opti.parameter(1)  # first time step size (used for sim)
        self.dt_max = self.opti.parameter(1)  # last time step size
        self.contact_schedule = self.opti.parameter(self.n_feet, self.nodes) # in_contact: 0 or 1
        self.swing_schedule = self.opti.parameter(self.n_feet, self.nodes) # swing_phase: from 0 to 1
        self.n_contacts = self.opti.parameter(1)  # number of contact feet
        self.swing_period = self.opti.parameter(1)  # swing period (in seconds)
        self.swing_height = self.opti.parameter(1)  # max swing height
        self.swing_vel_limits = self.opti.parameter(2)  # start and end swing velocities

        self.Q_diag = self.opti.parameter(self.ndx_opt)  # state weights
        self.R_diag = self.opti.parameter(self.nu_opt[0])  # input weights (with torques)
        self.W_diag = self.opti.parameter(self.nj)  # additional weights for joint torques

        self.base_vel_des = self.opti.parameter(6)  # linear + angular velocity
        self.arm_f_des = self.opti.parameter(3)  # force at end-effector
        self.arm_vel_des = self.opti.parameter(3)  # linear velocity at end-effector

        # Increasing time step sizes
        ratio = self.dt_max / self.dt_min
        gamma = ratio ** (1 / (self.nodes - 1))  # growth factor
        self.dts = [self.dt_min * gamma**i for i in range(self.nodes)]

        # Desired state
        v_des = ca.vertcat(self.base_vel_des, [0] * self.nj)
        x_des = ca.vertcat(robot.q0, v_des)
        self.dx_des = self.dyn.state_difference()(self.x_init, x_des)  # stay close to nominal state

        # Desired input: Use this for warm starting
        self.f_des = ca.repmat(ca.vertcat(0, 0, 9.81 * self.mass / self.n_contacts), self.n_feet, 1)  # gravity compensation
        if self.arm_ee_id:
            self.f_des = ca.vertcat(self.f_des, [0] * 3)  # zero force at end-effector
        self.u_des = ca.vertcat([0] * self.nv, self.f_des, [0] * self.nj)  # zero acc + torque

        # OBJECTIVE
        obj = 0
        Q = ca.diag(self.Q_diag)
        R = ca.diag(self.R_diag)
        W = ca.diag(self.W_diag)
        for i in range(self.nodes):
            # Track desired state and input
            dx = self.DX_opt[i]
            u = self.U_opt[i]
            if i >= self.tau_nodes:
                u = ca.vertcat(u, [0] * self.nj)  # zero torques

            err_dx = dx - self.dx_des
            err_u = u - self.u_des
            obj += err_dx.T @ Q @ err_dx
            obj += err_u.T @ R @ err_u

        # Keep torque close to previous solution
        tau_0 = self.U_opt[0][self.nv + self.nf :]
        tau_des = self.tau_prev
        err_tau = tau_0 - tau_des
        obj += err_tau.T @ W @ err_tau

        # Final state
        dx = self.DX_opt[self.nodes]
        err_dx = dx - self.dx_des
        obj += err_dx.T @ Q @ err_dx

        # CONSTRAINTS
        self.opti.subject_to(self.DX_opt[0] == [0] * self.ndx_opt)  # initial state

        for i in range(self.nodes):
            # Gather all state and input info
            dx = self.DX_opt[i]
            dq = dx[:self.nv]  # delta q, not v
            dv = dx[self.nv:]  # delta v
            x = self.dyn.state_integrate()(self.x_init, dx)
            u = self.U_opt[i]
            q = x[:self.nq]
            v = x[self.nq:]
            a = u[:self.nv]
            forces = u[self.nv : self.nv + self.nf]
            dt = self.dts[i]

            # Dynamics constraint
            dx_next = self.DX_opt[i+1]
            dq_next = dx_next[:self.nv]
            dv_next = dx_next[self.nv:]
            self.opti.subject_to(dq_next == dq + v * dt + 0.5 * a * dt**2)
            self.opti.subject_to(dv_next == dv + a * dt)

            # RNEA constraint
            tau_rnea = self.dyn.rnea_dynamics(self.arm_ee_id)(q, v, a, forces)
            self.opti.subject_to(tau_rnea[:6] == [0] * 6)  # base

            # Torque constraints
            if i < self.tau_nodes:
                tau_j = u[self.nv + self.nf :]
                self.opti.subject_to(tau_rnea[6:] == tau_j)  # joints

                # Torque limits
                tau_min = -robot.joint_torque_max
                tau_max = robot.joint_torque_max
                self.opti.subject_to(self.opti.bounded(tau_min, tau_j, tau_max))

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

                if i == 0:
                    # No state constraints at first time step
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

            if i == 0:
                # No state constraints at first time step
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
            pos_min = robot.joint_pos_min
            pos_max = robot.joint_pos_max
            vel_min = -robot.joint_vel_max
            vel_max = robot.joint_vel_max
            q_j = q[7:]  # skip base quaternion
            v_j = v[6:]  # skip base angular velocity
            self.opti.subject_to(self.opti.bounded(pos_min, q_j, pos_max))
            self.opti.subject_to(self.opti.bounded(vel_min, v_j, vel_max))

        # Warm start
        self.opti.set_initial(self.DX_opt[self.nodes], np.zeros(self.ndx_opt))

        # OBJECTIVE
        self.opti.minimize(obj)

        # TODO: Clean up how solutions are stored
        self.DX_prev = None
        self.U_prev = None
        self.lam_g = None
        self.qs = []
        self.vs = []
        self.accs = []
        self.forces = []
        self.taus = []

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

    def set_weights(self, Q_diag, R_diag, W_diag):
        self.opti.set_value(self.Q_diag, Q_diag)
        self.opti.set_value(self.R_diag, R_diag)
        self.opti.set_value(self.W_diag, W_diag)

    def update_initial_state(self, x_init):
        self.opti.set_value(self.x_init, x_init)

    def update_previous_torques(self, tau_prev):
        self.opti.set_value(self.tau_prev, tau_prev)

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
        # TODO: Look into interpolating
        if self.DX_prev is not None:
            for i in range(self.nodes + 1):
                # Previous solution for dx
                dx_prev = self.DX_prev[i]
                self.opti.set_initial(self.DX_opt[i], dx_prev)
                continue

        if self.U_prev is not None:
            for i in range(self.nodes):
                # Previous solution for acc, tau
                # Tracking target for f (gravity compensation)
                u_prev = self.U_prev[i]
                a_prev = u_prev[:self.nv]
                f_des = self.opti.value(self.f_des)
                u_warm = ca.vertcat(a_prev, f_des)
                if i < self.tau_nodes:
                    tau_prev = u_prev[self.nv + self.nf :]
                    u_warm = ca.vertcat(u_warm, tau_prev)
                self.opti.set_initial(self.U_opt[i], u_warm)

        if self.lam_g is not None:
            self.opti.set_initial(self.opti.lam_g, self.lam_g)

    def init_solver(self, solver="fatrop", compile_solver=False, warm_start=False):
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

            # Code generation
            if compile_solver:
                solver_params = [self.x_init, self.tau_prev, self.dt_min, self.dt_max, self.contact_schedule,
                                 self.swing_schedule, self.n_contacts, self.swing_period, self.swing_height, 
                                 self.swing_vel_limits, self.Q_diag, self.R_diag, self.W_diag, self.base_vel_des]
                if self.arm_ee_id:
                    solver_params += [self.arm_f_des, self.arm_vel_des]
                if warm_start:
                    solver_params += [self.opti.x]

                self.solver_function = self.opti.to_function(
                    "compiled_solver",
                    solver_params,
                    [self.opti.x] # , self.opti.lam_g]  # output
                )
                self.solver_function.generate("compiled_solver.c")

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

        else:
            raise ValueError(f"Solver {solver} not supported")

    def solve(self, retract_all=True):
        if self.solver_type == "fatrop":
            try:
                self.sol = self.opti.solve()
            except:  # noqa: E722
                self.sol = self.opti.debug

            self.solve_time = self.sol.stats()["t_wall_total"]

            # TODO: Check solution status
            self._retract_opti_sol(retract_all)

            # Store dual variables
            self.lam_g = self.sol.value(self.opti.lam_g)

        elif self.solver_type == "osqp":
            import osqp
            ocp_params = ca.vertcat(
                self.opti.value(self.x_init),
                self.opti.value(self.tau_prev),
                ca.vec(self.opti.value(self.contact_schedule)),  # flattened
                ca.vec(self.opti.value(self.swing_schedule)),  # flattened
                self.opti.value(self.n_contacts),
                self.opti.value(self.swing_period),
                self.opti.value(self.swing_height),
                self.opti.value(self.swing_vel_limits),
                self.opti.value(self.Q_diag),
                self.opti.value(self.R_diag),
                self.opti.value(self.W_diag),
                self.opti.value(self.base_vel_des),
            )
            if self.arm_ee_id:
                ocp_params = ca.vertcat(
                    ocp_params,
                    self.opti.value(self.arm_f_des),
                    self.opti.value(self.arm_vel_des),
                )

            # Initial guess
            current_x = self.opti.value(self.opti.x, self.opti.initial())
            start_time = time.time()

            # TODO: Termination condition
            for _ in range(3):
                # OSQP
                hess_f, grad_f, J_g, g, lbg, ubg = self.sqp_data(current_x, ocp_params)
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
                    ocp_params=ocp_params,
                )

            end_time = time.time()
            self.solve_time = end_time - start_time

            self._retract_stacked_sol(current_x, retract_all)

    def _retract_opti_sol(self, retract_all=True):
        # Retract self.opti solution stored in self.sol
        self.DX_prev = [self.sol.value(dx) for dx in self.DX_opt]
        self.U_prev = [self.sol.value(u) for u in self.U_opt]
        self.lam_g = self.sol.value(self.opti.lam_g)
        x_init = self.opti.value(self.x_init)

        for dx_sol, u_sol in zip(self.DX_prev, self.U_prev):
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            self.qs.append(np.array(x_sol[:self.nq]))
            self.vs.append(np.array(x_sol[self.nq:]))
            self.accs.append(np.array(u_sol[:self.nv]))
            self.forces.append(np.array(u_sol[self.nv : self.nv + self.nf]))
            self.taus.append(np.array(u_sol[self.nv + self.nf :]))

            if not retract_all:
                return
            
        x_last = self.dyn.state_integrate()(x_init, self.DX_prev[-1])
        self.qs.append(np.array(x_last[:self.nq]))
        self.vs.append(np.array(x_last[self.nq:]))

    def _retract_stacked_sol(self, sol_x, retract_all=True):
        # Retract the given solution, which contains all stacked states and inputs
        self.DX_prev = []
        self.U_prev = []
        x_init = self.opti.value(self.x_init)
        idx = 0

        for i in range(self.nodes):
            nx_opt = self.ndx_opt + self.nu_opt[i]
            sol = sol_x[idx : idx + nx_opt]
            idx += nx_opt

            dx_sol = sol[:self.ndx_opt]
            u_sol = sol[self.ndx_opt:]
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            self.DX_prev.append(np.array(dx_sol))
            self.U_prev.append(np.array(u_sol))
            
            if i == 0 or retract_all:
                self.qs.append(np.array(x_sol[:self.nq]))
                self.vs.append(np.array(x_sol[self.nq:]))
                self.accs.append(np.array(u_sol[:self.nv]))
                self.forces.append(np.array(u_sol[self.nv : self.nv + self.nf]))
                self.taus.append(np.array(u_sol[self.nv + self.nf :]))

        dx_last = sol_x[-self.ndx_opt:]
        x_last = self.dyn.state_integrate()(x_init, dx_last)
        self.DX_prev.append(np.array(dx_last))

        if retract_all:
            self.qs.append(np.array(x_last[:self.nq]))
            self.vs.append(np.array(x_last[self.nq:]))

    def _get_torque_sol(self, idx):
        u_sol = self.U_prev[idx]
        tau_sol = u_sol[self.nv + self.nf :]
        return tau_sol

    def _simulate_step(self, x_init, u):
        q = x_init[:self.nq]
        v = x_init[self.nq:]
        forces = u[self.nv : self.nv + self.nf]
        tau_j = u[self.nv + self.nf :]
        dt_sim = self.opti.value(self.dt_min)  # the first step size

        pin.framesForwardKinematics(self.model, self.data, q)
        f_ext = [pin.Force(np.zeros(6)) for _ in range(self.model.njoints)]
        for idx, frame_id in enumerate(self.ee_ids):
            joint_id = self.model.frames[frame_id].parentJoint
            translation_joint_to_contact_frame = self.model.frames[frame_id].placement.translation
            rotation_world_to_joint_frame = self.data.oMi[joint_id].rotation.T
            f_world = forces[idx * 3 : (idx + 1) * 3].flatten()

            f_lin = rotation_world_to_joint_frame @ f_world
            f_ang = np.cross(translation_joint_to_contact_frame, f_lin)
            f = np.concatenate((f_lin, f_ang))
            f_ext[joint_id] = pin.Force(f)

        tau = np.concatenate((np.zeros(6), tau_j.flatten()))
        a = pin.aba(self.model, self.data, q, v, tau, f_ext)

        dq = v * dt_sim + 0.5 * a * dt_sim**2
        dv = a * dt_sim

        q_next = pin.integrate(self.model, q, dq)
        v_next = v + dv
        x_next = np.concatenate((q_next, v_next))

        return x_next

    def _armijo_line_search(self, current_x, dx, ocp_params):
        # Params
        armijo_factor = 1e-4
        a = 1.0
        a_min = 1e-4
        a_decay = 0.5
        g_max = 1e-3
        g_min = 1e-5
        gamma = 1e-5

        # Get current data
        f, grad_f = self.f_data(current_x, ocp_params)
        g, lbg, ubg = self.g_data(current_x, ocp_params)

        g_metric = self._constraint_metric(g, lbg, ubg)
        armijo_metric = grad_f.T @ dx
        accepted = False

        while not accepted and a > a_min:
            # Evaluate new solution
            new_x = current_x + a * dx
            new_f, _ = self.f_data(new_x, ocp_params)
            new_g, lbg, ubg = self.g_data(new_x, ocp_params)

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
