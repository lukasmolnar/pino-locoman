import time
import numpy as np
import casadi as ca
from scipy import sparse

from helpers import *
from dynamics_centroidal import DynamicsCentroidal


class OCP_Centroidal:
    def __init__(
            self,
            robot,
            nodes,
            step_height=0.1,
            mu=0.7,
        ):
        self.opti = ca.Opti()
        self.model = robot.model
        self.data = robot.data
        self.gait_sequence = robot.gait_sequence
        self.nodes = nodes

        self.mass = self.data.mass[0]
        self.dt = self.gait_sequence.dt
        self.ee_ids = robot.ee_ids
        self.arm_ee_id = robot.arm_ee_id
        self.n_feet = len(self.ee_ids)

        # State and inputs to optimize
        self.dx_opt_indices = robot.dx_opt_indices
        self.ndx_opt = len(self.dx_opt_indices)
        self.nu_opt = robot.nf + robot.nj  # forces + joint velocities

        # Dynamics
        self.dyn = DynamicsCentroidal(self.model, self.mass, self.ee_ids, self.dx_opt_indices)

        # Decision variables
        self.DX_opt = []
        self.U_opt = []
        for i in range(self.nodes):
            self.DX_opt.append(self.opti.variable(self.ndx_opt))
            self.U_opt.append(self.opti.variable(self.nu_opt))
        self.DX_opt.append(self.opti.variable(self.ndx_opt))

        # Parameters
        self.x_init = self.opti.parameter(robot.nx)  # initial state
        self.contact_schedule = self.opti.parameter(self.n_feet, self.nodes) # in_contact: 0 or 1
        self.swing_schedule = self.opti.parameter(self.n_feet, self.nodes) # swing_phase: from 0 to 1
        self.n_contacts = self.opti.parameter(1)  # number of contact feet
        self.com_goal = self.opti.parameter(6)  # linear + angular momentum
        self.swing_height = self.opti.parameter(1)  # max swing height
        self.swing_vel_limits = self.opti.parameter(2)  # start and end swing velocities
        self.arm_f_des = self.opti.parameter(3)  # force at end-effector
        self.arm_vel_des = self.opti.parameter(3)  # velocity at end-effector

        self.Q_diag = self.opti.parameter(self.ndx_opt)  # state weights
        self.R_diag = self.opti.parameter(self.nu_opt)  # input weights
        self.Q = ca.diag(self.Q_diag)
        self.R = ca.diag(self.R_diag)

        # Desired state and input
        x_des = ca.vertcat(self.com_goal, robot.q0)
        dx_des = self.dyn.state_difference()(self.x_init, x_des)
        dx_des = dx_des[self.dx_opt_indices]
        f_des = ca.repmat(ca.vertcat(0, 0, 9.81 * self.mass / self.n_contacts), self.n_feet, 1)  # gravity compensation
        if self.arm_ee_id:
            f_des = ca.vertcat(f_des, [0] * 3)  # zero force at end-effector
        u_des = ca.vertcat(f_des, [0] * robot.nj)  # zero joint velocities

        # OBJECTIVE
        obj = 0

        # Tracking and regularization
        for i in range(self.nodes):
            dx = self.DX_opt[i]
            u = self.U_opt[i]
            err_dx = dx - dx_des
            err_u = u - u_des
            obj += 0.5 * err_dx.T @ self.Q @ err_dx
            obj += 0.5 * err_u.T @ self.R @ err_u

        # Final state
        dx = self.DX_opt[self.nodes]
        err_dx = dx - dx_des
        obj += 0.5 * err_dx.T @ self.Q @ err_dx

        # CONSTRAINTS
        self.opti.subject_to(self.DX_opt[0] == [0] * self.ndx_opt)

        for i in range(self.nodes):
            # Gather all state and input info
            dx = self.DX_opt[i]
            x = self.dyn.state_integrate()(self.x_init, dx)
            u = self.U_opt[i]
            h = x[:6]
            q = x[6:]
            forces = u[:robot.nf]
            dq_j = u[robot.nf:]
            dq_b = self.dyn.get_base_velocity()(h, q, dq_j)
            dq = ca.vertcat(dq_b, dq_j)

            # Dynamics constraint
            dx_next = self.DX_opt[i+1]
            dx_dyn = self.dyn.centroidal_dynamics(self.arm_ee_id)(x, u, dq_b)
            dx_dyn = dx_dyn[self.dx_opt_indices]
            self.opti.subject_to(dx_next == dx + dx_dyn * self.dt)

            # Contact and swing constraints
            for idx, frame_id in enumerate(self.ee_ids):
                f_e = forces[idx * 3 : (idx + 1) * 3]

                # Determine contact and bezier index from schedule
                in_contact = self.contact_schedule[idx, i]
                swing_phase = self.swing_schedule[idx, i]

                # Friction cone
                f_normal = f_e[2]
                f_tangent_square = f_e[0]**2 + f_e[1]**2
                self.opti.subject_to(in_contact * f_normal >= 0)
                self.opti.subject_to(in_contact * mu**2 * f_normal**2 >= in_contact * f_tangent_square)

                # Zero end-effector velocity (linear)
                vel = self.dyn.get_frame_velocity(frame_id)(q, dq)
                vel_lin = vel[:3]
                self.opti.subject_to(in_contact * vel_lin == [0] * 3)

                # Zero end-effector force
                self.opti.subject_to((1 - in_contact) * f_e == [0] * 3)

                # Track bezier velocity (only in z)
                vel_z = vel_lin[2]
                vel_z_des = self.gait_sequence.get_spline_vel_z(
                    swing_phase,
                    h_max=self.swing_height,
                    v_liftoff=self.swing_vel_limits[0],
                    v_touchdown=self.swing_vel_limits[1]    
                )
                vel_diff = vel_z - vel_z_des
                self.opti.subject_to((1 - in_contact) * vel_diff == 0)

            # Arm task
            if self.arm_ee_id:
                # Zero end-effector velocity (linear and angular)
                vel = self.dyn.get_frame_velocity(self.arm_ee_id)(q, dq)
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
            self.opti.subject_to(self.opti.bounded(pos_min, q_j, pos_max))
            self.opti.subject_to(self.opti.bounded(vel_min, dq_j, vel_max))

            # Warm start: Use n_contacts from gait sequence for u_des
            self.opti.set_value(self.n_contacts, self.gait_sequence.n_contacts)
            self.opti.set_initial(self.DX_opt[i], np.zeros(self.ndx_opt))
            self.opti.set_initial(self.U_opt[i], self.opti.value(u_des))

        # Warm start
        self.opti.set_initial(self.DX_opt[self.nodes], np.zeros(self.ndx_opt))

        # OBJECTIVE
        self.opti.minimize(obj)

        # TODO: Clean up how solutions are stored
        self.DX_prev = None
        self.U_prev = None
        self.lam_g = None
        self.hs = []
        self.qs = []
        self.us = []

    def update_initial_state(self, x_init):
        self.opti.set_value(self.x_init, x_init)

    def update_gait_sequence(self, shift_idx=0):
        contact_schedule = self.gait_sequence.shift_contact_schedule(shift_idx)
        swing_schedule = self.gait_sequence.shift_swing_schedule(shift_idx)
        n_contacts = self.gait_sequence.n_contacts
        self.opti.set_value(self.contact_schedule, contact_schedule[:, :self.nodes])
        self.opti.set_value(self.swing_schedule, swing_schedule[:, :self.nodes])
        self.opti.set_value(self.n_contacts, n_contacts)

    def set_com_goal(self, com_goal):
        self.opti.set_value(self.com_goal, com_goal)

    def set_swing_params(self, swing_height, swing_vel_limits):
        self.opti.set_value(self.swing_height, swing_height)
        self.opti.set_value(self.swing_vel_limits, swing_vel_limits)

    def set_arm_task(self, arm_f_des, arm_vel_des):
        self.opti.set_value(self.arm_f_des, arm_f_des)
        self.opti.set_value(self.arm_vel_des, arm_vel_des)

    def set_weights(self, Q_diag, R_diag):
        self.opti.set_value(self.Q_diag, Q_diag)
        self.opti.set_value(self.R_diag, R_diag)

    def warm_start(self):
        # Shift previous solution
        # NOTE: No warm-start for last node, copying the 2nd last node performs worse.
        if self.DX_prev is not None:
            DX_init = self.DX_prev[1]
            for i in range(self.nodes):
                DX_diff = self.DX_prev[i+1] - DX_init
                self.opti.set_initial(self.DX_opt[i], DX_diff)

        if self.U_prev is not None:
            for i in range(self.nodes - 1):
                self.opti.set_initial(self.U_opt[i], self.U_prev[i+1])

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
                # "constr_viol_tol": 1e-2,
            }
            self.opti.solver(solver, opts)

            # Code generation
            if compile_solver:
                solver_params = [
                    self.x_init,
                    self.contact_schedule,
                    self.swing_schedule,
                    self.n_contacts,
                    self.Q_diag,
                    self.R_diag,
                    self.com_goal,
                    self.swing_height,
                    self.swing_vel_limits
                ]
                if self.arm_ee_id:
                    solver_params += [
                        self.arm_f_des,
                        self.arm_vel_des
                    ]
                if warm_start:
                    solver_params += [
                        self.opti.x,  # initial guess
                        # self.opti.lam_g,  # dual variables
                    ]
                self.solver_function = self.opti.to_function(
                    "compiled_solver",
                    solver_params,
                    [self.opti.x] # , self.opti.lam_g]  # output
                )
                self.solver_function.generate("compiled_solver.c")

        elif solver == "osqp":
            import osqp
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
            self.sqp_data = ca.Function("sqp_data", [x, p], [hess_f, grad_f, J_g, g, lbg, ubg]).expand()
            self.f_data = ca.Function("f_data", [x, p], [f, grad_f]).expand()
            self.g_data = ca.Function("g_data", [x, p], [g, lbg, ubg]).expand()

            # OSQP options
            self.osqp_opts = {
                "max_iter": 100,
                "alpha": 1.4,
                "rho": 2e-2,
            }

            # OSQP formulation with dummy data
            P = sparse.csc_matrix(np.eye(hess_f.shape[0]))  # diagonal
            A = sparse.csc_matrix(np.ones(J_g.shape))  # dense
            q = np.ones(grad_f.shape)
            l = -np.ones(g.shape)
            u = np.ones(g.shape)

            self.osqp_prob = osqp.OSQP()
            self.osqp_prob.setup(P, q, A, l, u, **self.osqp_opts)
            # self.osqp_prob.codegen(
            #     "codegen/osqp",
            #     parameters="matrices",
            # )

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
                ca.vec(self.opti.value(self.contact_schedule)),  # flattened
                self.opti.value(self.com_goal),
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

            # TODO: How many iterations?
            for _ in range(3):
                # OSQP
                hess_f, grad_f, J_g, g, lbg, ubg = self.sqp_data(current_x, ocp_params)
                q = np.array(grad_f)
                l = np.array(lbg - g)
                u = np.array(ubg - g)

                # Redifine OSQP
                P = sparse.csc_matrix(hess_f)
                A = sparse.csc_matrix(J_g)
                self.osqp_prob = osqp.OSQP()
                self.osqp_prob.setup(P, q, A, l, u, **self.osqp_opts)

                # Update OSQP
                # P_data = np.diag(hess_f)  # diagonal
                # A_data = np.array(J_g).flatten(order="F")  # dense, flattened columns
                # self.osqp_prob.update(Px=P_data, q=q, Ax=A_data, l=l, u=u)

                sol_dx = self.osqp_prob.solve().x

                # Armijo line search
                current_x = self._armijo_line_search(
                    current_x=current_x,
                    dx=sol_dx,
                    ocp_params=ocp_params,
                )

            end_time = time.time()
            self.solve_time = end_time - start_time

            self._retract_sqp_sol(current_x, retract_all)

    def _retract_opti_sol(self, retract_all=True):
        # Retract self.opti solution stored in self.sol
        self.DX_prev = [self.sol.value(dx) for dx in self.DX_opt]
        self.U_prev = [self.sol.value(u) for u in self.U_opt]
        x_init = self.opti.value(self.x_init)

        for dx_sol, u_sol in zip(self.DX_prev, self.U_prev):
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            self.hs.append(np.array(x_sol[:6]))
            self.qs.append(np.array(x_sol[6:]))
            self.us.append(np.array(u_sol))

            if not retract_all:
                return

    def _retract_stacked_sol(self, sol_x, retract_all=True):
        # Retract the given solution, which contains all stacked states and inputs
        self.DX_prev = []
        self.U_prev = []
        x_init = self.opti.value(self.x_init)
        nx_opt = self.ndx_opt + self.nu_opt

        for i in range(self.nodes):
            sol = sol_x[i*nx_opt : (i+1)*nx_opt]
            dx_sol = sol[:self.ndx_opt]
            u_sol = sol[self.ndx_opt:]
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            self.DX_prev.append(np.array(dx_sol))
            self.U_prev.append(np.array(u_sol))

            if i == 0 or retract_all:
                self.hs.append(np.array(x_sol[:6]))
                self.qs.append(np.array(x_sol[6:]))
                self.us.append(np.array(u_sol))

        dx_last = sol_x[self.nodes*nx_opt:]
        x_last = self.dyn.state_integrate()(x_init, dx_last)
        self.DX_prev.append(np.array(dx_last))

        if retract_all:
            self.hs.append(np.array(x_last[:6]))
            self.qs.append(np.array(x_last[6:]))

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
        # TODO: Look into handling equality constraints differently
        lb_violations = np.maximum(0, lbg - g)
        ub_violations = np.maximum(0, g - ubg)
        violations = np.concatenate((lb_violations, ub_violations))

        metric = np.linalg.norm(violations)
        return metric
