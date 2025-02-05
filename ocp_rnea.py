import numpy as np
import casadi
import time
import osqp
from scipy import sparse

from helpers import *
from dynamics_rnea import DynamicsRNEA


class OCP_RNEA:
    def __init__(
            self,
            robot,
            nodes,
            step_height=0.1,
            mu=0.7,
        ):
        self.opti = casadi.Opti()
        self.model = robot.model
        self.data = robot.data
        self.gait_sequence = robot.gait_sequence
        self.nq = robot.nq
        self.nv = robot.nv
        self.nf = robot.nf
        self.nj = robot.nj

        self.nodes = nodes
        self.mass = self.data.mass[0]
        self.dt = self.gait_sequence.dt
        self.ee_ids = robot.ee_ids
        self.arm_ee_id = robot.arm_ee_id
        self.n_feet = len(self.ee_ids)
        self.n_contact_feet = self.gait_sequence.n_contacts

        # State and inputs to optimize
        self.nx = self.nq + self.nv  # positions + velocities
        self.ndx_opt = self.nv * 2  # position deltas + velocities
        self.nu_opt = self.nv + self.nj + self.nf  # velocities + joint torques + forces

        # Dynamics
        self.dyn = DynamicsRNEA(self.model, self.mass, self.ee_ids)

        # Decision variables
        self.DX_opt = []
        self.U_opt = []
        for i in range(self.nodes):
            self.DX_opt.append(self.opti.variable(self.ndx_opt))
            self.U_opt.append(self.opti.variable(self.nu_opt))
        self.DX_opt.append(self.opti.variable(self.ndx_opt))

        # Parameters
        self.x_init = self.opti.parameter(self.nx)  # initial state
        self.contact_schedule = self.opti.parameter(self.n_feet, self.nodes)  # gait schedule: contacts / bezier indices
        self.com_goal = self.opti.parameter(6)  # linear + angular momentum
        self.arm_f_des = self.opti.parameter(3)  # force at end-effector
        self.arm_vel_des = self.opti.parameter(3)  # velocity at end-effector

        self.Q_diag = self.opti.parameter(self.ndx_opt)  # state weights
        self.R_diag = self.opti.parameter(self.nu_opt)  # input weights
        self.Q = casadi.diag(self.Q_diag)
        self.R = casadi.diag(self.R_diag)

        # Desired state and input
        v_des = casadi.vertcat(self.com_goal, [0] * self.nj)
        x_des = casadi.vertcat(robot.q0, v_des)
        dx_des = self.dyn.state_difference()(self.x_init, x_des)
        f_des = np.tile([0, 0, 9.81 * self.mass / self.n_contact_feet], self.n_feet)
        if self.arm_ee_id:
            # TODO: Check if arm force = 0 is ok (it is constrained later)
            f_des = np.concatenate((f_des, np.zeros(3)))
        u_des = np.concatenate((np.zeros(robot.nv + robot.nj), f_des))  # zero velocity + torque

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
        # self.opti.subject_to(self.DX_opt[0][:self.nv] == [0] * self.nv)  # initial pos
        self.opti.subject_to(self.DX_opt[0] == [0] * 2 * self.nv)  # initial pos + vel

        for i in range(self.nodes):
            # Gather all state and input info
            dx_opt = self.DX_opt[i]
            dq = dx_opt[:self.nv]  # delta q, not v
            x = self.dyn.state_integrate()(self.x_init, dx_opt)
            u = self.U_opt[i]
            q = x[:self.nq]
            v = x[self.nq:]
            v_next_des = u[:self.nv]
            tau_j = u[self.nv : self.nv + self.nj]
            forces = u[self.nv + self.nj :]

            # Dynamics constraint
            dx_next = self.DX_opt[i+1]
            dq_next = dx_next[:self.nv]
            x_next = self.dyn.state_integrate()(self.x_init, dx_next)
            v_next = x_next[self.nq:]
            self.opti.subject_to(dq_next == dq + 0.5 * (v + v_next_des) * self.dt)
            self.opti.subject_to(v_next == v_next_des)

            # RNEA constraint
            a = (v_next_des - v) / self.dt  # finite difference
            tau_rnea = self.dyn.rnea_dynamics(self.arm_ee_id)(q, v, a, forces)
            self.opti.subject_to(tau_rnea[:6] == [0] * 6)  # base
            if i < 2:
                # Joint torques only for first two nodes
                self.opti.subject_to(tau_rnea[6:] == tau_j)  # joints

            # Contact and swing constraints
            for idx, frame_id in enumerate(self.ee_ids):
                f_e = forces[idx * 3 : (idx + 1) * 3]

                # Determine contact and bezier index from schedule
                schedule_idx = self.contact_schedule[idx, i]
                in_contact = casadi.if_else(schedule_idx == 0, 1, 0)
                bezier_idx = schedule_idx - 1

                # Friction cone
                f_normal = f_e[2]
                f_tangent_square = f_e[0]**2 + f_e[1]**2
                self.opti.subject_to(in_contact * f_normal >= 0)
                self.opti.subject_to(in_contact * mu**2 * f_normal**2 >= in_contact * f_tangent_square)

                # Zero end-effector force
                self.opti.subject_to((1 - in_contact) * f_e == [0] * 3)

                if i == 0:
                    # No state constraints at first time step
                    continue

                # Zero end-effector velocity (linear)
                vel = self.dyn.get_frame_velocity(frame_id)(q, v)
                vel_lin = vel[:3]
                self.opti.subject_to(in_contact * vel_lin == [0] * 3)

                # Track bezier velocity (only in z)
                vel_z = vel_lin[2]
                vel_z_des = self.gait_sequence.get_bezier_vel_z(0, bezier_idx, h=step_height)
                vel_diff = vel_z - vel_z_des
                self.opti.subject_to((1 - in_contact) * vel_diff == 0)

                # pos_z = self.dyn.get_frame_position(frame_id)(q)[2]
                # pos_z_des = self.gait_sequence.get_bezier_pos_z(0, bezier_idx, h=step_height)
                # pos_diff = pos_z - pos_z_des
                # self.opti.subject_to((1 - in_contact) * pos_diff == 0)

                # Humanoid
                if self.n_feet == 2:
                    # Flat feet
                    vel_ang = vel[3:]
                    self.opti.subject_to(vel_ang == [0] * 3)

            # Humanoid
            if self.n_feet == 2:
                # Minimum distance between feet
                min_dist = 0.095  # standing pose is 0.1
                foot_0 = self.dyn.get_frame_position(self.ee_ids[0])(q)
                foot_1 = self.dyn.get_frame_position(self.ee_ids[1])(q)
                foot_diff = foot_1 - foot_0
                self.opti.subject_to(foot_diff[0]**2 + foot_diff[1]**2 >= min_dist**2)

            # Arm task
            if self.arm_ee_id:
                # Zero end-effector velocity (linear and angular)
                vel = self.dyn.get_frame_velocity(self.arm_ee_id)(q, v)
                vel_lin = vel[:3]
                vel_diff = vel_lin - self.arm_vel_des
                self.opti.subject_to(vel_diff == [0] * 3)

                # Force at end-effector (after all feet)
                f_e = forces[3*self.n_feet:]
                self.opti.subject_to(f_e == self.arm_f_des)

            # Warm start
            self.opti.set_initial(self.DX_opt[i], np.zeros(self.ndx_opt))
            self.opti.set_initial(self.U_opt[i], u_des)

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
        self.vdes = []
        self.taus = []
        self.fs = []

    def update_initial_state(self, x_init):
        self.opti.set_value(self.x_init, x_init)

    def update_contact_schedule(self, shift_idx=0):
        contact_schedule = self.gait_sequence.shift_contact_schedule(shift_idx)
        contact_schedule = contact_schedule[:, :self.nodes]
        self.opti.set_value(self.contact_schedule, contact_schedule)

    def set_com_goal(self, com_goal):
        self.opti.set_value(self.com_goal, com_goal)

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
            # Last node
            # DX_diff = self.DX_prev[-1] - DX_init
            # self.opti.set_initial(self.DX_opt[self.nodes], DX_diff)

        if self.U_prev is not None:
            for i in range(self.nodes - 1):
                self.opti.set_initial(self.U_opt[i], self.U_prev[i+1])
            # Last node
            # self.opti.set_initial(self.U_opt[self.nodes-1], self.U_prev[-1])

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
                "max_iter": 6,
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
                    self.Q_diag,
                    self.R_diag,
                    self.com_goal
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
            # Get all info from self.opti
            x = self.opti.x
            p = self.opti.p
            f = self.opti.f
            g = self.opti.g
            lbg = self.opti.lbg
            ubg = self.opti.ubg

            J_g = casadi.jacobian(g, x)
            hess_f, grad_f = casadi.hessian(f, x)

            # Store data functions
            # TODO: Code generation
            self.sqp_data = casadi.Function("sqp_data", [x, p], [hess_f, grad_f, J_g, g, lbg, ubg])
            self.f_data = casadi.Function("f_data", [x, p], [f, grad_f])
            self.g_data = casadi.Function("g_data", [x, p], [g, lbg, ubg])

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
            ocp_params = casadi.vertcat(
                self.opti.value(self.x_init),
                casadi.vec(self.opti.value(self.contact_schedule)),  # flattened
                self.opti.value(self.com_goal),
            )
            if self.arm_ee_id:
                ocp_params = casadi.vertcat(
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

            self._retract_sqp_sol(current_x, retract_all)

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
            self.vdes.append(np.array(u_sol[:self.nv]))
            self.taus.append(np.array(u_sol[self.nv : self.nv + self.nj]))
            self.fs.append(np.array(u_sol[self.nv + self.nj :]))

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
                self.qs.append(np.array(x_sol[:self.nq]))
                self.vs.append(np.array(x_sol[self.nq:]))
                self.taus.append(np.array(u_sol[:self.nj]))
                self.fs.append(np.array(u_sol[self.nj:]))

        dx_last = sol_x[self.nodes*nx_opt:]
        x_last = self.dyn.state_integrate()(x_init, dx_last)
        self.DX_prev.append(np.array(dx_last))

        if retract_all:
            self.qs.append(np.array(x_last[:self.nq]))
            self.vs.append(np.array(x_last[self.nq:]))
            self.taus.append(np.array(u_sol[:self.nj]))
            self.fs.append(np.array(u_sol[self.nj:]))

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
