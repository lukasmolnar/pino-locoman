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
        self.mass = self.data.mass[0]
        self.dt = self.gait_sequence.dt
        self.ee_ids = robot.ee_ids
        self.arm_ee_id = robot.arm_ee_id
        self.n_feet = len(self.ee_ids)

        # State and inputs to optimize
        self.nx = self.nq + self.nv  # positions + velocities
        self.ndx_opt = self.nv * 2  # position deltas + velocities
        self.nu_opt = self.nv + self.nf + self.nj  # velocities + forces + torques

        # Dynamics
        self.dyn = DynamicsRNEA(self.model, self.mass, self.ee_ids)

        # Decision variables
        # TODO: Check about removing torques for i >= 2 (but this makes problem overconstrained)
        self.DX_opt = []
        self.U_opt = []
        for i in range(self.nodes):
            self.DX_opt.append(self.opti.variable(self.ndx_opt))
            self.U_opt.append(self.opti.variable(self.nu_opt))
        self.DX_opt.append(self.opti.variable(self.ndx_opt))

        # Parameters
        self.x_init = self.opti.parameter(self.nx)  # initial state
        self.tau_prev = self.opti.parameter(self.nj)  # previous joint torques
        self.contact_schedule = self.opti.parameter(self.n_feet, self.nodes) # in_contact: 0 or 1
        self.bezier_schedule = self.opti.parameter(self.n_feet, self.nodes) # bez_idx: normalized to [0, 1]
        self.n_contacts = self.opti.parameter(1)  # number of contact feet
        self.com_goal = self.opti.parameter(6)  # linear + angular momentum
        self.step_height = self.opti.parameter(1)  # step height
        self.arm_f_des = self.opti.parameter(3)  # force at end-effector
        self.arm_vel_des = self.opti.parameter(3)  # velocity at end-effector

        self.Q_diag = self.opti.parameter(self.ndx_opt)  # state weights
        self.R_diag = self.opti.parameter(self.nu_opt)  # input weights (with torques)
        self.W_diag = self.opti.parameter(self.nj)  # additional weights for joint torques
        self.Q = ca.diag(self.Q_diag)
        self.R = ca.diag(self.R_diag)
        self.W = ca.diag(self.W_diag)

        # Desired state and input
        v_des = ca.vertcat(self.com_goal, [0] * self.nj)
        x_des = ca.vertcat(robot.q0, v_des)
        dx_des = self.dyn.state_difference()(self.x_init, x_des)
        f_des = ca.repmat(ca.vertcat(0, 0, 9.81 * self.mass / self.n_contacts), self.n_feet, 1)
        if self.arm_ee_id:
            f_des = ca.vertcat(f_des, self.arm_f_des)
        u_des = ca.vertcat(ca.MX.zeros(self.nv), f_des, ca.MX.zeros(self.nj))  # zero velocity + torque

        # OBJECTIVE
        obj = 0
        for i in range(self.nodes):
            # Track desired state and input
            dx = self.DX_opt[i]
            u = self.U_opt[i]
            err_dx = dx - dx_des
            err_u = u - u_des
            obj += 0.5 * err_dx.T @ self.Q @ err_dx
            obj += 0.5 * err_u.T @ self.R @ err_u

        # Keep initial torque close to previous solution
        tau_0 = self.U_opt[0][self.nv + self.nf :]
        err_tau = tau_0 - self.tau_prev
        obj += 0.5 * err_tau.T @ self.W @ err_tau

        # Final state
        dx = self.DX_opt[self.nodes]
        err_dx = dx - dx_des
        obj += 0.5 * err_dx.T @ self.Q @ err_dx

        # CONSTRAINTS
        self.opti.subject_to(self.DX_opt[0] == [0] * 2 * self.nv)  # initial pos + vel

        for i in range(self.nodes):
            # Gather all state and input info
            dx = self.DX_opt[i]
            dq = dx[:self.nv]  # delta q, not v
            x = self.dyn.state_integrate()(self.x_init, dx)
            u = self.U_opt[i]
            q = x[:self.nq]
            v = x[self.nq:]
            v_next_des = u[:self.nv]
            forces = u[self.nv : self.nv + self.nf]

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
                tau_j = u[self.nv + self.nf :]
                self.opti.subject_to(tau_rnea[6:] == tau_j)  # joints

                # Torque limits
                tau_min = -robot.joint_torque_max
                tau_max = robot.joint_torque_max
                self.opti.subject_to(self.opti.bounded(tau_min, tau_j, tau_max))

            # Contact and swing constraints
            for idx, frame_id in enumerate(self.ee_ids):
                f_e = forces[idx * 3 : (idx + 1) * 3]

                # Get contact schedule info
                in_contact = self.contact_schedule[idx, i]
                bezier_idx = self.bezier_schedule[idx, i]

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
                vel_z_des = self.gait_sequence.get_bezier_vel_z(0, bezier_idx, h=self.step_height)
                vel_diff = vel_z - vel_z_des
                self.opti.subject_to((1 - in_contact) * vel_diff == 0)

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

            # Joint limits
            pos_min = robot.joint_pos_min
            pos_max = robot.joint_pos_max
            vel_min = -robot.joint_vel_max
            vel_max = robot.joint_vel_max
            q_j = q[7:]  # skip base quaternion
            v_j = v[6:]  # skip base angular velocity
            self.opti.subject_to(self.opti.bounded(pos_min, q_j, pos_max))
            self.opti.subject_to(self.opti.bounded(vel_min, v_j, vel_max))

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
        self.qs = []
        self.vs = []
        self.vdes = []
        self.fs = []
        self.taus = []

    def update_initial_state(self, x_init):
        self.opti.set_value(self.x_init, x_init)

    def update_previous_torques(self, tau_prev):
        self.opti.set_value(self.tau_prev, tau_prev)

    def update_gait_sequence(self, shift_idx=0):
        contact_schedule = self.gait_sequence.shift_contact_schedule(shift_idx)
        bezier_schedule = self.gait_sequence.shift_bezier_schedule(shift_idx)
        n_contacts = self.gait_sequence.n_contacts
        self.opti.set_value(self.contact_schedule, contact_schedule[:, :self.nodes])
        self.opti.set_value(self.bezier_schedule, bezier_schedule[:, :self.nodes])
        self.opti.set_value(self.n_contacts, n_contacts)

    def set_com_goal(self, com_goal):
        self.opti.set_value(self.com_goal, com_goal)

    def set_step_height(self, step_height):
        self.opti.set_value(self.step_height, step_height)

    def set_arm_task(self, arm_f_des, arm_vel_des):
        self.opti.set_value(self.arm_f_des, arm_f_des)
        self.opti.set_value(self.arm_vel_des, arm_vel_des)

    def set_weights(self, Q_diag, R_diag, W_diag):
        self.opti.set_value(self.Q_diag, Q_diag)
        self.opti.set_value(self.R_diag, R_diag)
        self.opti.set_value(self.W_diag, W_diag)

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
                    self.tau_prev,
                    self.contact_schedule,
                    self.bezier_schedule,
                    self.n_contacts,
                    self.Q_diag,
                    self.R_diag,
                    self.com_goal,
                    self.step_height
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
            self.fs.append(np.array(u_sol[self.nv : self.nv + self.nf]))
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
                self.vdes.append(np.array(u_sol[:self.nv]))
                self.fs.append(np.array(u_sol[self.nv : self.nv + self.nf]))
                self.taus.append(np.array(u_sol[self.nv + self.nf :]))

        dx_last = sol_x[self.nodes*nx_opt:]
        x_last = self.dyn.state_integrate()(x_init, dx_last)
        self.DX_prev.append(np.array(dx_last))

        if retract_all:
            self.qs.append(np.array(x_last[:self.nq]))
            self.vs.append(np.array(x_last[self.nq:]))

    def _get_torque_sol(self, idx):
        u_sol = self.U_prev[idx]
        tau_sol = u_sol[self.nv + self.nf :]
        return tau_sol

    def _simulate_step(self, x_init, u, dt):
        q = x_init[:self.nq]
        v = x_init[self.nq:]
        forces = u[self.nv : self.nv + self.nf]
        tau = u[self.nv + self.nf :]

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

        tau_all = np.concatenate((np.zeros(6), tau.flatten()))
        a = pin.aba(self.model, self.data, q, v, tau_all, f_ext)

        dq = v * dt + 0.5 * a * dt**2
        dv = a * dt

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
