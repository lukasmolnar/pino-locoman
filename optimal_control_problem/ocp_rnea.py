import numpy as np
import casadi as ca
import pinocchio as pin

from utils.helpers import *
from dynamics import DynamicsRNEA
from .ocp import OCP


class OCP_RNEA(OCP):
    def __init__(self, robot, nodes, tau_nodes):
        super().__init__(robot, nodes)
        self.tau_nodes = tau_nodes

        self.dyn = DynamicsRNEA(self.model, self.mass, self.feet_ids)

        # Store solutions
        self.qs = []
        self.vs = []
        self.accs = []
        self.forces = []
        self.taus = []

    def setup_variables(self):
        # State size
        self.nx = self.nq + self.nv  # positions + velocities
        self.ndx_opt = self.nv * 2  # position deltas + velocities

        # Input size: Adaptive for each node
        self.nu_opt = (
            [self.nv + self.nf + self.nj] * self.tau_nodes +  # accelerations + forces + torques
            [self.nv + self.nf] * (self.nodes - self.tau_nodes)  # accelerations + forces
        )
        self.f_idx = self.nv  # start index for forces
        self.tau_idx = self.nv + self.nf  # start index for torques

        # Decision variables
        self.DX_opt = []
        self.U_opt = []
        for i in range(self.nodes):
            self.DX_opt.append(self.opti.variable(self.ndx_opt))
            self.U_opt.append(self.opti.variable(self.nu_opt[i]))
        self.DX_opt.append(self.opti.variable(self.ndx_opt))

    def setup_parameters(self):
        super().setup_parameters()
        self.tau_prev = self.opti.parameter(self.nj)  # previous joint torques
        self.W_diag = self.opti.parameter(self.nj)  # additional weights for joint torques

    def setup_targets(self):
        # Desired state
        x_des = ca.vertcat(self.robot.q0, self.base_vel_des, [0] * self.nj)  # Nominal q + base target + zero joint vel
        self.dx_des = self.dyn.state_difference()(self.x_init, x_des)

        # Desired forces, TODO: customize weight distribution
        f_gravity = 9.81 * self.mass
        # f_front = ca.repmat(ca.vertcat(0, 0, 0.8 * f_gravity / self.n_contacts), 2, 1)
        # f_rear = ca.repmat(ca.vertcat(0, 0, 1.2 * f_gravity / self.n_contacts), 2, 1)
        # self.f_des = ca.vertcat(f_front, f_rear)
        self.f_des = ca.repmat(ca.vertcat(0, 0, f_gravity / self.n_contacts), self.n_feet, 1)
        if self.arm_id:
            self.f_des = ca.vertcat(self.f_des, [0] * 3)  # zero force at end-effector

        # Desired input: Use this for warm starting
        self.u_des = ca.vertcat([0] * self.nv, self.f_des, [0] * self.nj)  # zero acc + torque

    def setup_objective(self):
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
        tau_0 = self.U_opt[0][self.tau_idx:]
        tau_des = self.tau_prev
        err_tau = tau_0 - tau_des
        obj += err_tau.T @ W @ err_tau

        # Final state
        dx = self.DX_opt[self.nodes]
        err_dx = dx - self.dx_des
        obj += err_dx.T @ Q @ err_dx

        return obj

    def setup_dynamics_constraints(self, i):
        # Gather all state and input info
        dx = self.DX_opt[i]
        dq = dx[:self.nv]  # delta q, not v
        dv = dx[self.nv:]  # delta v
        q = self.get_q(i)
        v = self.get_v(i)
        a = self.get_acc(i)
        forces = self.get_forces(i)
        tau_j = self.get_tau(i)
        dt = self.dts[i]

        # Dynamics constraint
        dx_next = self.DX_opt[i+1]
        dq_next = dx_next[:self.nv]
        dv_next = dx_next[self.nv:]
        self.opti.subject_to(dq_next == dq + v * dt + 0.5 * a * dt**2)
        self.opti.subject_to(dv_next == dv + a * dt)

        # RNEA constraint
        tau_rnea = self.dyn.tau_dynamics(self.arm_id)(q, v, a, forces)
        self.opti.subject_to(tau_rnea[:6] == [0] * 6)  # base

        # Torque constraints
        if i < self.tau_nodes:
            self.opti.subject_to(tau_rnea[6:] == tau_j)  # joints

            # Torque limits
            tau_min = -self.robot.joint_torque_max
            tau_max = self.robot.joint_torque_max
            self.opti.subject_to(self.opti.bounded(tau_min, tau_j, tau_max))

    def get_q(self, i):
        dx = self.DX_opt[i]
        x = self.dyn.state_integrate()(self.x_init, dx)
        return x[:self.nq]

    def get_v(self, i):
        dx = self.DX_opt[i]
        x = self.dyn.state_integrate()(self.x_init, dx)
        return x[self.nq:]

    def get_acc(self, i):
        return self.U_opt[i][:self.f_idx]

    def get_forces(self, i):
        return self.U_opt[i][self.f_idx:self.tau_idx]

    def get_tau(self, i):
        return self.U_opt[i][self.tau_idx:]

    def get_tau_sol(self, i):
        u_sol = self.U_prev[i]
        tau_sol = u_sol[self.tau_idx:]
        return tau_sol

    def set_weights(self, Q_diag, R_diag, W_diag):
        super().set_weights(Q_diag, R_diag)
        self.opti.set_value(self.W_diag, W_diag)

    def update_previous_torques(self, tau_prev):
        self.opti.set_value(self.tau_prev, tau_prev)

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
                a_prev = u_prev[:self.f_idx]
                f_des = self.opti.value(self.f_des)
                u_warm = ca.vertcat(a_prev, f_des)
                if i < self.tau_nodes:
                    tau_prev = u_prev[self.tau_idx:]
                    u_warm = ca.vertcat(u_warm, tau_prev)
                self.opti.set_initial(self.U_opt[i], u_warm)

        if self.lam_g is not None:
            self.opti.set_initial(self.opti.lam_g, self.lam_g)

    def init_solver(self, solver="fatrop", warm_start=False):
        super().init_solver(solver, warm_start)

        # Add RNEA specific parameters
        if solver == "fatrop":
            self.solver_params += [self.tau_prev, self.W_diag]

        elif solver == "osqp":
            self.solver_params = ca.vertcat(
                self.solver_params,
                self.opti.value(self.tau_prev),
                self.opti.value(self.W_diag),
            )

        else:
            raise ValueError(f"Solver {solver} not supported")

    def retract_opti_sol(self, retract_all=True):
        # Retract self.opti solution stored in self.sol
        self.DX_prev = [self.sol.value(dx) for dx in self.DX_opt]
        self.U_prev = [self.sol.value(u) for u in self.U_opt]
        self.lam_g = self.sol.value(self.opti.lam_g)
        x_init = self.opti.value(self.x_init)

        for dx_sol, u_sol in zip(self.DX_prev, self.U_prev):
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            self.qs.append(np.array(x_sol[:self.nq]))
            self.vs.append(np.array(x_sol[self.nq:]))
            self.accs.append(np.array(u_sol[:self.f_idx]))
            self.forces.append(np.array(u_sol[self.f_idx:self.tau_idx]))
            self.taus.append(np.array(u_sol[self.tau_idx:]))

            if not retract_all:
                return

        x_last = self.dyn.state_integrate()(x_init, self.DX_prev[-1])
        self.qs.append(np.array(x_last[:self.nq]))
        self.vs.append(np.array(x_last[self.nq:]))

    def retract_stacked_sol(self, sol_x, retract_all=True):
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
                self.accs.append(np.array(u_sol[:self.f_idx]))
                self.forces.append(np.array(u_sol[self.f_idx:self.tau_idx]))
                self.taus.append(np.array(u_sol[self.tau_idx:]))

        dx_last = sol_x[-self.ndx_opt:]
        x_last = self.dyn.state_integrate()(x_init, dx_last)
        self.DX_prev.append(np.array(dx_last))

        if retract_all:
            self.qs.append(np.array(x_last[:self.nq]))
            self.vs.append(np.array(x_last[self.nq:]))

    
    def simulate_step(self, x_init, u):
        q = x_init[:self.nq]
        v = x_init[self.nq:]
        forces = u[self.f_idx:self.tau_idx]
        tau_j = u[self.tau_idx:]
        dt_sim = self.opti.value(self.dt_min)  # the first step size

        ee_ids = self.feet_ids.copy()
        if self.arm_id:
            ee_ids.append(self.arm_id)

        pin.framesForwardKinematics(self.model, self.data, q)
        f_ext = [pin.Force(np.zeros(6)) for _ in range(self.model.njoints)]
        for idx, frame_id in enumerate(ee_ids):
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
