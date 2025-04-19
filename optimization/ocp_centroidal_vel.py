import numpy as np
import casadi as ca

from dynamics import DynamicsCentroidalVel
from .ocp import OCP


class OCPCentroidalVel(OCP):
    def __init__(self, robot, solver, nodes, include_base=False):
        super().__init__(robot, solver, nodes)

        # Dynamics
        self.dyn = DynamicsCentroidalVel(self.model, self.mass, self.foot_frames)

        # Nominal state
        self.x_nom = np.concatenate(([0] * 6, self.robot.q0))  # CoM + joint pos

        # Whether to include base velocities in the inputs
        self.include_base = include_base
        if self.include_base:
            self.nv_opt = self.nv
        else:
            self.nv_opt = self.nj

    def set_weights(self):
        # State and input weights
        Q_com_diag = np.concatenate((
            [1000] * 3,   # CoM lin momentum
            [1000] * 3,   # CoM ang momentum
        ))
        Q_base_pos_diag = np.concatenate((
            [0] * 2,      # base x/y
            [1000],       # base z
            [10000] * 2,  # base rot x/y
            [0],          # base rot z
        ))
        Q_joint_pos_diag = np.tile([1000, 500, 500], 4)  # hip, thigh, calf

        if self.arm_ee_frame:
            Q_joint_pos_diag = np.concatenate((Q_joint_pos_diag, [100] * 6))  # arm

        Q_diag = np.concatenate((Q_com_diag, Q_base_pos_diag, Q_joint_pos_diag))
        R_diag = np.concatenate((
            [1] * self.nv_opt,  # velocities
            [1e-3] * self.nf,  # forces
        ))

        self.opti.set_value(self.Q_diag, Q_diag)
        self.opti.set_value(self.R_diag, R_diag)

    def setup_variables(self):
        # State size
        self.nx = 6 + self.nq  # centroidal momentum + joint positions
        self.ndx_opt = 6 + self.nv  # centroidal momentum + position deltas

        # Input size: Can be adaptive for each node
        self.nu_opt = [self.nv_opt + self.nf] * self.nodes  # velocities + end-effector forces
        self.f_idx = self.nv_opt  # start index for forces

        # Decision variables
        self.DX_opt = []
        self.U_opt = []
        for i in range(self.nodes):
            self.DX_opt.append(self.opti.variable(self.ndx_opt))
            self.U_opt.append(self.opti.variable(self.nu_opt[i]))
        self.DX_opt.append(self.opti.variable(self.ndx_opt))

    def setup_targets(self):
        # Desired state
        x_des = ca.vertcat(self.base_vel_des, self.robot.q0)  # CoM target + nominal configuration
        self.dx_des = self.dyn.state_difference()(self.x_init, x_des)

        # Desired forces, TODO: customize weight distribution
        f_gravity = 9.81 * self.mass
        f_front = ca.repmat(ca.vertcat(0, 0, 0.8 * f_gravity / self.n_contacts), 2, 1)
        f_rear = ca.repmat(ca.vertcat(0, 0, 1.2 * f_gravity / self.n_contacts), 2, 1)
        self.f_des = ca.vertcat(f_front, f_rear)
        # self.f_des = ca.repmat(ca.vertcat(0, 0, f_gravity / self.n_contacts), self.n_feet, 1)
        if self.ext_force_frame:
            self.f_des = ca.vertcat(self.f_des, [0] * 3)  # zero force at end-effector

        # Desired input (use this for warm starting)
        self.u_des = ca.vertcat([0] * self.nv_opt, self.f_des)  # zero velocities

    def setup_dynamics_constraints(self, i):
        # Gather all state and input info
        dx = self.DX_opt[i]
        dh = dx[:6]  # delta h
        dq = dx[6:]  # delta q, not v
        h = self.get_h(i)
        q = self.get_q(i)
        v = self.get_v(i)
        forces = self.get_forces(i)
        dt = self.dts[i]

        # Dynamics constraint
        dx_next = self.DX_opt[i+1]
        dh_next = dx_next[:6]
        dq_next = dx_next[6:]
        h_dot = self.dyn.com_dynamics(self.ext_force_frame)(q, forces)
        self.opti.subject_to(dh_next == dh + h_dot * dt)
        self.opti.subject_to(dq_next == dq + v * dt)

        if self.include_base:
            # Path constraint for dynamics gaps
            gaps = self.dyn.dynamics_gaps()(h, q, v)
            self.opti.subject_to(gaps == [0] * 6)

    def get_h(self, i):
        dx = self.DX_opt[i]
        x = self.dyn.state_integrate()(self.x_init, dx)
        return x[:6]

    def get_q(self, i):
        dx = self.DX_opt[i]
        x = self.dyn.state_integrate()(self.x_init, dx)
        return x[6:]

    def get_v(self, i):
        if self.include_base:
            v = self.U_opt[i][:self.nv_opt]
        else:
            v_j = self.U_opt[i][:self.nv_opt]
            # Compute base acceleration from dynamics
            h = self.get_h(i)
            q = self.get_q(i)
            v_b = self.dyn.base_vel_dynamics()(h, q, v_j)
            v = ca.vertcat(v_b, v_j)
        return v

    def get_forces(self, i):
        u = self.U_opt[i]
        return u[self.f_idx:]

    def warm_start(self):
        # Previous solution for dx
        if self.DX_prev is not None:
            for i in range(self.nodes + 1):
                dx_prev = self.DX_prev[i]
                self.opti.set_initial(self.DX_opt[i], dx_prev)
                continue

        # Previous solution for vel
        # Tracking target for f (gravity compensation)
        if self.U_prev is not None:
            contact_schedule = self.opti.value(self.contact_schedule)
            for i in range(self.nodes):
                f_des = self.opti.value(self.f_des)
                for j in range(self.n_feet):
                    # Set forces to zero if not in contact
                    if contact_schedule[j, i] == 0:
                        f_des[3 * j : 3 * j + 3] = [0] * 3

                u_prev = self.U_prev[i]
                v_prev = u_prev[:self.nv_opt]
                u_warm = ca.vertcat(v_prev, f_des)
                self.opti.set_initial(self.U_opt[i], u_warm)

        if self.lam_g is not None:
            self.opti.set_initial(self.opti.lam_g, self.lam_g)

    def retract_opti_sol(self, retract_all=True):
        # Retract self.opti solution stored in self.sol
        self.DX_prev = [self.sol.value(dx) for dx in self.DX_opt]
        self.U_prev = [self.sol.value(u) for u in self.U_opt]
        x_init = self.opti.value(self.x_init)

        for dx_sol, u_sol in zip(self.DX_prev, self.U_prev):
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            h = np.array(x_sol[:6])
            q = np.array(x_sol[6:])
            forces = np.array(u_sol[self.f_idx:])
            if self.include_base:
                v = np.array(u_sol[:self.nv_opt])
            else:
                v_j = np.array(u_sol[:self.nv_opt])
                # Compute base velocity from dynamics
                v_b = self.dyn.base_vel_dynamics()(h, q, v_j)
                v = np.concatenate((v_b, v_j))

            self.q_sol.append(q)
            self.v_sol.append(v)
            self.forces_sol.append(forces)

            if not retract_all:
                return

    def retract_stacked_sol(self, sol_x, retract_all=True):
        # Retract the given solution, which contains all stacked states and inputs
        self.DX_prev = []
        self.U_prev = []
        x_init = self.opti.value(self.x_init)
        nx_opt = self.ndx_opt + self.nu_opt[0]  # nu_opt is constant

        for i in range(self.nodes):
            sol = sol_x[i*nx_opt : (i+1)*nx_opt]
            dx_sol = sol[:self.ndx_opt]
            u_sol = sol[self.ndx_opt:]
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            self.DX_prev.append(np.array(dx_sol))
            self.U_prev.append(np.array(u_sol))

            if i == 0 or retract_all:
                h = np.array(x_sol[:6])
                q = np.array(x_sol[6:])
                forces = np.array(u_sol[self.f_idx:])
                if self.include_base:
                    v = np.array(u_sol[:self.nv_opt])
                else:
                    v_j = np.array(u_sol[:self.nv_opt])
                    # Compute base velocity from dynamics
                    v_b = self.dyn.base_vel_dynamics()(h, q, v_j)
                    v_b = np.array(v_b).flatten()
                    v = np.concatenate((v_b, v_j))

                self.q_sol.append(q)
                self.v_sol.append(v)
                self.forces_sol.append(forces)

        dx_last = sol_x[self.nodes*nx_opt:]
        x_last = self.dyn.state_integrate()(x_init, dx_last)
        self.DX_prev.append(np.array(dx_last))

        if retract_all:
            self.q_sol.append(np.array(x_last[6:]))
