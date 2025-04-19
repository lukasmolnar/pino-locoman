import numpy as np
import casadi as ca
from scipy.interpolate import interp1d

from dynamics import DynamicsWholeBodyTorque
from .ocp import OCP


class OCPWholeBodyABA(OCP):
    def __init__(self, robot, nodes):
        super().__init__(robot, nodes)

        # Dynamics
        self.dyn = DynamicsWholeBodyTorque(self.model, self.mass, self.foot_frames)

        # Nominal state
        self.x_nom = np.concatenate((self.robot.q0, [0] * self.nv))  # joint pos + vel

        # Store solution
        self.tau_sol = []


    def set_weights(self):
        # State and input weights
        Q_base_pos_diag = np.concatenate((
            [0] * 2,      # base x/y
            [1000],       # base z
            [10000] * 2,  # base rot x/y
            [0],          # base rot z
        ))
        Q_joint_pos_diag = np.tile([1000, 500, 500], 4)  # hip, thigh, calf

        if self.arm_ee_frame:
            Q_joint_pos_diag = np.concatenate((Q_joint_pos_diag, [100] * 6))  # arm

        Q_vel_diag = np.concatenate((
            [2000] * 2,     # base lin x/y
            [1000],         # base lin z
            [1000] * 2,     # base ang x/y
            [2000],         # base ang z
            [1] * self.nj,  # joint vel (all of them)
        ))

        Q_diag = np.concatenate((Q_base_pos_diag, Q_joint_pos_diag, Q_vel_diag))
        R_diag = np.concatenate((
            [1e-3] * self.nj,  # joint torques
            [1e-3] * self.nf,  # forces
        ))

        self.opti.set_value(self.Q_diag, Q_diag)
        self.opti.set_value(self.R_diag, R_diag)

    def setup_variables(self):
        # State size
        self.nx = self.nq + self.nv  # positions + velocities
        self.ndx_opt = self.nv * 2  # position deltas + velocities

        # Input size: Can be adaptive for each node
        self.nu_opt = [self.nj + self.nf] * self.nodes  # joint torques + end-effector forces
        self.f_idx = self.nj  # start index for forces

        # Decision variables
        self.DX_opt = []
        self.U_opt = []
        for i in range(self.nodes):
            self.DX_opt.append(self.opti.variable(self.ndx_opt))
            self.U_opt.append(self.opti.variable(self.nu_opt[i]))
        self.DX_opt.append(self.opti.variable(self.ndx_opt))

    def setup_targets(self):
        # Desired state
        x_des = ca.vertcat(self.robot.q0, self.base_vel_des, [0] * self.nj)  # Nominal q + base target + zero joint vel
        self.dx_des = self.dyn.state_difference()(self.x_init, x_des)

        # Desired forces, TODO: customize weight distribution
        f_gravity = 9.81 * self.mass
        f_front = ca.repmat(ca.vertcat(0, 0, 0.8 * f_gravity / self.n_contacts), 2, 1)
        f_rear = ca.repmat(ca.vertcat(0, 0, 1.2 * f_gravity / self.n_contacts), 2, 1)
        self.f_des = ca.vertcat(f_front, f_rear)
        # self.f_des = ca.repmat(ca.vertcat(0, 0, f_gravity / self.n_contacts), self.n_feet, 1)
        if self.ext_force_frame:
            self.f_des = ca.vertcat(self.f_des, [0] * 3)  # zero force at end-effector

        # Desired input: Use this for warm starting
        self.u_des = ca.vertcat([0] * self.nj, self.f_des)  # zero torques

    def setup_dynamics_constraints(self, i):
        # Gather all state and input info
        dx = self.DX_opt[i]
        dq = dx[:self.nv]  # delta q, not v
        dv = dx[self.nv:]  # delta v
        q = self.get_q(i)
        v = self.get_v(i)
        tau_j = self.get_tau_j(i)
        forces = self.get_forces(i)
        dt = self.dts[i]

        # ABA accelerations
        a = self.dyn.aba_dynamics(self.ext_force_frame)(q, v, tau_j, forces)

        # Dynamics constraint
        dx_next = self.DX_opt[i+1]
        dq_next = dx_next[:self.nv]
        dv_next = dx_next[self.nv:]
        # self.opti.subject_to(dq_next == dq + v * dt + 0.5 * a * dt**2)
        self.opti.subject_to(dq_next == dq + v * dt)
        self.opti.subject_to(dv_next == dv + a * dt)

    def get_q(self, i):
        dx = self.DX_opt[i]
        x = self.dyn.state_integrate()(self.x_init, dx)
        return x[:self.nq]

    def get_v(self, i):
        dx = self.DX_opt[i]
        x = self.dyn.state_integrate()(self.x_init, dx)
        return x[self.nq:]
    
    def get_tau_j(self, i):
        return self.U_opt[i][:self.nj]

    def get_forces(self, i):
        return self.U_opt[i][self.f_idx:]

    def warm_start(self):
        dts = [self.opti.value(self.dts[i]) for i in range(self.nodes)]
        t_prev = np.concatenate(([0], np.cumsum(dts)))
        t_new = t_prev + dts[0]  # shifted by the first timestep

        # Interpolate previous solution
        if self.DX_prev is not None:
            DX_prev = np.array(self.DX_prev)
            DX_interp_fn = interp1d(t_prev, DX_prev, axis=0, kind='linear', fill_value='extrapolate')
            DX_init = self.DX_prev[1]
            for i in range(self.nodes + 1):
                DX_interp = DX_interp_fn(t_new[i])
                DX_warm = DX_interp - DX_init  # always with respect to the initial state
                self.opti.set_initial(self.DX_opt[i], DX_warm)

        # Interpolate previous solution for tau
        # Use tracking target for forces!
        if self.U_prev is not None:
            U_prev = np.array(self.U_prev)
            U_interp_fn = interp1d(t_prev[:self.nodes], U_prev, axis=0, kind='linear', fill_value='extrapolate')

            contact_schedule = self.opti.value(self.contact_schedule)
            for i in range(self.nodes):
                f_des = self.opti.value(self.f_des)
                for j in range(self.n_feet):
                    # Set forces to zero if not in contact
                    if contact_schedule[j, i] == 0:
                        f_des[3 * j : 3 * j + 3] = [0] * 3

                    U_interp = U_interp_fn(t_new[i])
                    tau_j = U_interp[:self.nj]
                    U_warm = ca.vertcat(tau_j, f_des)
                self.opti.set_initial(self.U_opt[i], U_warm)

        if self.lam_g is not None:
            self.opti.set_initial(self.opti.lam_g, self.lam_g)

    def retract_opti_sol(self, retract_all=True):
        # Retract self.opti solution stored in self.sol
        self.DX_prev = [self.sol.value(dx) for dx in self.DX_opt]
        self.U_prev = [self.sol.value(u) for u in self.U_opt]
        x_init = self.opti.value(self.x_init)

        for dx_sol, u_sol in zip(self.DX_prev, self.U_prev):
            x_sol = self.dyn.state_integrate()(x_init, dx_sol)
            q = np.array(x_sol[:self.nq])
            v = np.array(x_sol[self.nq:])
            tau_j = np.array(u_sol[:self.nj])
            forces = np.array(u_sol[self.f_idx:])

            # ABA dynamics
            a = self.dyn.aba_dynamics(self.ext_force_frame)(q, v, tau_j, forces)
            a = np.array(a)

            self.q_sol.append(q)
            self.v_sol.append(v)
            self.a_sol.append(a)
            self.tau_sol.append(tau_j)
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
                q = np.array(x_sol[:self.nq])
                v = np.array(x_sol[self.nq:])
                tau_j = np.array(u_sol[:self.nj])
                forces = np.array(u_sol[self.f_idx:])

                # ABA dynamics
                a = self.dyn.aba_dynamics(self.ext_force_frame)(q, v, tau_j, forces)
                a = np.array(a)

                self.q_sol.append(q)
                self.v_sol.append(v)
                self.a_sol.append(a)
                self.tau_sol.append(tau_j)
                self.forces_sol.append(forces)

        dx_last = sol_x[self.nodes*nx_opt:]
        x_last = self.dyn.state_integrate()(x_init, dx_last)
        self.DX_prev.append(np.array(dx_last))

        if retract_all:
            self.q_sol.append(np.array(x_last[:self.nq]))
            self.v_sol.append(np.array(x_last[self.nq:]))
