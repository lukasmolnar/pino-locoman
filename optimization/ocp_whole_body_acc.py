import numpy as np
import casadi as ca

from utils.helpers import *
from dynamics import DynamicsWholeBodyAcc
from .ocp import OCP


class OCPWholeBodyAcc(OCP):
    def __init__(self, robot, nodes):
        super().__init__(robot, nodes)

        self.dyn = DynamicsWholeBodyAcc(self.model, self.mass, self.foot_frames)

        # Store solutions
        self.qs = []
        self.vs = []
        self.accs = []
        self.forces = []

    def setup_variables(self):
        # State size
        self.nx = self.nq + self.nv  # positions + velocities
        self.ndx_opt = self.nv * 2  # position deltas + velocities

        # Input size: Can be adaptive for each node
        self.nu_opt = [self.nj + self.nf] * self.nodes  # joint accelerations + end-effector forces
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
        self.u_des = ca.vertcat([0] * self.nj, self.f_des)  # zero joint acc

    def setup_dynamics_constraints(self, i):
        # Gather all state and input info
        dx = self.DX_opt[i]
        dq = dx[:self.nv]  # delta q, not v
        dv = dx[self.nv:]  # delta v
        q = self.get_q(i)
        v = self.get_v(i)
        a_j = self.get_a_j(i)
        forces = self.get_forces(i)
        dt = self.dts[i]

        # Get base acceleration from dynamics
        a_b = self.dyn.base_acceleration_dynamics(self.ext_force_frame)(q, v, a_j, forces)
        a = ca.vertcat(a_b, a_j)

        # Dynamics constraint
        dx_next = self.DX_opt[i+1]
        dq_next = dx_next[:self.nv]
        dv_next = dx_next[self.nv:]
        self.opti.subject_to(dq_next == dq + v * dt)  # TODO: possibly consider acceleration
        self.opti.subject_to(dv_next == dv + a * dt)

    def get_q(self, i):
        dx = self.DX_opt[i]
        x = self.dyn.state_integrate()(self.x_init, dx)
        return x[:self.nq]

    def get_v(self, i):
        dx = self.DX_opt[i]
        x = self.dyn.state_integrate()(self.x_init, dx)
        return x[self.nq:]

    def get_a_j(self, i):
        return self.U_opt[i][:self.f_idx]

    def get_forces(self, i):
        return self.U_opt[i][self.f_idx:]

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
                # Previous solution for a_j
                # Tracking target for f (gravity compensation)
                u_prev = self.U_prev[i]
                a_j_prev = u_prev[:self.f_idx]
                f_des = self.opti.value(self.f_des)
                u_warm = ca.vertcat(a_j_prev, f_des)
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
            self.qs.append(np.array(x_sol[:self.nq]))
            self.vs.append(np.array(x_sol[self.nq:]))
            self.accs.append(np.array(u_sol[:self.f_idx]))
            self.forces.append(np.array(u_sol[self.f_idx:]))

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
                self.qs.append(np.array(x_sol[:self.nq]))
                self.vs.append(np.array(x_sol[self.nq:]))
                self.accs.append(np.array(u_sol[:self.f_idx]))
                self.forces.append(np.array(u_sol[self.f_idx:]))

        dx_last = sol_x[self.nodes*nx_opt:]
        x_last = self.dyn.state_integrate()(x_init, dx_last)
        self.DX_prev.append(np.array(dx_last))

        if retract_all:
            self.qs.append(np.array(x_last[:self.nq]))
            self.vs.append(np.array(x_last[self.nq:]))
