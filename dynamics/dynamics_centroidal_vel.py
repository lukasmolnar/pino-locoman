import numpy as np
import pinocchio.casadi as cpin
import casadi as ca

from .dynamics import Dynamics


class DynamicsCentroidalVel(Dynamics):
    def __init__(self, model, mass, foot_frames):
        super().__init__(model, mass, foot_frames)

    def state_integrate(self):
        x = ca.SX.sym("x", 6 + self.nq)
        dx = ca.SX.sym("dx", 6 + self.nv)

        h = x[:6]
        dh = dx[:6]
        h_next = h + dh

        q = x[6:]
        dq = dx[6:]
        q_next = cpin.integrate(self.model, q, dq)

        x_next = ca.vertcat(h_next, q_next)

        return ca.Function("integrate", [x, dx], [x_next], ["x", "dx"], ["x_next"])

    def state_difference(self):
        x0 = ca.SX.sym("x0", 6 + self.nq)
        x1 = ca.SX.sym("x1", 6 + self.nq)

        h0 = x0[:6]
        q0 = x0[6:]
        h1 = x1[:6]
        q1 = x1[6:]

        dh = h1 - h0
        dq = cpin.difference(self.model, q0, q1)
        dx = ca.vertcat(dh, dq)

        return ca.Function("difference", [x0, x1], [dx], ["x0", "x1"], ["dx"])

    def com_dynamics(self, ext_force_frame=None):
        q = ca.SX.sym("q", self.nq)  # positions

        # End-effector forces
        nf = len(self.foot_frames)
        if ext_force_frame:
            nf += 1
        f_e = [ca.SX.sym(f"f_e_{i}", 3) for i in range(nf)]
        forces = ca.vertcat(*f_e)

        # Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.centerOfMass(self.model, self.data)
        cpin.updateFramePlacements(self.model, self.data)

        # COM Dynamics
        g = np.array([0, 0, -9.81 * self.mass])
        dp_com = sum(f_e) + g
        dl_com = ca.SX.zeros(3)
        for idx, frame_id in enumerate(self.foot_frames):
            r_ee = self.data.oMf[frame_id].translation - self.data.com[0]
            dl_com += ca.cross(r_ee, f_e[idx])
        if ext_force_frame:
            r_ee = self.data.oMf[ext_force_frame].translation - self.data.com[0]
            dl_com += ca.cross(r_ee, f_e[-1])

        h_dot = ca.vertcat(dp_com, dl_com) / self.mass  # scale h by mass

        return ca.Function("com_dyn", [q, forces], [h_dot], ["q", "forces"], ["dh"])

    def base_vel_dynamics(self):
        h = ca.SX.sym("h", 6)  # COM momentum
        q = ca.SX.sym("q", self.nq)  # positions
        v_j = ca.SX.sym("v_j", self.nj)  # joint velocities

        # Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        A = cpin.computeCentroidalMap(self.model, self.data, q)

        # Base velocity dynamics
        A_b = A[:, :6]
        A_j = A[:, 6:]
        # A_b_inv = self._compute_Ab_inv(Ab)
        A_b_inv = ca.inv(A_b)
        v_b = A_b_inv @ (h * self.mass - A_j @ v_j)  # scale h by mass

        return ca.Function("base_vel", [h, q, v_j], [v_b], ["h", "q", "v_j"], ["v_b"])

    def base_acc_dynamics(self, ext_force_frame=None):
        """
        Special case: When computing the accelerations from the solution,
        we want this instead of finite differences.
        """
        q = ca.SX.sym("q", self.nq)  # positions
        v = ca.SX.sym("v", self.nv)  # velocities
        a_j = ca.SX.sym("a_j", self.nj)  # joint accelerations

        # End-effector forces
        nf = len(self.foot_frames)
        if ext_force_frame:
            nf += 1
        f_e = [ca.SX.sym(f"f_e_{i}", 3) for i in range(nf)]
        forces = ca.vertcat(*f_e)

        # Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.updateFramePlacements(self.model, self.data)
        r_com = cpin.centerOfMass(self.model, self.data)
        A = cpin.computeCentroidalMap(self.model, self.data, q)
        Adot = cpin.dccrba(self.model, self.data, q, v)

        # COM Dynamics
        g = np.array([0, 0, -9.81 * self.mass])
        dp_com = sum(f_e) + g
        dl_com = ca.SX.zeros(3)
        for idx, frame_id in enumerate(self.foot_frames):
            r_ee = self.data.oMf[frame_id].translation - r_com
            dl_com += ca.cross(r_ee, f_e[idx])
        if ext_force_frame:
            r_ee = self.data.oMf[ext_force_frame].translation - r_com
            dl_com += ca.cross(r_ee, f_e[-1])

        dh = ca.vertcat(dp_com, dl_com)

        # Base acceleration dynamics
        A_j = A[:, 6:]
        A_b = A[:, :6]
        # A_b_inv = self._compute_Ab_inv(A_b)
        A_b_inv = ca.inv(A_b)
        a_b = A_b_inv @ (dh - Adot @ v - A_j @ a_j)

        return ca.Function("base_acc", [q, v, a_j, forces], [a_b], ["q", "v", "a_j", "forces"], ["a_b"])

    def dynamics_gaps(self):
        h = ca.SX.sym("h", 6)  # COM momentum
        q = ca.SX.sym("q", self.nq)  # positions
        v = ca.SX.sym("v", self.nv)  # velocities

        # Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        A = cpin.computeCentroidalMap(self.model, self.data, q)

        # Dynamics gaps
        gaps = A @ v - h * self.mass  # scale h by mass

        return ca.Function("dyn_gaps", [h, q, v], [gaps], ["h", "q", "v"], ["gaps"])

    def _compute_Ab_inv(self, Ab):
        # NOTE: This is the OCS2 implementation
        mass = Ab[0, 0]
        Ab_22_inv = ca.inv(Ab[3:, 3:])
        Ab_inv = ca.SX.zeros(6, 6)
        Ab_inv[:3, :3] = 1 / mass * ca.SX.eye(3)
        Ab_inv[:3, 3:] = -1 / mass * Ab[:3, 3:] @ Ab_22_inv
        Ab_inv[3:, :3] = ca.SX.zeros(3, 3)
        Ab_inv[3:, 3:] = Ab_22_inv
        return Ab_inv
