import numpy as np
import pinocchio.casadi as cpin
import casadi as ca

from .dynamics import Dynamics


class DynamicsCentroidalAcc(Dynamics):
    def __init__(self, model, mass, foot_frames):
        super().__init__(model, mass, foot_frames)

    def state_integrate(self):
        x = ca.SX.sym("x", self.nq + self.nv)
        dx = ca.SX.sym("dx", self.nv + self.nv)

        q = x[:self.nq]
        dq = dx[:self.nv]
        q_next = cpin.integrate(self.model, q, dq)

        v = x[self.nq:]
        dv = dx[self.nv:]
        v_next = v + dv

        x_next = ca.vertcat(q_next, v_next)

        return ca.Function("integrate", [x, dx], [x_next], ["x", "dx"], ["x_next"])

    def state_difference(self):
        x0 = ca.SX.sym("x0", self.nq + self.nv)
        x1 = ca.SX.sym("x1", self.nq + self.nv)

        q0 = x0[:self.nq]
        q1 = x1[:self.nq]
        v0 = x0[self.nq:]
        v1 = x1[self.nq:]

        dq = cpin.difference(self.model, q0, q1)
        dv = v1 - v0
        dx = ca.vertcat(dq, dv)

        return ca.Function("difference", [x0, x1], [dx], ["x0", "x1"], ["dx"])

    def base_acceleration_dynamics(self, ext_force_frame=None):
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
        A_b_inv = self._compute_Ab_inv(A_b)
        a_b = A_b_inv @ (dh - Adot @ v - A_j @ a_j)

        return ca.Function("base_acc", [q, v, a_j, forces], [a_b], ["q", "v", "a_j", "forces"], ["a_b"])

    def dynamics_gaps(self, ext_force_frame=None):
        q = ca.SX.sym("q", self.nq)  # positions
        v = ca.SX.sym("v", self.nv)  # velocities
        a = ca.SX.sym("a", self.nv)  # accelerations

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

        # Dynamics gaps
        gaps = A @ a + Adot @ v - dh

        return ca.Function("dyn", [q, v, a, forces], [gaps], ["q", "v", "a", "forces"], ["a_b"])

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
