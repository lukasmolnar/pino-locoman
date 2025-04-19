import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca

from .dynamics import Dynamics


class DynamicsWholeBodyAcc(Dynamics):
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

    def base_acc_dynamics(self, ext_force_frame=None):
        q = ca.SX.sym("q", self.nq)  # positions
        v = ca.SX.sym("v", self.nv)  # velocities
        a_j = ca.SX.sym("a", self.nj)  # joint accelerations

        # End-effector forces
        ee_frames = self.foot_frames.copy()
        if ext_force_frame:
            ee_frames.append(ext_force_frame)
        forces = ca.SX.sym("forces", 3 * len(ee_frames))

        # Pinocchio terms
        M = cpin.crba(self.model, self.data, q)  # Mass matrix
        nle = cpin.nonLinearEffects(self.model, self.data, q, v)  # Coriolis + gravity
        tau_ext_base = ca.SX.zeros(6)
        for idx, frame_id in enumerate(ee_frames):
            f_world = forces[idx * 3 : (idx + 1) * 3]
            J_c = cpin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
            J_c_lin_base = J_c[:3, :6]
            tau_ext_base += J_c_lin_base.T @ f_world

        # Base acceleration dynamics
        M_bb = M[:6, :6]
        M_bj = M[:6, 6:]
        intermediate = -nle[:6] - M_bj @ a_j + tau_ext_base  # EOM for the base

        # NOTE: The following trick for computing M_bb_inv is from the original 1X implementation,
        # but it only works if the base center is defined at the CoM (in general this is not the case).
        # M_bb_lin = M_bb[:3, :3]
        # M_bb_ang = M_bb[3:6, 3:6]
        # M_bb_lin_inv = ca.inv(M_bb_lin)
        # M_bb_ang_inv = ca.inv(M_bb_ang)
        # a_b_lin = M_bb_lin_inv @ intermediate[:3]
        # a_b_ang = M_bb_ang_inv @ intermediate[3:]
        # a_b = ca.vertcat(a_b_lin, a_b_ang)

        # General case
        M_bb_inv = ca.inv(M_bb)
        a_b = M_bb_inv @ intermediate

        return ca.Function("base_acc_dyn", [q, v, a_j, forces], [a_b], ["q", "v", "a_j", "forces"], ["tau_rnea"])

    def dynamics_gaps(self, ext_force_frame=None):
        q = ca.SX.sym("q", self.nq)  # positions
        v = ca.SX.sym("v", self.nv)  # velocities
        a = ca.SX.sym("a", self.nv)  # accelerations

        # End-effector forces
        ee_frames = self.foot_frames.copy()
        if ext_force_frame:
            ee_frames.append(ext_force_frame)
        forces = ca.SX.sym("forces", 3 * len(ee_frames))

        # Pinocchio terms
        M = cpin.crba(self.model, self.data, q)  # Mass matrix
        nle = cpin.nonLinearEffects(self.model, self.data, q, v)  # Coriolis + gravity
        tau_ext_base = ca.SX.zeros(6)
        for idx, frame_id in enumerate(ee_frames):
            f_world = forces[idx * 3 : (idx + 1) * 3]
            J_c = cpin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
            J_c_lin_base = J_c[:3, :6]
            tau_ext_base += J_c_lin_base.T @ f_world

        # Dynamics gaps
        M_b = M[:6, :]
        gaps = M_b @ a + nle[:6] - tau_ext_base

        return ca.Function("dyn_gaps", [q, v, a, forces], [gaps], ["q", "v", "a", "forces"], ["gaps"])
