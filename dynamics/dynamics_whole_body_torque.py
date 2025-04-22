import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca

from .dynamics import Dynamics


class DynamicsWholeBodyTorque(Dynamics):
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

    def rnea_dynamics(self, ext_force_frame=None):
        q = ca.SX.sym("q", self.nq)  # positions
        v = ca.SX.sym("v", self.nv)  # velocities
        a = ca.SX.sym("a", self.nv)  # accelerations

        # End-effector forces
        ee_frames = self.foot_frames.copy()
        if ext_force_frame:
            ee_frames.append(ext_force_frame)
        forces = ca.SX.sym("forces", 3 * len(ee_frames))

        # RNEA
        cpin.framesForwardKinematics(self.model, self.data, q)
        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(self.model.njoints)]
        for idx, frame_id in enumerate(ee_frames):
            # OCS2 implementation
            joint_id = self.model.frames[frame_id].parentJoint
            translation_joint_to_contact_frame = self.model.frames[frame_id].placement.translation
            rotation_world_to_joint_frame = self.data.oMi[joint_id].rotation.T

            f_world = forces[idx * 3 : (idx + 1) * 3]
            f_lin = rotation_world_to_joint_frame @ f_world
            f_ang = ca.cross(translation_joint_to_contact_frame, f_lin)
            f = ca.vertcat(f_lin, f_ang)
            f_ext[joint_id] = cpin.Force(f)

        # Return whole-body torques (base + joints)
        tau_rnea = cpin.rnea(self.model, self.data, q, v, a, f_ext)

        return ca.Function("rnea_dyn", [q, v, a, forces], [tau_rnea], ["q", "v", "a", "forces"], ["tau_rnea"])

    def aba_dynamics(self, ext_force_frame=None):
        q = ca.SX.sym("q", self.nq)  # positions
        v = ca.SX.sym("v", self.nv)  # velocites
        tau_j = ca.SX.sym("tau_j", self.nj)  # joint torques
        tau = ca.vertcat(ca.SX.zeros(6), tau_j)  # zero base torques

        # End-effector forces
        ee_frames = self.foot_frames.copy()
        if ext_force_frame:
            ee_frames.append(ext_force_frame)
        forces = ca.SX.sym("forces", 3 * len(ee_frames))

        # ABA
        cpin.framesForwardKinematics(self.model, self.data, q)
        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(self.model.njoints)]
        for idx, frame_id in enumerate(ee_frames):
            # OCS2 implementation
            joint_id = self.model.frames[frame_id].parentJoint
            translation_joint_to_contact_frame = self.model.frames[frame_id].placement.translation
            rotation_world_to_joint_frame = self.data.oMi[joint_id].rotation.T

            f_world = forces[idx * 3 : (idx + 1) * 3]
            f_lin = rotation_world_to_joint_frame @ f_world
            f_ang = ca.cross(translation_joint_to_contact_frame, f_lin)
            f = ca.vertcat(f_lin, f_ang)
            f_ext[joint_id] = cpin.Force(f)

        # Return whole-body accelerations (base + joints)
        a = cpin.aba(self.model, self.data, q, v, tau, f_ext)

        return ca.Function("aba_dyn", [q, v, tau_j, forces], [a], ["q", "v", "tau_j", "forces"], ["a"])

    def srb_dynamics(self, ext_force_frame=None):
        q = ca.SX.sym("q", self.nq)  # positions
        q_j_fixed = ca.SX.sym("q_j_fixed", self.nj)  # joint positions fixed
        v_b = ca.SX.sym("v_b", 6)  # velocities
        a_b = ca.SX.sym("a_b", 6)  # accelerations

        # End-effector forces
        ee_frames = self.foot_frames.copy()
        if ext_force_frame:
            ee_frames.append(ext_force_frame)
        forces = ca.SX.sym("forces", 3 * len(ee_frames))

        q_b = q[:7]
        q_fixed = ca.vertcat(q_b, q_j_fixed)
        v = ca.vertcat(v_b, [0] * self.nj)  # zero joint velocities
        a = ca.vertcat(a_b, [0] * self.nj)  # zero joint accelerations

        # SRB with RNEA
        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(self.model.njoints)]
        cpin.framesForwardKinematics(self.model, self.data, q)
        for idx, frame_id in enumerate(ee_frames):
            # OCS2 implementation
            joint_id = self.model.frames[frame_id].parentJoint
            translation_joint_to_contact_frame = self.model.frames[frame_id].placement.translation
            rotation_world_to_joint_frame = self.data.oMi[joint_id].rotation.T

            f_world = forces[idx * 3 : (idx + 1) * 3]
            f_lin = rotation_world_to_joint_frame @ f_world
            f_ang = ca.cross(translation_joint_to_contact_frame, f_lin)
            f = ca.vertcat(f_lin, f_ang)
            f_ext[joint_id] = cpin.Force(f)

        tau_rnea = cpin.rnea(self.model, self.data, q_fixed, v, a)  # q_fixed!

        tau_ext_base = ca.SX.zeros(6)
        for idx, frame_id in enumerate(ee_frames):
            f_world = forces[idx * 3 : (idx + 1) * 3]
            J_c = cpin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
            J_c_lin_base = J_c[:3, :6]
            tau_ext_base += J_c_lin_base.T @ f_world

        tau_srb = tau_rnea[:6] - tau_ext_base

        return ca.Function(
            "srb_dyn",
            [q, q_j_fixed, v_b, a_b, forces],
            [tau_srb],
            ["q", "q_j_fixed", "v_b", "a_b", "forces"],
            ["tau_srb"]
        )
