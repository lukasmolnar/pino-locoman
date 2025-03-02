import pinocchio.casadi as cpin
import casadi as ca

from .dynamics import Dynamics


class DynamicsRNEA(Dynamics):
    def __init__(self, model, mass, feet_ids):
        super().__init__(model, mass, feet_ids)

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

    def tau_dynamics(self, arm_id=None):
        # States
        q = ca.SX.sym("q", self.nq)  # positions
        v = ca.SX.sym("v", self.nv)  # velocities
        a = ca.SX.sym("a", self.nv)  # accelerations

        # Inputs
        nf = len(self.feet_ids)
        if arm_id:
            nf += 1
        forces = ca.SX.sym("forces", 3 * nf)  # end-effector forces

        # RNEA
        cpin.framesForwardKinematics(self.model, self.data, q)
        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(self.model.njoints)]
        for idx, frame_id in enumerate(self.feet_ids):
            # TODO: Check this. it is from OCS2.
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
