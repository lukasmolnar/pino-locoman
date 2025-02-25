import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca


class DynamicsRNEA:
    def __init__(
            self,
            model,
            mass,
            ee_ids,
        ):
        self.model = cpin.Model(model)
        self.data = self.model.createData()
        self.mass = mass
        self.ee_ids = ee_ids

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7  # without base position and quaternion

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

    def rnea_dynamics(self, arm_ee_id=None):
        # States
        q = ca.SX.sym("q", self.nq)  # positions
        v = ca.SX.sym("v", self.nv)  # velocities
        a = ca.SX.sym("a", self.nv)  # accelerations

        # Inputs
        nf = len(self.ee_ids)
        if arm_ee_id:
            nf += 1
        forces = ca.SX.sym("forces", 3 * nf)  # end-effector forces

        # RNEA
        cpin.framesForwardKinematics(self.model, self.data, q)
        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(self.model.njoints)]
        for idx, frame_id in enumerate(self.ee_ids):
            # TODO: Check this. it is from OCS2.
            joint_id = self.model.frames[frame_id].parentJoint
            translation_joint_to_contact_frame = self.model.frames[frame_id].placement.translation
            rotation_world_to_joint_frame = self.data.oMi[joint_id].rotation.T

            f_world = forces[idx * 3 : (idx + 1) * 3]
            f_lin = rotation_world_to_joint_frame @ f_world
            f_ang = ca.cross(translation_joint_to_contact_frame, f_lin)
            f = ca.vertcat(f_lin, f_ang)
            f_ext[joint_id] = cpin.Force(f)

        tau_rnea = cpin.rnea(self.model, self.data, q, v, a, f_ext)

        # Separate external forces
        # tau_rnea = cpin.rnea(self.model, self.data, q, v, a)
        # tau_ext = ca.SX.zeros(self.nv)
        # for idx, frame_id in enumerate(self.ee_ids):
        #     J = cpin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
        #     J_lin = J[:3]
        #     f_ext = forces[idx * 3 : (idx + 1) * 3]
        #     tau_ext += J_lin.T @ f_ext
        # if arm_ee_id:
        #     J = cpin.computeFrameJacobian(self.model, self.data, q, arm_ee_id, pin.LOCAL_WORLD_ALIGNED)
        #     J_lin = J[:3]
        #     f_ext = forces[-3:]
        #     tau_ext += J_lin.T @ f_ext
        # tau_rnea -= tau_ext

        return ca.Function("rnea_dyn", [q, v, a, forces], [tau_rnea], ["q", "v", "a", "forces"], ["tau_rnea"])
    
    def srb_dynamics(self, arm_ee_id=None):
        # States
        q_b = ca.SX.sym("q_b", 7)  # base position and quaternion
        q_j = ca.SX.sym("q_j", self.nj)  # joint positions
        q_j_fixed = ca.SX.sym("q_j_fixed", self.nj)  # joint positions
        v_b = ca.SX.sym("v_b", 6)  # base velocities
        a_b = ca.SX.sym("a_b", 6)  # base accelerations

        # Inputs
        nf = len(self.ee_ids)
        if arm_ee_id:
            nf += 1
        forces = ca.SX.sym("forces", 3 * nf)  # end-effector forces

        q = ca.vertcat(q_b, q_j)
        q_fixed = ca.vertcat(q_b, q_j_fixed)
        v = ca.vertcat(v_b, [0] * self.nj)
        a = ca.vertcat(a_b, [0] * self.nj)

        # SRB with RNEA
        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(self.model.njoints)]
        cpin.framesForwardKinematics(self.model, self.data, q)
        for idx, frame_id in enumerate(self.ee_ids):
            # TODO: Check this. it is from OCS2.
            joint_id = self.model.frames[frame_id].parentJoint
            translation_joint_to_contact_frame = self.model.frames[frame_id].placement.translation
            rotation_world_to_joint_frame = self.data.oMi[joint_id].rotation.T

            f_world = forces[idx * 3 : (idx + 1) * 3]
            f_lin = rotation_world_to_joint_frame @ f_world
            f_ang = ca.cross(translation_joint_to_contact_frame, f_lin)
            f = ca.vertcat(f_lin, f_ang)
            f_ext[joint_id] = cpin.Force(f)

        tau_srb = cpin.rnea(self.model, self.data, q_fixed, v, a, f_ext)  # q_fixed!

        return ca.Function("srb_dyn", [q_b, q_j, q_j_fixed, v_b, a_b, forces], [tau_srb], ["q_b", "q_j", "q_j_fixed", "v_b", "a_b", "forces"], ["tau_srb"])
    
    def get_frame_position(self, frame_id):
        q = ca.SX.sym("q", self.nq)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.updateFramePlacement(self.model, self.data, frame_id)
        pos = self.data.oMf[frame_id].translation

        return ca.Function("frame_pos", [q], [pos], ["q"], ["pos"])

    def get_frame_velocity(self, frame_id, ref=pin.LOCAL_WORLD_ALIGNED):
        q = ca.SX.sym("q", self.nq)
        dq = ca.SX.sym("dq", self.nv)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q, dq)
        vel = cpin.getFrameVelocity(self.model, self.data, frame_id, ref).vector

        return ca.Function("frame_vel", [q, dq], [vel], ["q", "dq"], ["vel"])