import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi


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
        x = casadi.SX.sym("x", self.nq + self.nv)
        dx = casadi.SX.sym("dx", self.nv + self.nv)

        q = x[:self.nq]
        dq = dx[:self.nv]
        q_next = cpin.integrate(self.model, q, dq)

        v = x[self.nq:]
        dv = dx[self.nv:]
        v_next = v + dv

        x_next = casadi.vertcat(q_next, v_next)

        return casadi.Function("integrate", [x, dx], [x_next], ["x", "dx"], ["x_next"])
    
    def state_difference(self):
        x0 = casadi.SX.sym("x0", self.nq + self.nv)
        x1 = casadi.SX.sym("x1", self.nq + self.nv)

        q0 = x0[:self.nq]
        q1 = x1[:self.nq]
        v0 = x0[self.nq:]
        v1 = x1[self.nq:]

        dq = cpin.difference(self.model, q0, q1)
        dv = v1 - v0
        dx = casadi.vertcat(dq, dv)

        return casadi.Function("difference", [x0, x1], [dx], ["x0", "x1"], ["dx"])

    def rnea_dynamics(self, arm_ee_id=None):
        # States
        q = casadi.SX.sym("q", self.nq)  # positions
        v = casadi.SX.sym("v", self.nv)  # velocities
        a = casadi.SX.sym("a", self.nv)  # accelerations

        # Inputs
        nf = len(self.ee_ids)
        if arm_ee_id:
            nf += 1
        forces = casadi.SX.sym("forces", 3 * nf)  # end-effector forces

        # RNEA
        cpin.framesForwardKinematics(self.model, self.data, q)
        f_ext = [cpin.Force(casadi.SX.zeros(6)) for _ in range(self.model.njoints)]
        for idx, frame_id in enumerate(self.ee_ids):
            # TODO: Check this. it is from OCS2.
            joint_id = self.model.frames[frame_id].parentJoint
            translation_joint_to_contact_frame = self.model.frames[frame_id].placement.translation
            rotation_world_to_joint_frame = self.data.oMi[joint_id].rotation.T

            f_world = forces[idx * 3 : (idx + 1) * 3]
            f_lin = rotation_world_to_joint_frame @ f_world
            f_ang = casadi.cross(translation_joint_to_contact_frame, f_lin)
            f = casadi.vertcat(f_lin, f_ang)
            f_ext[joint_id] = cpin.Force(f)

        tau_rnea = cpin.rnea(self.model, self.data, q, v, a, f_ext)

        # Separate external forces
        # tau_rnea = cpin.rnea(self.model, self.data, q, v, a)
        # tau_ext = casadi.SX.zeros(self.nv)
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

        return casadi.Function("rnea_dyn", [q, v, a, forces], [tau_rnea], ["q", "v", "a", "forces"], ["tau_rnea"])
    
    def get_frame_position(self, frame_id):
        q = casadi.SX.sym("q", self.nq)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.updateFramePlacement(self.model, self.data, frame_id)
        pos = self.data.oMf[frame_id].translation

        return casadi.Function("frame_pos", [q], [pos], ["q"], ["pos"])

    def get_frame_velocity(self, frame_id, ref=pin.LOCAL_WORLD_ALIGNED):
        q = casadi.SX.sym("q", self.nq)
        dq = casadi.SX.sym("dq", self.nv)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q, dq)
        vel = cpin.getFrameVelocity(self.model, self.data, frame_id, ref).vector

        return casadi.Function("frame_vel", [q, dq], [vel], ["q", "dq"], ["vel"])