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
            opt_dofs,
        ):
        self.model = cpin.Model(model)
        self.data = self.model.createData()
        self.mass = mass
        self.ee_ids = ee_ids
        self.opt_dofs = opt_dofs

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7  # without base position and quaternion

    def state_integrate(self):
        x = ca.SX.sym("x", self.nq + self.nv)
        dq_opt = ca.SX.sym("dq_opt", len(self.opt_dofs))
        dv_opt = ca.SX.sym("dv_opt", len(self.opt_dofs))
        dx_opt = ca.vertcat(dq_opt, dv_opt)

        # The other DOFs remain zero
        dq = ca.SX.zeros(self.nv)
        dv = ca.SX.zeros(self.nv)
        dq[self.opt_dofs] = dq_opt
        dv[self.opt_dofs] = dv_opt

        q = x[:self.nq]
        q_next = cpin.integrate(self.model, q, dq)

        v = x[self.nq:]
        v_next = v + dv

        x_next = ca.vertcat(q_next, v_next)

        return ca.Function("integrate", [x, dx_opt], [x_next], ["x", "dx_opt"], ["x_next"])
    
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

        # Inputs
        nf = len(self.ee_ids)
        if arm_ee_id:
            nf += 1
        forces = ca.SX.sym("forces", 3 * nf)  # end-effector forces
        a_opt = ca.SX.sym("a_opt", len(self.opt_dofs))  # accelerations

        # The other DOFs remain zero
        a = ca.SX.zeros(self.nv)
        a[self.opt_dofs] = a_opt

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

        return ca.Function("rnea_dyn", [q, v, a_opt, forces], [tau_rnea], ["q", "v", "a", "forces"], ["tau_rnea"])

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