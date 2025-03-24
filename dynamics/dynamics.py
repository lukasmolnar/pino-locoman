import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca


class Dynamics:
    def __init__(
            self,
            model,
            mass,
            foot_frames,
        ):
        self.model = cpin.Model(model)
        self.data = self.model.createData()
        self.mass = mass
        self.foot_frames = foot_frames
        self.base_frame = self.model.getFrameId("base_link")

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7  # without base position and quaternion

    def state_integrate(self):
        pass

    def state_difference(self):
        pass

    def dynamics(self):
        pass

    def get_frame_position(self, frame_id):
        q = ca.SX.sym("q", self.nq)

        # Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.updateFramePlacement(self.model, self.data, frame_id)
        pos = self.data.oMf[frame_id].translation

        return ca.Function("frame_pos", [q], [pos], ["q"], ["pos"])

    def get_frame_velocity(self, frame_id, relative_to_base=False):
        q = ca.SX.sym("q", self.nq)
        v = ca.SX.sym("v", self.nv)

        # Pinocchio terms
        ref = pin.LOCAL_WORLD_ALIGNED
        cpin.forwardKinematics(self.model, self.data, q, v)
        frame_vel = cpin.getFrameVelocity(self.model, self.data, frame_id, ref).vector

        if relative_to_base:
            # Compute frame velocity relative to the base frame
            cpin.framesForwardKinematics(self.model, self.data, q)
            base_vel = cpin.getFrameVelocity(self.model, self.data, self.base_frame, ref).vector
            base_rot = self.data.oMf[self.base_frame].rotation

            frame_pos_world = self.data.oMf[frame_id].translation
            base_pos_world = self.data.oMf[self.base_frame].translation
            rel_pos_world = frame_pos_world - base_pos_world

            # Linear velocity correction due to base angular velocity
            base_ang_vel = base_vel[3:]
            correction = ca.cross(base_ang_vel, rel_pos_world)

            rel_lin_vel_world = frame_vel[:3] - base_vel[:3] - correction
            rel_ang_vel_world = frame_vel[3:] - base_vel[3:]

            # Rotate to the base frame
            rel_lin_vel_base = base_rot.T @ rel_lin_vel_world
            rel_ang_vel_base = base_rot.T @ rel_ang_vel_world

            vel = ca.vertcat(rel_lin_vel_base, rel_ang_vel_base)

        else:
            vel = frame_vel

        return ca.Function("frame_vel", [q, v], [vel], ["q", "v"], ["vel"])
