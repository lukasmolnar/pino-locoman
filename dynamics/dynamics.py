import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca


class Dynamics:
    def __init__(
            self,
            model,
            mass,
            feet_ids,
        ):
        self.model = cpin.Model(model)
        self.data = self.model.createData()
        self.mass = mass
        self.feet_ids = feet_ids

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

    def get_frame_velocity(self, frame_id, ref=pin.LOCAL_WORLD_ALIGNED):
        q = ca.SX.sym("q", self.nq)
        v = ca.SX.sym("v", self.nv)

        # Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q, v)
        vel = cpin.getFrameVelocity(self.model, self.data, frame_id, ref).vector

        return ca.Function("frame_vel", [q, v], [vel], ["q", "v"], ["vel"])
