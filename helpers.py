from os.path import dirname, abspath

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper


class Robot:
    def __init__(self, urdf_path, srdf_path, reference_pose):
        urdf_dir = dirname(abspath(urdf_path))
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [urdf_dir], pin.JointModelFreeFlyer())
        self.model = self.robot.model
        self.data = self.robot.data
        if srdf_path and reference_pose:
            pin.loadReferenceConfigurations(self.model, srdf_path)
            self.q0 = self.model.referenceConfigurations[reference_pose]

class B2(Robot):
    def __init__(self):
        urdf_path = "b2_description/urdf/b2.urdf"
        srdf_path = "b2_description/srdf/b2.srdf"
        reference_pose = "standing"
        super().__init__(urdf_path, srdf_path, reference_pose)

        self.x_nom = [
            0, 0, 0, 0, 0, 0,            # centroidal momentum
            0, 0, 0.55, 0, 0, 0, 1,      # base position and quaternion
            0, 0.7, -1.5, 0, 0.7, -1.5,  # FL, FR joints
            0, 0.7, -1.5, 0, 0.7, -1.5,  # RL, RR joints
        ]

class B2G(Robot):
    def __init__(self):
        urdf_path = "b2g_description/urdf/b2g.urdf"
        srdf_path = "b2g_description/srdf/b2g.srdf"
        reference_pose = "standing_with_arm_home"
        super().__init__(urdf_path, srdf_path, reference_pose)

        self.x_nom = [
            0, 0, 0, 0, 0, 0,            # centroidal momentum
            0, 0, 0.55, 0, 0, 0, 1,      # base position and quaternion
            0, 0.7, -1.5, 0, 0.7, -1.5,  # FL, FR joints
            0, 0.7, -1.5, 0, 0.7, -1.5,  # RL, RR joints
            0, 0.26, -0.26, 0, 0, 0, 0   # arm joints
        ]


class GaitSequence:
    def __init__(self, gait="trot", nodes=40, dt=0.05):
        feet = feet=["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.nodes = nodes
        self.dt = dt
        self.contact_list = []
        self.swing_list = []

        if gait == "trot":
            self.N = nodes // 2  # number of nodes in one cycle
            self.n_contacts = 2  # at each time step
            for i in range(nodes):
                if i < self.N:
                    contact = [feet[0], feet[3]]
                    swing = [feet[1], feet[2]]
                else:
                    contact = [feet[1], feet[2]]
                    swing = [feet[0], feet[3]]
                self.contact_list.append(contact)
                self.swing_list.append(swing)

    def get_bezier_vel_z(self, p0_z, idx, h=0.1):
        T = self.N * self.dt
        if idx < self.N:
            t = idx * self.dt
        else:
            t = (idx - self.N) * self.dt

        p1_z = p0_z + h
        p2_z = p0_z
        return 2 * (1 - t / T)**2 * (p1_z - p0_z) / T + 2 * (t / T) * (p2_z - p1_z) / T