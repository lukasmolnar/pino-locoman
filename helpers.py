from os.path import dirname, abspath

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

def load(urdf_path, srdf_path=None):
    urdf_dir = dirname(abspath(urdf_path))
    robot = RobotWrapper.BuildFromURDF(urdf_path, [urdf_dir], pin.JointModelFreeFlyer())
    if srdf_path:
        pin.loadReferenceConfigurations(robot.model, srdf_path)
        q0 = pin.neutral(robot.model)
        robot.q0 = pin.normalize(robot.model, q0)
    return robot


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