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

def swing_bezier_pos(p0, t, T, l=0.0, h=0.1):
    p1 = p0 + np.array([l/2, 0, h])
    p2 = p0 + np.array([l, 0, 0])
    return (1 - t / T)**2 * p0 + 2 * (1 - t / T) * t / T * p1 + (t / T)**2 * p2

def swing_bezier_vel(p0, t, T, l=0.0, h=0.1):
    p1 = p0 + np.array([l/2, 0, h])
    p2 = p0 + np.array([l, 0, 0])
    return 2 * (1 - t / T)**2 * (p1 - p0) / T + 2 * (t / T) * (p2 - p1) / T
