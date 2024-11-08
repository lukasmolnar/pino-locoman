from os.path import dirname, abspath

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