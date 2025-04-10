from os.path import dirname, abspath

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from .gait_sequence import GaitSequence


class Robot:
    def __init__(self, urdf_path, srdf_path, reference_pose, use_quaternion=True, lock_joints=None):
        urdf_dir = dirname(abspath(urdf_path))
        if use_quaternion:
            joint_model = pin.JointModelFreeFlyer()
        else:
            joint_model = pin.JointModelComposite()
            joint_model.addJoint(pin.JointModelTranslation())
            joint_model.addJoint(pin.JointModelSphericalZYX())

        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [urdf_dir], joint_model)
        if lock_joints:
            self.robot = self.robot.buildReducedRobot(lock_joints)

        self.model = self.robot.model
        self.data = self.robot.data
        if srdf_path and reference_pose:
            pin.loadReferenceConfigurations(self.model, srdf_path)
            self.q0 = self.model.referenceConfigurations[reference_pose]
        else:
            self.q0 = self.robot.q0

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7  # without base position and quaternion
        self.nf = 12  # forces at feet

        self.ext_force_frame = None  # external force
        self.arm_ee_frame = None  # arm end-effector

    def set_gait_sequence(self, gait_type, gait_period):
        self.gait_sequence = GaitSequence(gait_type, gait_period)
        self.foot_frames = [self.model.getFrameId(f) for f in self.gait_sequence.feet]


class Go2(Robot):
    def __init__(self, reference_pose="standing"):
        urdf_path = "robots/go2_description/urdf/go2.urdf"
        srdf_path = "robots/go2_description/srdf/go2.srdf"
        super().__init__(urdf_path, srdf_path, reference_pose)

        # Joint limits (tiled: hip, thigh, calf)
        self.joint_pos_min = np.tile([-1.0472, -1.5708, -2.7227], 4)
        self.joint_pos_max = np.tile([1.0472, 3.4907, -0.83776], 4)
        self.joint_vel_max = np.tile([30.1, 30.1, 15.70], 4)
        self.joint_torque_max = np.tile([23.7, 23.7, 45.43], 4)


class B2(Robot):
    def __init__(self, reference_pose="standing", payload=None):
        urdf_path = "robots/b2_description/urdf/b2.urdf"
        srdf_path = "robots/b2_description/srdf/b2.srdf"
        super().__init__(urdf_path, srdf_path, reference_pose)

        # Joint limits (tiled: hip, thigh, calf)
        self.joint_pos_min = np.tile([-0.87, -0.94, -2.82], 4)
        self.joint_pos_max = np.tile([0.87, 4.69, -0.43], 4)
        self.joint_vel_max = np.tile([23.0, 23.0, 14.0], 4)
        self.joint_torque_max = np.tile([200, 200, 320], 4)

        # External force as payload on base
        if payload == "front":
            self.ext_force_frame = self.model.getFrameId("payload_joint_front", type=pin.FIXED_JOINT)
            self.nf += 3
        elif payload == "rear":
            self.ext_force_frame = self.model.getFrameId("payload_joint_rear", type=pin.FIXED_JOINT)
            self.nf += 3


class B2G(Robot):
    def __init__(self, reference_pose="standing_with_arm_up", ignore_arm=False):
        urdf_path = "robots/b2g_description/urdf/b2g.urdf"
        srdf_path = "robots/b2g_description/srdf/b2g.srdf"
        if ignore_arm:
            lock_joints = range(14, 21)  # all arm joints
        else:
            lock_joints = [20]  # just gripper joint

        super().__init__(urdf_path, srdf_path, reference_pose, lock_joints=lock_joints)

        # Leg joint limits (tiled: hip, thigh, calf)
        self.joint_pos_min = np.tile([-0.87, -0.94, -2.82], 4)
        self.joint_pos_max = np.tile([0.87, 4.69, -0.43], 4)
        self.joint_vel_max = np.tile([23.0, 23.0, 14.0], 4)
        self.joint_torque_max = np.tile([200, 200, 320], 4)

        if not ignore_arm:
            # External force at gripper
            self.ext_force_frame = self.model.getFrameId("gripperStator", type=pin.FIXED_JOINT)
            self.arm_ee_frame = self.model.getFrameId("gripperStator", type=pin.FIXED_JOINT)
            self.nf += 3

            # Arm joint limits
            self.joint_pos_min = np.concatenate((
                self.joint_pos_min,
                [-2.62, 0.0, -2.88, -1.52, -1.34, -2.79]
            ))
            self.joint_pos_max = np.concatenate((
                self.joint_pos_max,
                [2.62, 2.97, 0.0, 1.52, 1.34, 2.79]
            ))
            self.joint_vel_max = np.concatenate((
                self.joint_vel_max,
                [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
            ))
            self.joint_torque_max = np.concatenate((
                self.joint_torque_max,
                [30, 60, 30, 30, 30, 30]
            ))
