from os.path import dirname, abspath

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper



class Robot:
    def __init__(self, urdf_path, srdf_path, reference_pose, use_quaternion=True):
        urdf_dir = dirname(abspath(urdf_path))
        if use_quaternion:
            joint_model = pin.JointModelFreeFlyer()
        else:
            joint_model = pin.JointModelComposite()
            joint_model.addJoint(pin.JointModelTranslation())
            joint_model.addJoint(pin.JointModelSphericalZYX())
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [urdf_dir], joint_model)
        self.model = self.robot.model
        self.data = self.robot.data
        if srdf_path and reference_pose:
            pin.loadReferenceConfigurations(self.model, srdf_path)
            self.q0 = self.model.referenceConfigurations[reference_pose]

        # Set nominal state: COM + joint positions
        self.x_nom = np.concatenate((np.zeros(6), self.q0))

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7  # without base position and quaternion

        self.Q_weights = {
            "scaling": 1e0,
            "com": 100,
            "base_xy": 0,
            "base_z": 10,
            "base_rot": 1000,
            "joints": 10,
        }
        self.R_weights = {
            "scaling": 1e-3,
            "forces": 1,
            "joints": 100,
        }

    def set_gait_sequence(self, gait, nodes, dt, arm_task=False):
        self.gait_sequence = GaitSequence(gait, nodes, dt)
        self.nf = 3 * self.gait_sequence.n_contacts
        if arm_task:
            self.nf += 3
            self.arm_ee = "gripperMover"
            self.arm_f_des = [-100, 0, 0]  # desired force at end-effector
        else:
            self.arm_ee = None

        # Weight matrices
        Q_diag = np.concatenate((
            [self.Q_weights["com"]] * 6,
            [self.Q_weights["base_xy"]] * 2,
            [self.Q_weights["base_z"]],
            [self.Q_weights["base_rot"]] * 3,
            [self.Q_weights["joints"]] * self.nj
        ))
        R_diag = np.concatenate((
            [self.R_weights["forces"]] * self.nf,
            [self.R_weights["joints"]] * self.nj
        ))
        self.Q = self.Q_weights["scaling"] * np.diag(Q_diag)
        self.R = self.R_weights["scaling"] * np.diag(R_diag)


class B2(Robot):
    def __init__(self, reference_pose="standing"):
        urdf_path = "b2_description/urdf/b2.urdf"
        srdf_path = "b2_description/srdf/b2.srdf"
        super().__init__(urdf_path, srdf_path, reference_pose)


class B2G(Robot):
    def __init__(self, reference_pose="standing_with_arm_up"):
        urdf_path = "b2g_description/urdf/b2g.urdf"
        srdf_path = "b2g_description/srdf/b2g.srdf"
        super().__init__(urdf_path, srdf_path, reference_pose)


class GaitSequence:
    def __init__(self, gait="trot", nodes=20, dt=0.02):
        feet = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.nodes = nodes
        self.dt = dt
        self.contact_list = []
        self.swing_list = []

        if gait == "trot":
            self.N = nodes // 2
            self.n_contacts = 2
            for i in range(nodes):
                if i < self.N:
                    contact = [feet[0], feet[3]]
                    swing = [feet[1], feet[2]]
                else:
                    contact = [feet[1], feet[2]]
                    swing = [feet[0], feet[3]]
                self.contact_list.append(contact)
                self.swing_list.append(swing)

        if gait == "walk":
            self.N = nodes  # for now just 1 step
            self.n_contacts = 3
            for i in range(nodes):
                if i < self.N:
                    contact = [feet[0], feet[2], feet[3]]
                    swing = [feet[1]] # FL
                # elif i < 2 * self.N:
                #     contact = [feet[0], feet[1], feet[3]]
                #     swing = [feet[2]] # RR 
                # elif i < 3 * self.N:
                #     contact = [feet[1], feet[2], feet[3]]
                #     swing = [feet[0]] # FR
                # else:
                #     contact = [feet[0], feet[1], feet[2]]
                #     swing = [feet[3]] # RL
                self.contact_list.append(contact)
                self.swing_list.append(swing)

        if gait == "stand":
            self.N = nodes
            self.n_contacts = 4
            for i in range(nodes):
                contact = [feet[0], feet[1], feet[2], feet[3]]
                self.contact_list.append(contact)
                self.swing_list.append([])

    def get_bezier_vel_z(self, p0_z, idx, h=0.1):
        T = self.N * self.dt
        if idx < self.N:
            t = idx * self.dt
        else:
            t = (idx - self.N) * self.dt

        p1_z = p0_z + h
        p2_z = p0_z
        return 2 * (1 - t / T)**2 * (p1_z - p0_z) / T + 2 * (t / T) * (p2_z - p1_z) / T