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
        else:
            self.q0 = self.robot.q0

        # Set nominal state: COM + joint positions
        self.x_nom = np.concatenate((np.zeros(6), self.q0))

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7  # without base position and quaternion

        self.Q_weights = {
            "scaling": 1e0,
            "com": 100,
            "base_xy": 0,
            "base_z": 500,
            "base_rot": 500,
            "joints": 10,
        }
        self.R_weights = {
            "scaling": 1e-3,
            "forces": 1,
            "joints": 100,
        }
        self.arm_ee = None

    def set_gait_sequence(self, gait_type, gait_nodes, dt):
        self.gait_sequence = GaitSequence(gait_type, gait_nodes, dt)
        self.nf = 12  # forces at feet

    def add_arm_task(self, f_des, vel_des=None):
        self.nf += 3
        self.arm_ee = "gripperMover"
        self.arm_f_des = f_des
        self.arm_vel_des = vel_des

    def initialize_weights(self):
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
    def __init__(self, gait_type="trot", gait_nodes=20, dt=0.02):
        self.feet = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.gait_nodes = gait_nodes
        self.dt = dt

        # 0: in contact, otherwise index of bezier curve (starting at 1)
        self.contact_schedule = np.zeros((4, gait_nodes))

        if gait_type == "trot":
            self.N = gait_nodes // 2
            self.n_contacts = 2
            for i in range(gait_nodes):
                bez_idx = i % self.N + 1  # start at 1 to distinguish from contact
                if i < self.N:
                    # FL, RR in swing
                    self.contact_schedule[1, i] = bez_idx
                    self.contact_schedule[2, i] = bez_idx
                else:
                    # FR, RL in swing
                    self.contact_schedule[0, i] = bez_idx
                    self.contact_schedule[3, i] = bez_idx

        elif gait_type == "walk":
            self.N = gait_nodes // 4
            self.n_contacts = 3
            for i in range(gait_nodes):
                bez_idx = i % self.N + 1  # start at 1 to distinguish from contact
                if i < self.N:
                    # FL in swing
                    self.contact_schedule[1, i] = bez_idx
                elif i < 2 * self.N:
                    # RR in swing
                    self.contact_schedule[2, i] = bez_idx
                elif i < 3 * self.N:
                    # FR in swing
                    self.contact_schedule[0, i] = bez_idx
                else:
                    # RL in swing
                    self.contact_schedule[3, i] = bez_idx

        elif gait_type == "stand":
            self.N = gait_nodes
            self.n_contacts = 4

        else:
            raise ValueError(f"Gait: {gait_type} not supported")

    def shift_contact_schedule(self, shift_idx):
        shift_idx %= self.gait_nodes
        return np.roll(self.contact_schedule, -shift_idx, axis=1)
    
    def get_bezier_vel_z(self, p0_z, idx, h=0.1):
        # NOTE: idx needs to be in [0, N)
        t = idx * self.dt
        T = self.N * self.dt

        p1_z = p0_z + h
        p2_z = p0_z
        return 2 * (1 - t / T)**2 * (p1_z - p0_z) / T + 2 * (t / T) * (p2_z - p1_z) / T