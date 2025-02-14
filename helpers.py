from os.path import dirname, abspath

import numpy as np
import pinocchio as pin
import casadi as ca
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

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7  # without base position and quaternion

        # Nominal state: COM + DOFs
        self.x_nom = np.concatenate((np.zeros(6), self.q0))

        # DX indicies that are optimized
        self.nx = len(self.x_nom)
        self.ndx = self.nx - 1  # exclude quaternion
        self.dx_opt_indices = np.arange(self.ndx)  # all by default

        self.arm_ee_id = None

    def set_gait_sequence(self, gait_type, gait_nodes, dt):
        self.gait_sequence = GaitSequence(gait_type, gait_nodes, dt)
        self.ee_ids = [self.model.getFrameId(f) for f in self.gait_sequence.feet]
        self.nf = 12  # forces at feet

    def add_arm_task(self, f_des, vel_des=None):
        self.nf += 3
        self.arm_ee_id = self.model.getFrameId("gripperMover")
        self.arm_f_des = f_des
        self.arm_vel_des = vel_des

    def initialize_weights(self, dynamics):
        if dynamics == "centroidal":
            self.Q_diag = np.concatenate((
                [1000] * 6,  # com
                [10] * 2,  # base x/y
                [500],  # base z
                [500] * 3,  # base rot
                [10] * self.nj,  # joint pos
            ))
            self.R_diag = np.concatenate((
                [1e-3] * self.nf,  # forces
                [1e-1] * self.nj,  # joint vel
            ))

        elif dynamics == "rnea":
            Q_base_pos_diag = np.concatenate((
                [0] * 2,  # base x/y
                [1000],  # base z
                [10000] * 2,  # base rot x/y
                [0],  # base rot z
            ))
            Q_joint_pos_diag = np.tile([1000, 100, 100], 4)  # hip, thigh, calf

            if self.arm_ee_id:
                Q_joint_pos_diag = np.concatenate((Q_joint_pos_diag, [100] * 6))  # arm

            assert(len(Q_joint_pos_diag) == self.nj)

            Q_vel_diag = np.concatenate((
                [1000] * 3,  # base linear
                [1000] * 3,  # base angular
                [1] * self.nj,  # joint vel (all of them)
            ))

            self.Q_diag = np.concatenate((Q_base_pos_diag, Q_joint_pos_diag, Q_vel_diag))
            self.R_diag = np.concatenate((
                [1e-3] * self.nv,  # velocities
                [1e-3] * self.nf,  # forces
                [1e-3] * self.nj,  # joint torques
            ))

            # Additional weights
            self.W_diag = np.concatenate((
                [1e-1] * self.nj,  # keep tau_0 close to tau_prev
                [1e-2] * self.nj,  # keep tau_1 close to tau_prev
            ))

        else:
            raise ValueError(f"Dynamics: {dynamics} not supported")

        self.Q = np.diag(self.Q_diag)
        self.R = np.diag(self.R_diag)

        if dynamics == "centroidal":
            # Only consider optimized indices
            # TODO: Add this for RNEA
            self.Q = self.Q[self.dx_opt_indices][:, self.dx_opt_indices]


class B2(Robot):
    def __init__(self, reference_pose="standing"):
        urdf_path = "b2_description/urdf/b2.urdf"
        srdf_path = "b2_description/srdf/b2.srdf"
        super().__init__(urdf_path, srdf_path, reference_pose)

        # Joint limits (tiled: hip, thigh, calf)
        self.joint_pos_min = np.tile([-0.87, -0.94, -2.82], 4)
        self.joint_pos_max = np.tile([0.87, 4.69, -0.43], 4)
        self.joint_vel_max = np.tile([23.0, 23.0, 14.0], 4)
        self.joint_torque_max = np.tile([200, 200, 320], 4)


class Go2(Robot):
    def __init__(self, reference_pose="standing"):
        urdf_path = "go2_description/urdf/go2.urdf"
        srdf_path = "go2_description/srdf/go2.srdf"
        super().__init__(urdf_path, srdf_path, reference_pose)

        # Joint limits (tiled: hip, thigh, calf)
        self.joint_pos_min = np.tile([-1.0472, -1.5708, -2.7227], 4)
        self.joint_pos_max = np.tile([1.0472, 3.4907, -0.83776], 4)
        self.joint_vel_max = np.tile([30.1, 30.1, 15.70], 4)
        self.joint_torque_max = np.tile([23.7, 23.7, 45.43], 4)


class B2G(Robot):
    def __init__(self, reference_pose="standing_with_arm_up", ignore_arm=False):
        urdf_path = "b2g_description/urdf/b2g.urdf"
        srdf_path = "b2g_description/srdf/b2g.srdf"
        super().__init__(urdf_path, srdf_path, reference_pose)

        # Joint limits (tiled: hip, thigh, calf)
        # TODO: arm joints
        self.joint_pos_min = np.tile([-0.87, -0.94, -2.82], 4)
        self.joint_pos_max = np.tile([0.87, 4.69, -0.43], 4)
        self.joint_vel_max = np.tile([23.0, 23.0, 14.0], 4)
        self.joint_torque_max = np.tile([200, 200, 320], 4)

        # Ignore gripper joint in OCP
        self.dx_opt_indices = self.dx_opt_indices[:-1]

        self.ignore_arm = ignore_arm    
        if self.ignore_arm:
            # Ignore all arm joints in OCP
            self.dx_opt_indices = self.dx_opt_indices[:-6]


class GaitSequence:
    def __init__(self, gait_type="trot", gait_nodes=20, dt=0.02):
        self.feet = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.gait_nodes = gait_nodes
        self.dt = dt

        # Separate in contact (0 or 1) from bezier index (from 0 to 1)
        self.contact_schedule = np.ones((4, gait_nodes))  # default: in contact
        self.bezier_schedule = np.zeros((4, gait_nodes))  # default: index 0

        if gait_type == "trot":
            self.N = gait_nodes // 2
            self.n_contacts = 2
            for i in range(gait_nodes):
                bez_idx = i % self.N / (self.N - 1)  # normalize to [0, 1]
                if i < self.N:
                    # FR, RL in swing
                    self.contact_schedule[0, i] = 0
                    self.contact_schedule[3, i] = 0
                    self.bezier_schedule[0, i] = bez_idx
                    self.bezier_schedule[3, i] = bez_idx
                else:
                    # FL, RR in swing
                    self.contact_schedule[1, i] = 0
                    self.contact_schedule[2, i] = 0
                    self.bezier_schedule[1, i] = bez_idx
                    self.bezier_schedule[2, i] = bez_idx

        elif gait_type == "walk":
            self.N = gait_nodes // 4
            self.n_contacts = 3
            for i in range(gait_nodes):
                bez_idx = i % self.N / (self.N - 1)  # normalize to [0, 1]
                if i < self.N:
                    # FL in swing
                    self.contact_schedule[1, i] = 0
                    self.bezier_schedule[1, i] = bez_idx
                elif i < 2 * self.N:
                    # RR in swing
                    self.contact_schedule[2, i] = 0
                    self.bezier_schedule[2, i] = bez_idx
                elif i < 3 * self.N:
                    # FR in swing
                    self.contact_schedule[0, i] = 0
                    self.bezier_schedule[0, i] = bez_idx
                else:
                    # RL in swing
                    self.contact_schedule[3, i] = 0
                    self.bezier_schedule[3, i] = bez_idx

        elif gait_type == "stand":
            self.N = gait_nodes
            self.n_contacts = 4

        else:
            raise ValueError(f"Gait: {gait_type} not supported")

    def shift_contact_schedule(self, shift_idx):
        shift_idx %= self.gait_nodes
        return np.roll(self.contact_schedule, -shift_idx, axis=1)

    def shift_bezier_schedule(self, shift_idx):
        shift_idx %= self.gait_nodes
        return np.roll(self.bezier_schedule, -shift_idx, axis=1)

    def get_bezier_vel_z(self, p0_z, idx, h=0.1):
        # NOTE: idx needs to be normalized to [0, 1]
        T = self.N * self.dt  # period in seconds

        # Implementation from crl-loco
        vel_z = ca.if_else(
            idx < 0.5,
            self.cubic_bezier_derivative(p0_z, h, 2 * idx),
            self.cubic_bezier_derivative(h, p0_z, 2 * idx - 1)
        ) * 2 / T

        return vel_z

    def cubic_bezier_derivative(self, p0, p1, idx):
        return 6 * idx * (1 - idx) * (p1 - p0)