from os.path import dirname, abspath

import numpy as np
import pinocchio as pin
import casadi as ca
from pinocchio.robot_wrapper import RobotWrapper


class Robot:
    def __init__(self, urdf_path, srdf_path, dynamics, reference_pose, use_quaternion=True, lock_joints=None):
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

        self.dynamics = dynamics

        # Set initial state
        if self.dynamics == "centroidal_vel":
            # COM linear/angular momentum + generalized coordinates
            self.x_init = np.concatenate((np.zeros(6), self.q0))

        elif self.dynamics == "centroidal_acc" or self.dynamics == "rnea": 
            # Generalized coordinates + velocities
            self.x_init = np.concatenate((self.q0, np.zeros(self.nv)))

        else:
            raise ValueError(f"Dynamics: {self.dynamics} not supported")

        self.ext_force_frame = None  # external force
        self.arm_ee_frame = None  # arm end-effector
        self.ignore_arm = True

    def set_gait_sequence(self, gait_type, gait_period):
        self.gait_sequence = GaitSequence(gait_type, gait_period)
        self.foot_frames = [self.model.getFrameId(f) for f in self.gait_sequence.feet]

    def initialize_weights(self):
        if self.dynamics == "centroidal_vel":
            self.Q_diag = np.concatenate((
                [1000] * 6,  # com
                [0] * 2,  # base x/y
                [1000],  # base z
                [10000] * 2,  # base rot x/y
                [0],  # base rot z
                [100] * 12,  # leg joint pos
            ))
            if not self.ignore_arm:
                self.Q_diag = np.concatenate((self.Q_diag, [1] * 6))  # arm joint pos

            self.R_diag = np.concatenate((
                [1e-3] * self.nf,  # forces
                [1e-1] * self.nj,  # joint vel
            ))

        elif self.dynamics == "centroidal_acc":
            Q_base_pos_diag = np.concatenate((
                [0] * 2,  # base x/y
                [10000],  # base z
                [10000] * 2,  # base rot x/y
                [0],  # base rot z
            ))
            Q_joint_pos_diag = np.tile([1000, 1000, 100], 4)  # hip, thigh, calf

            if not self.ignore_arm:
                Q_joint_pos_diag = np.concatenate((Q_joint_pos_diag, [100] * 6))  # arm

            assert(len(Q_joint_pos_diag) == self.nj)

            Q_vel_diag = np.concatenate((
                [1000] * 2,  # base lin x/y
                [10000],  # base lin z
                [1000] * 3,  # base ang
                [1] * self.nj,  # joint vel (all of them)
            ))

            self.Q_diag = np.concatenate((Q_base_pos_diag, Q_joint_pos_diag, Q_vel_diag))
            self.R_diag = np.concatenate((
                [1e-3] * self.nj,  # joint acc
                [1e-3] * self.nf,  # forces
            ))

        elif self.dynamics == "rnea":
            Q_base_pos_diag = np.concatenate((
                [0] * 2,  # base x/y
                [1000],  # base z
                [10000] * 2,  # base rot x/y
                [0],  # base rot z
            ))
            Q_joint_pos_diag = np.tile([1000, 500, 500], 4)  # hip, thigh, calf

            if not self.ignore_arm:
                Q_joint_pos_diag = np.concatenate((Q_joint_pos_diag, [100] * 6))  # arm

            assert(len(Q_joint_pos_diag) == self.nj)

            Q_vel_diag = np.concatenate((
                [2000] * 2,  # base lin x/y
                [1000],  # base lin z
                [1000] * 2,  # base ang x/y
                [2000],  # base ang z
                [2] * self.nj,  # joint vel (all of them)
            ))

            self.Q_diag = np.concatenate((Q_base_pos_diag, Q_joint_pos_diag, Q_vel_diag))
            self.R_diag = np.concatenate((
                [1e-3] * self.nv,  # accelerations
                [1e-3] * self.nf,  # forces
                [1e-4] * self.nj,  # joint torques
            ))

            # Additional weights
            self.W_diag = np.concatenate((
                [0] * self.nj,  # keep tau_0 close to tau_prev
            ))


class B2(Robot):
    def __init__(self, dynamics, reference_pose="standing", payload=None):
        urdf_path = "robots/b2_description/urdf/b2.urdf"
        srdf_path = "robots/b2_description/srdf/b2.srdf"
        super().__init__(urdf_path, srdf_path, dynamics, reference_pose)

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


class Go2(Robot):
    def __init__(self, dynamics, reference_pose="standing"):
        urdf_path = "robots/go2_description/urdf/go2.urdf"
        srdf_path = "robots/go2_description/srdf/go2.srdf"
        super().__init__(urdf_path, srdf_path, dynamics, reference_pose)

        # Joint limits (tiled: hip, thigh, calf)
        self.joint_pos_min = np.tile([-1.0472, -1.5708, -2.7227], 4)
        self.joint_pos_max = np.tile([1.0472, 3.4907, -0.83776], 4)
        self.joint_vel_max = np.tile([30.1, 30.1, 15.70], 4)
        self.joint_torque_max = np.tile([23.7, 23.7, 45.43], 4)


class B2G(Robot):
    def __init__(self, dynamics, reference_pose="standing_with_arm_up", ignore_arm=False):
        urdf_path = "robots/b2g_description/urdf/b2g.urdf"
        srdf_path = "robots/b2g_description/srdf/b2g.srdf"
        if ignore_arm:
            lock_joints = range(14, 21)  # all arm joints
        else:
            lock_joints = [20]  # just gripper joint

        super().__init__(urdf_path, srdf_path, dynamics, reference_pose, lock_joints=lock_joints)
        self.ignore_arm = ignore_arm

        # Leg joint limits (tiled: hip, thigh, calf)
        self.joint_pos_min = np.tile([-0.87, -0.94, -2.82], 4)
        self.joint_pos_max = np.tile([0.87, 4.69, -0.43], 4)
        self.joint_vel_max = np.tile([23.0, 23.0, 14.0], 4)
        self.joint_torque_max = np.tile([200, 200, 320], 4)

        if not self.ignore_arm:
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


class GaitSequence:
    def __init__(self, gait_type="trot", gait_period=0.5):
        self.feet = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        self.gait_type = gait_type
        self.gait_period = gait_period

        if self.gait_type == "trot":
            self.n_contacts = 2
            self.swing_period = 0.5 * self.gait_period

        elif self.gait_type == "walk":
            self.n_contacts = 3
            self.swing_period = 0.25 * self.gait_period

        elif self.gait_type == "stand":
            self.n_contacts = 4
            self.swing_period = self.gait_period  # zero becomes degenerate

        else:
            raise ValueError(f"Gait: {self.gait_type} not supported")

    def get_gait_schedule(self, t_current, dts, nodes):
        """
        Returns contact and swing schedules for the horizon, given the time steps in dts
        """
        contact_schedule = np.ones((4, nodes))  # in_contact: 0 or 1
        swing_schedule = np.zeros((4, nodes))  # swing_phase: from 0 to 1

        if self.gait_type == "trot":
            t = t_current
            for i in range(nodes):
                if i > 0:
                    t += dts[i - 1]
                gait_phase = t % self.gait_period / self.gait_period
                swing_phase = t % self.swing_period / self.swing_period
                if gait_phase < 0.5:
                    # FR, RL in swing
                    contact_schedule[0, i] = 0
                    contact_schedule[3, i] = 0
                    swing_schedule[0, i] = swing_phase
                    swing_schedule[3, i] = swing_phase
                else:
                    # FL, RR in swing
                    contact_schedule[1, i] = 0
                    contact_schedule[2, i] = 0
                    swing_schedule[1, i] = swing_phase
                    swing_schedule[2, i] = swing_phase

        elif self.gait_type == "walk":
            t = t_current
            for i in range(nodes):
                if i > 0:
                    t += dts[i - 1]
                gait_phase = t % self.gait_period / self.gait_period
                swing_phase = t % self.swing_period / self.swing_period
                if gait_phase < 0.25:
                    # FL in swing
                    contact_schedule[1, i] = 0
                    swing_schedule[1, i] = swing_phase
                elif gait_phase < 0.5:
                    # RR in swing
                    contact_schedule[2, i] = 0
                    swing_schedule[2, i] = swing_phase
                elif gait_phase < 0.75:
                    # FR in swing
                    contact_schedule[0, i] = 0
                    swing_schedule[0, i] = swing_phase
                else:
                    # RL in swing
                    contact_schedule[3, i] = 0
                    swing_schedule[3, i] = swing_phase

        return contact_schedule, swing_schedule


"""
Swing trajectory helpers
"""
def get_bezier_vel_z(swing_phase, swing_period, h_max=0.1):
    # Implementation from crl-loco
    vel_z = ca.if_else(
        swing_phase < 0.5,
        cubic_bezier_derivative(0, h_max, 2 * swing_phase),
        cubic_bezier_derivative(h_max, 0, 2 * swing_phase - 1)
    ) * 2 / swing_period

    return vel_z

def cubic_bezier_derivative(p0, p1, phase):
    return 6 * phase * (1 - phase) * (p1 - p0)

def get_spline_vel_z(swing_phase, swing_period, h_max=0.1, v_liftoff=0.1, v_touchdown=-0.2):
    mid_time = swing_period / 2
    spline1 = CubicSpline(0, mid_time, 0, v_liftoff, h_max, 0)
    spline2 = CubicSpline(mid_time, swing_period, h_max, 0, 0, v_touchdown)

    vel_z = ca.if_else(
        swing_phase < 0.5,
        spline1.velocity(swing_phase * swing_period),
        spline2.velocity(swing_phase * swing_period)
    )

    return vel_z


class CubicSpline:
    """
    Implementation from OCS2
    """
    def __init__(self, t0, t1, pos0, vel0, pos1, vel1):
        self.t0 = t0
        self.t1 = t1
        self.dt = t1 - t0

        dpos = pos1 - pos0
        dvel = vel1 - vel0

        self.c0 = pos0
        self.c1 = vel0 * self.dt
        self.c2 = -(3.0 * vel0 + dvel) * self.dt + 3.0 * dpos
        self.c3 = (2.0 * vel0 + dvel) * self.dt - 2.0 * dpos
    
    def position(self, t):
        tn = (t - self.t0) / self.dt
        return self.c3 * tn**3 + self.c2 * tn**2 + self.c1 * tn + self.c0

    def velocity(self, t):
        tn = (t - self.t0) / self.dt
        return (3.0 * self.c3 * tn**2 + 2.0 * self.c2 * tn + self.c1) / self.dt
