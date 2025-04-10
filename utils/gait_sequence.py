import numpy as np
import casadi as ca


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
