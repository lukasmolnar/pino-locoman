import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi


class CentroidalDynamics:
    def __init__(self, model, mass):
        self.model = cpin.Model(model)
        self.data = self.model.createData()
        self.mass = mass

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = 19
        self.dt = 0.02

    def state_integrate(self):
        x = casadi.SX.sym("x", 6 + self.nq)
        dx = casadi.SX.sym("dx", 6 + self.nv)

        h = x[:6]
        q = x[6:]
        dh = dx[:6]
        dq = dx[6:]

        h_next = h + dh
        q_next = cpin.integrate(self.model, q, dq)
        x_next = casadi.vertcat(h_next, q_next)

        return casadi.Function("integrate", [x, dx], [x_next], ["x", "dx"], ["x_next"])

    def state_difference(self):
        x0 = casadi.SX.sym("x0", 6 + self.nq)
        x1 = casadi.SX.sym("x1", 6 + self.nq)

        h0 = x0[:6]
        q0 = x0[6:]
        h1 = x1[:6]
        q1 = x1[6:]

        h_diff = h1 - h0
        q_diff = cpin.difference(self.model, q0, q1)
        x_diff = casadi.vertcat(h_diff, q_diff)

        return casadi.Function("difference", [x0, x1], [x_diff], ["x0", "x1"], ["x_diff"])

    def centroidal_dynamics(self, contact_ee_ids):
        # States
        p_com = casadi.SX.sym("p_com", 3)  # COM linear momentum
        l_com = casadi.SX.sym("l_com", 3)  # COM angular momentum
        q = casadi.SX.sym("q", self.nq)  # generalized coordinates (base + joints)

        # Inputs
        f_e = [casadi.SX.sym(f"f_e_{i}", 3) for i in range(len(contact_ee_ids))]  # end-effector forces
        dq_j = casadi.SX.sym("v", self.nj)  # joint velocities

        # Base velocity
        dq_b = casadi.SX.sym("dq_b", 6)
        dq = casadi.vertcat(dq_b, dq_j)

        # TODO: Check Pinocchio terms
        cpin.computeAllTerms(self.model, self.data, q, dq)

        # COM Dynamics
        g = np.array([0, 0, -9.81 * self.mass])
        dp_com = sum(f_e) + g
        dl_com = casadi.SX.zeros(3)
        for idx, frame_id in enumerate(contact_ee_ids):
            r_ee = self.data.oMf[frame_id].translation # - self.data.com[0]
            dl_com += casadi.cross(r_ee, f_e[idx])

        h = casadi.vertcat(p_com, l_com)
        dh = casadi.vertcat(dp_com, dl_com)

        # Stack states and inputs
        x = casadi.vertcat(h, q)
        dx = casadi.vertcat(dh, dq_b, dq_j)
        u = casadi.SX.sym("u", 0)
        for f in f_e:
            u = casadi.vertcat(u, f)
        u = casadi.vertcat(u, dq_j)

        x_next = self.state_integrate()(x, dx * self.dt)

        return casadi.Function("int_dyn", [x, u, dq_b], [x_next], ["x", "u", "dq_b"], ["x_next"])

    def get_base_velocity(self):
        h = casadi.SX.sym("h", 6)
        q = casadi.SX.sym("q", self.nq)
        dq_j = casadi.SX.sym("v", self.nj)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.computeCentroidalMap(self.model, self.data, q)

        A = self.data.Ag
        Ab = A[:, :6]
        Aj = A[:, 6:]
        Ab_inv = casadi.inv(Ab + 1e-8 * casadi.SX.eye(6))
        # Ab_inv = self.compute_Ab_inv(Ab)
        dq_b = Ab_inv @ (h - Aj @ dq_j)

        return casadi.Function("base_vel", [h, q, dq_j], [dq_b], ["h", "q", "dq_j"], ["dq_b"])

    def get_frame_position(self, frame_id):
        q = casadi.SX.sym("q", self.nq)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        pos = self.data.oMf[frame_id].translation

        return casadi.Function("frame_pos", [q], [pos], ["q"], ["pos"])

    def get_frame_velocity(self, frame_id):
        q = casadi.SX.sym("q", self.nq)
        dq = casadi.SX.sym("dq", self.nv)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q, dq)
        vel = cpin.getFrameVelocity(self.model, self.data, frame_id, pin.LOCAL_WORLD_ALIGNED)
        vel_lin = vel.vector[:3]

        return casadi.Function("frame_vel", [q, dq], [vel_lin], ["q", "dq"], ["vel_lin"])

    def get_centroidal_momentum(self):
        q = casadi.SX.sym("q", self.nq)
        dq = casadi.SX.sym("dq", self.nv)

        # TODO: Check Pinocchio terms
        cpin.computeAllTerms(self.model, self.data, q, dq)
        h = self.data.hg.vector

        return casadi.Function("centroidal_momentum", [q, dq], [h], ["q", "dq"], ["h"])

    def compute_Ab_inv(self, Ab):
        # NOTE: This is the OCS2 implementation
        mass = Ab[0, 0]
        Ab_22_inv = casadi.inv(Ab[3:, 3:])
        Ab_inv = casadi.SX.zeros(6, 6)
        Ab_inv[:3, :3] = 1 / mass * casadi.SX.eye(3)
        Ab_inv[:3, 3:] = -1 / mass * Ab[:3, 3:] @ Ab_22_inv
        Ab_inv[3:, :3] = casadi.SX.zeros(3, 3)
        Ab_inv[3:, 3:] = Ab_22_inv
        return Ab_inv
