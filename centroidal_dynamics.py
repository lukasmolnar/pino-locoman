import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi


class CentroidalDynamics:
    def __init__(
            self,
            model,
            mass,
            ee_ids,
            dx_opt_indices,
        ):
        self.model = cpin.Model(model)
        self.data = self.model.createData()
        self.mass = mass
        self.ee_ids = ee_ids
        self.dx_opt_indices = dx_opt_indices

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nj = self.nq - 7  # without base position and quaternion

        self.quat_nom = casadi.SX([0, 0, 0, 1])
        self.SO3 = cpin.liegroups.SO3()

    def state_integrate(self, use_quat_nom=False):
        x = casadi.SX.sym("x", 6 + self.nq)
        dx_opt = casadi.SX.sym("dx_opt", len(self.dx_opt_indices))

        dx = casadi.SX.zeros(6 + self.nv)  # The other DXs remain zero
        dx[self.dx_opt_indices] = dx_opt

        h = x[:6]
        q = x[6:]
        dh = dx[:6]
        dq = dx[6:]

        if use_quat_nom:
            qb_pos = q[:3]
            dqb_pos = dq[:3]
            qb_quat = q[3:7]
            dqb_quat = dq[3:6]
            qj = q[7:]
            dqj = dq[6:]

            qb_next_pos = qb_pos + dqb_pos
            qb_next_quat = self.SO3.integrate(self.quat_nom, dqb_quat)
            qj_next = qj + dqj
            q_next = casadi.vertcat(qb_next_pos, qb_next_quat, qj_next)

        else:
            q_next = cpin.integrate(self.model, q, dq)

        h_next = h + dh
        x_next = casadi.vertcat(h_next, q_next)

        return casadi.Function("integrate", [x, dx_opt], [x_next], ["x", "dx_opt"], ["x_next"])
    
    def state_difference(self):
        x0 = casadi.SX.sym("x0", 6 + self.nq)
        x1 = casadi.SX.sym("x1", 6 + self.nq)

        h0 = x0[:6]
        q0 = x0[6:]
        h1 = x1[:6]
        q1 = x1[6:]

        dh = h1 - h0
        dq = cpin.difference(self.model, q0, q1)
        dx = casadi.vertcat(dh, dq)

        return casadi.Function("difference", [x0, x1], [dx], ["x0", "x1"], ["dx"])

    def centroidal_dynamics(self, arm_ee_id=None):
        # States
        p_com = casadi.SX.sym("p_com", 3)  # COM linear momentum
        l_com = casadi.SX.sym("l_com", 3)  # COM angular momentum
        q = casadi.SX.sym("q", self.nq)  # generalized coordinates (base + joints)

        # Inputs
        nf = len(self.ee_ids)
        if arm_ee_id:
            nf += 1
        f_e = [casadi.SX.sym(f"f_e_{i}", 3) for i in range(nf)]  # end-effector forces
        dq_j = casadi.SX.sym("dq_j", self.nj)  # joint velocities

        # Base velocity
        dq_b = casadi.SX.sym("dq_b", 6)
        dq = casadi.vertcat(dq_b, dq_j)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.centerOfMass(self.model, self.data)
        cpin.updateFramePlacements(self.model, self.data)

        # COM Dynamics
        g = np.array([0, 0, -9.81 * self.mass])
        dp_com = sum(f_e) + g
        dl_com = casadi.SX.zeros(3)
        for idx, frame_id in enumerate(self.ee_ids):
            r_ee = self.data.oMf[frame_id].translation - self.data.com[0]
            dl_com += casadi.cross(r_ee, f_e[idx])
        if arm_ee_id:
            r_ee = self.data.oMf[arm_ee_id].translation - self.data.com[0]
            dl_com += casadi.cross(r_ee, f_e[-1])

        h = casadi.vertcat(p_com, l_com)
        dh = casadi.vertcat(dp_com, dl_com) / self.mass # scale by mass

        # Stack states and inputs
        x = casadi.vertcat(h, q)
        u = casadi.SX.sym("u", 0)
        for f in f_e:
            u = casadi.vertcat(u, f)
        u = casadi.vertcat(u, dq_j)

        # Return dynamics: dx = f(x, u)
        dx = casadi.vertcat(dh, dq)

        return casadi.Function("int_dyn", [x, u, dq_b], [dx], ["x", "u", "dq_b"], ["dx"])

    def get_base_velocity(self):
        h = casadi.SX.sym("h", 6)
        q = casadi.SX.sym("q", self.nq)
        dq_j = casadi.SX.sym("dq_j", self.nj)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.computeCentroidalMap(self.model, self.data, q)

        A = self.data.Ag
        Ab = A[:, :6]
        Aj = A[:, 6:]
        # Ab_inv = casadi.inv(Ab + 1e-6 * casadi.SX.eye(6))
        Ab_inv = self.compute_Ab_inv(Ab)
        dq_b = Ab_inv @ (h * self.mass - Aj @ dq_j)  # scale by mass

        return casadi.Function("base_vel", [h, q, dq_j], [dq_b], ["h", "q", "dq_j"], ["dq_b"])

    def get_frame_position(self, frame_id):
        q = casadi.SX.sym("q", self.nq)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q)
        cpin.updateFramePlacement(self.model, self.data, frame_id)
        pos = self.data.oMf[frame_id].translation

        return casadi.Function("frame_pos", [q], [pos], ["q"], ["pos"])

    def get_frame_velocity(self, frame_id, ref=pin.LOCAL_WORLD_ALIGNED):
        q = casadi.SX.sym("q", self.nq)
        dq = casadi.SX.sym("dq", self.nv)

        # TODO: Check Pinocchio terms
        cpin.forwardKinematics(self.model, self.data, q, dq)
        vel = cpin.getFrameVelocity(self.model, self.data, frame_id, ref).vector

        return casadi.Function("frame_vel", [q, dq], [vel], ["q", "dq"], ["vel"])

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
