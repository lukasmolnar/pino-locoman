from time import sleep

import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi

from load import load

URDF_PATH = "b2g_description/urdf/b2g.urdf"
SRDF_PATH = "b2g_description/srdf/b2g.srdf"
REFERENCE_POSE = "standing_with_arm_up"

N_JOINTS = 19  # for now including gripper
N_STATES = 13 + N_JOINTS  # 6 centroidal momentum + 7 floating base + joint positions
N_INPUTS = 12 + N_JOINTS  # 4x3 end-effector forces + joint velocities

DEBUG = True  # print info

# Problem parameters
x_goal = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0.6, 0, 0, 0, 1,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.5, -0.5, 0, 0, 0, 0
]
x0 = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0.55, 0, 0, 0, 1,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.26, -0.26, 0, 0, 0, 0
]
x_weights = [
    0, 0, 0, 0, 0, 0,
    100, 100, 100, 100, 100, 100,  # no quaternion
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1
]
Q = np.diag(x_weights)
nodes = 20
dt = 0.05

# Other variables
x_nom = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0.55, 0, 0, 0, 1,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.26, -0.26, 0, 0, 0, 0
]

assert len(x_goal) == N_STATES
assert len(x0) == N_STATES
assert len(x_nom) == N_STATES
assert len(x_weights) == N_STATES - 1  # no quaternion

def state_integrate(model):
    x = casadi.SX.sym("x", 6 + model.nq)
    dx = casadi.SX.sym("dx", 6 + model.nv)

    h = x[:6]
    q = x[6:]
    dh = dx[:6]
    dq = dx[6:]

    h_next = h + dh
    q_next = cpin.integrate(model, q, dq)
    x_next = casadi.vertcat(h_next, q_next)

    return casadi.Function("integrate", [x, dx], [x_next], ["x", "dx"], ["x_next"])


def state_difference(model):
    x0 = casadi.SX.sym("x0", 6 + model.nq)
    x1 = casadi.SX.sym("x1", 6 + model.nq)

    h0 = x0[:6]
    q0 = x0[6:]
    h1 = x1[:6]
    q1 = x1[6:]

    h_diff = h1 - h0
    q_diff = cpin.difference(model, q0, q1)
    x_diff = casadi.vertcat(h_diff, q_diff)

    return casadi.Function("difference", [x0, x1], [x_diff], ["x0", "x1"], ["x_diff"])


def compute_Ab_inv(Ab):
    # NOTE: This is the OCS2 implementation
    mass = Ab[0, 0]
    Ab_22_inv = casadi.inv(Ab[3:, 3:])
    Ab_inv = casadi.SX.zeros(6, 6)
    Ab_inv[:3, :3] = 1 / mass * casadi.SX.eye(3)
    Ab_inv[:3, 3:] = -1 / mass * Ab[:3, 3:] @ Ab_22_inv
    Ab_inv[3:, :3] = casadi.SX.zeros(3, 3)
    Ab_inv[3:, 3:] = Ab_22_inv
    return Ab_inv


def centroidal_dynamics(model, data, ee_frame_ids, dt):
    # States
    p_com = casadi.SX.sym("p_com", 3)  # COM linear momentum
    l_com = casadi.SX.sym("l_com", 3)  # COM angular momentum
    q = casadi.SX.sym("q", model.nq)  # generalized coords

    # Inputs
    f_e = [casadi.SX.sym(f"f_e_{i}", 3) for i in range(len(ee_frame_ids))]  # end-effector forces
    v = casadi.SX.sym("v", N_JOINTS)  # joint velocities

    # Pinocchio terms
    # NOTE: OCS2 uses only these
    cpin.forwardKinematics(model, data, q)
    cpin.computeCentroidalMap(model, data, q)
    cpin.updateFramePlacements(model, data)

    # COM Dynamics
    g = np.array([0, 0, -9.81 * data.mass[0]])
    dp_com = sum(f_e) + g
    dl_com = casadi.SX.zeros(3)
    for idx, frame_id in enumerate(ee_frame_ids):
        r_ee = data.oMf[frame_id].translation
        dl_com += casadi.cross(r_ee, f_e[idx])

    h = casadi.vertcat(p_com, l_com)
    dh = casadi.vertcat(dp_com, dl_com)

    # Joint Dynamics
    Ab = data.Ag[:, :6]
    Aj = data.Ag[:, 6:]
    Ab_inv = casadi.inv(Ab + 1e-8 * casadi.SX.eye(6))
    # Ab_inv = compute_Ab_inv(Ab)
    dq_b = Ab_inv @ (h - Aj @ v)
    dq_j = v

    x = casadi.vertcat(h, q)
    dx = casadi.vertcat(dh, dq_b, dq_j)
    u = casadi.SX.sym("u", 0)
    for f in f_e:
        u = casadi.vertcat(u, f)
    u = casadi.vertcat(u, v)

    x_next = state_integrate(model)(x, dx * dt)

    return casadi.Function("int_dyn", [x, u], [x_next], ["x", "u"], ["x_next"])


def get_frame_velocity(model, data, frame_id):
    q = casadi.SX.sym("q", model.nq)
    v = casadi.SX.sym("v", N_JOINTS)
    dq = casadi.vertcat([0] * 6, v)  # zero base velocity

    # TODO: Check what Pinocchio terms are necessary
    cpin.computeAllTerms(model, data, q, dq)
    vel = cpin.getFrameVelocity(model, data, frame_id)
    vel_lin = vel.vector[:3]

    return casadi.Function("frame_vel", [q, v], [vel_lin], ["q", "v"], ["vel_lin"])


def get_centroidal_momentum(model, data):
    # TODO: Possibly use this as a constraint?
    q = casadi.SX.sym("q", model.nq)
    v = casadi.SX.sym("v", N_JOINTS)
    dq = casadi.vertcat([0] * 6, v)  # zero base velocity

    # TODO: Check what Pinocchio terms are necessary
    cpin.computeAllTerms(model, data, q, dq)
    h = data.hg.vector

    return casadi.Function("centroidal_momentum", [q], [h], ["q"], ["h"])


class OptimalControlProblem:
    def __init__(self, model, ee_frame_ids):
        self.opti = casadi.Opti()
        self.model = model

        self.c_model = cpin.Model(self.model)
        self.c_data = self.c_model.createData()
        print(self.c_model)

        ndx = N_STATES - 1  # x includes quaternion, dx does not
        nu = N_INPUTS
        mu = 0.5  # friction coefficient

        self.c_dxs = self.opti.variable(ndx, nodes + 1)  # state trajectory
        self.c_us = self.opti.variable(nu, nodes)  # control trajectory

        # Objective function
        obj = 0

        # Interpolate between start and goal
        x_interpol = np.linspace(x0, x_goal, nodes)

        # State & Control regularization
        for i in range(nodes):
            x_i = state_integrate(self.c_model)(x_nom, self.c_dxs[:, i])
            e_reg = state_difference(self.c_model)(x_interpol[i], x_i)
            obj += (
                1e-5 * 0.5 * e_reg.T @ Q @ e_reg
                # + 1e-5 * 0.5 * self.c_us[:, i].T @ self.c_us[:, i]
            )

        self.opti.minimize(obj)

        for i in range(nodes):
            # Dynamical constraints
            x_i = state_integrate(self.c_model)(x_nom, self.c_dxs[:, i])
            x_i_1 = state_integrate(self.c_model)(x_nom, self.c_dxs[:, i + 1])
            f_x_u = centroidal_dynamics(self.c_model, self.c_data, ee_frame_ids, dt)(
                x_i, self.c_us[:, i]
            )
            gap = state_difference(self.c_model)(f_x_u, x_i_1)
            self.opti.subject_to(gap == [0] * ndx)

            q_i = x_i[6:]
            v_i = self.c_us[12:, i]

            # Contact constraints
            for j, frame_id in enumerate(ee_frame_ids):
                # Friction cone
                f_e_j = self.c_us[j * 3 : (j + 1) * 3, i]
                f_normal = f_e_j[2]
                f_tangent_square = f_e_j[0]**2 + f_e_j[1]**2

                self.opti.subject_to(f_normal >= 0)
                self.opti.subject_to(mu**2 * f_normal**2 >= f_tangent_square)

                # Zero end-effector velocity
                vel_lin = get_frame_velocity(self.c_model, self.c_data, frame_id)(q_i, v_i)
                self.opti.subject_to(vel_lin == [0] * 3)

        # State and input constraints
        self.opti.subject_to(self.opti.bounded(-1, self.c_dxs, 1))
        self.opti.subject_to(self.opti.bounded(-500, self.c_us[:12, :], 500))
        self.opti.subject_to(self.opti.bounded(-1, self.c_us[12:, :], 1))

        # Final constraint
        # x_N = state_integrate(self.c_model)(x_nom, self.c_dxs[:, nodes])
        # e_goal = state_difference(self.c_model)(x_N, x_goal)
        # self.opti.subject_to(e_goal == [0] * ndx)

        # Initial state
        x_0 = state_integrate(self.c_model)(x_nom, self.c_dxs[:, 0])
        self.opti.subject_to(state_difference(self.c_model)(x0, x_0) == [0] * ndx)

        # Warm start
        self.opti.set_initial(
            self.c_dxs, np.vstack([np.zeros(ndx) for _ in range(nodes + 1)]).T
        )
        self.opti.set_initial(
            self.c_us, np.vstack([np.zeros(nu) for _ in range(nodes)]).T
        )

    def solve(self, approx_hessian=True):
        opts = {"verbose": False}
        opts["ipopt"] = {
            "max_iter": 1000,
            "linear_solver": "mumps",
            "tol": 3.82e-6,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
        }

        if approx_hessian:
            opts["ipopt"]["hessian_approximation"] = "limited-memory"

        # Solver initialization
        self.opti.solver("ipopt", opts)  # set numerical backend

        try:
            self.sol = self.opti.solve()
        except:  # noqa: E722
            self.sol = self.opti.debug

        if self.sol.stats()["return_status"] == "Solve_Succeeded":
            self._retract_trajectory()
            self._compute_gaps()

    def _retract_trajectory(self):
        self.hs = []
        self.qs = []
        self.us = []
        self.gaps = []

        h_nom = np.array(x_nom)[:6]
        q_nom = np.array(x_nom)[6:]

        for idx, (dx_sol, u_sol) in enumerate(
            zip(self.sol.value(self.c_dxs).T, self.sol.value(self.c_us).T)
        ):
            dh = dx_sol[:6]
            dq = dx_sol[6:]
            h = h_nom + dh
            q = pin.integrate(self.model, q_nom, dq)
            self.hs.append(h)
            self.qs.append(q)
            self.us.append(u_sol)

        q = pin.integrate(
            self.model, np.array(x_nom)[6:], self.sol.value(self.c_dxs).T[nodes, 6:]
        )
        self.qs.append(q)

    def _compute_gaps(self):
        # TODO
        return
        # self.gaps = {"vector": [np.zeros(self.model.nv)], "norm": [0]}

        # nq = self.model.nq

        # for idx, (x, u) in enumerate(zip(self.xs, self.us)):
        #     x_pin = self._simulate_step(x, u)

        #     gap_q = pin.difference(self.model, x_pin[:nq], self.xs[idx + 1][:nq])
        #     gap_v = self.xs[idx + 1][nq:] - x_pin[nq:]

        #     gap = np.concatenate([gap_q, gap_v])
        #     self.gaps["vector"].append(gap)
        #     self.gaps["norm"].append(np.linalg.norm(gap))

    def _simulate_step(self, x, u):
        # TODO
        return
        # h = x[:6]
        # q = x[6:]
        # dx = np.concatenate((np.zeros(6), u * dt))
        # x_next = pin.integrate(self.model, x, dx)

        # return x_next


def main():
    robot = load(URDF_PATH, SRDF_PATH)
    model = robot.model
    data = robot.data

    robot.q0 = robot.model.referenceConfigurations[REFERENCE_POSE]
    pin.computeAllTerms(model, data, robot.q0, np.zeros(model.nv))

    ee_frame_ids = [model.getFrameId(f) for f in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]]

    oc_problem = OptimalControlProblem(model, ee_frame_ids)
    oc_problem.solve(approx_hessian=True)

    hs = np.vstack(oc_problem.hs)
    qs = np.vstack(oc_problem.qs)
    us = np.vstack(oc_problem.us)

    print("Initial state: ", qs[0])
    print("Final state: ", qs[-1])

    # Visualize
    robot.initViewer()
    robot.loadViewerModel("pinocchio")
    for _ in range(10):
        for k in range(nodes):
            q = qs[k]
            robot.display(q)
            if DEBUG:
                h = hs[k]
                u = us[k]
                pin.forwardKinematics(model, data, q)
                pin.computeCentroidalMomentum(model, data)
                pin.updateFramePlacements(model, data)
                h_true = data.hg.vector
                v_foot = pin.getFrameVelocity(model, data, ee_frame_ids[0]).vector[:3]
                print("k: ", k)
                print("h: ", h)
                print("h_true: ", h_true)
                print("q: ", q)
                print("u: ", u)
                print("v_foot: ", v_foot)

            sleep(dt)


if __name__ == "__main__":
    main()
