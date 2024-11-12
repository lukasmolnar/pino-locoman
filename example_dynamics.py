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

# Problem parameters
x_goal = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0.55, 0, 0, 0, 1,
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
nodes = 80
dt = 0.02

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


def centroidal_dynamics(model, data, Ag, mass, dt, ne=4):
    # States
    p_com = casadi.SX.sym("p_com", 3)  # COM linear momentum
    l_com = casadi.SX.sym("l_com", 3)  # COM angular momentum
    q = casadi.SX.sym("q", model.nq)  # generalized coords

    # Inputs
    f_e = [casadi.SX.sym(f"f_e_{i}", 3) for i in range(ne)]  # end-effector forces
    v = casadi.SX.sym("v", N_JOINTS)  # joint velocities

    # COM Dynamics
    g = np.array([0, 0, -9.81 * mass])
    dp_com = sum(f_e) + g
    dl_com = casadi.SX.zeros(3)
    # TODO: dl_com depending on step locations

    h = casadi.vertcat(p_com, l_com)
    dh = casadi.vertcat(dp_com, dl_com)

    # Joint Dynamics
    Ag_b = Ag[:, :6]
    Ag_j = Ag[:, 6:]
    Ag_b_reg = Ag_b + 1e-6 * casadi.SX.eye(6)
    dq_b = casadi.inv(Ag_b_reg) @ (h - Ag_j @ v)
    dq_j = v

    x = casadi.vertcat(h, q)
    dx = casadi.vertcat(dh, dq_b, dq_j)
    u = casadi.SX.sym("u", 0)
    for f in f_e:
        u = casadi.vertcat(u, f)
    u = casadi.vertcat(u, v)

    x_next = state_integrate(model)(x, dx * dt)

    return casadi.Function("int_dyn", [x, u], [x_next], ["x", "u"], ["x_next"])


class OptimalControlProblem:
    def __init__(
            self,
            model,
            Ag,
            mass,
            ee_frame_ids,
        ):
        self.opti = casadi.Opti()
        self.model = model

        self.c_model = cpin.Model(model)
        self.c_data = self.c_model.createData()
        print(self.c_model)

        ndx = N_STATES - 1  # x includes quaternion, dx does not
        nu = N_INPUTS
        ne = len(ee_frame_ids)
        mu = 0.5  # friction coefficient

        self.c_dxs = casadi.SX.sym("dx", ndx, nodes + 1)
        self.c_us = casadi.SX.sym("u", nu, nodes)

        constraints = []
        g_lb = []
        g_ub = []

        # Objective function
        obj = 0

        # Interpolate between start and goal
        x_interpol = np.linspace(x0, x_goal, nodes)

        # State & Control regularization
        for i in range(nodes):
            x_i = state_integrate(self.c_model)(x_nom, self.c_dxs[:, i])
            e_reg = state_difference(self.c_model)(x_interpol[i], x_i)
            obj += (
                1e-5 * 0.5 * e_reg.T @ e_reg
                # + 1e-5 * 0.5 * self.c_us[:, i].T @ self.c_us[:, i]
            )

        # Dynamical constraints
        for i in range(nodes):
            x_i = state_integrate(self.c_model)(x_nom, self.c_dxs[:, i])
            x_i_1 = state_integrate(self.c_model)(x_nom, self.c_dxs[:, i + 1])
            f_x_u = centroidal_dynamics(self.c_model, self.c_data, Ag, mass, dt, ne)(
                x_i, self.c_us[:, i]
            )
            gap = state_difference(self.c_model)(f_x_u, x_i_1)
            constraints.append(gap)
            for _ in range(ndx):
                g_lb.append(0)
                g_ub.append(0)

        # Contact constraints
        for i in range(nodes):
            for j, frame_id in enumerate(ee_frame_ids):
                # Friction cone
                f_e_j = self.c_us[j * 3 : (j + 1) * 3, i]
                f_normal = f_e_j[2]
                f_tangent_square = f_e_j[0]**2 + f_e_j[1]**2

                constraints.append(f_normal)
                g_lb.append(0)
                g_ub.append(casadi.inf)
                constraints.append(mu**2 * f_normal**2 - f_tangent_square)
                g_lb.append(0)
                g_ub.append(casadi.inf)

                # Zero end-effector velocity
                x_i = state_integrate(self.c_model)(x_nom, self.c_dxs[:, i])
                q_i = x_i[6:]
                v_i = self.c_us[12:, i]
                dq_i = casadi.vertcat([0] * 6, v_i)  # zero base velocity

                cpin.computeAllTerms(self.c_model, self.c_data, q_i, dq_i)
                vel = cpin.getFrameVelocity(self.c_model, self.c_data, frame_id)
                vel_lin = vel.vector[:3]
                constraints.append(vel_lin)
                for _ in range(3):
                    g_lb.append(0)
                    g_ub.append(0)

        # Control constraints
        # TODO

        # Final constraint
        x_N = state_integrate(self.c_model)(x_nom, self.c_dxs[:, nodes])
        e_goal = state_difference(self.c_model)(x_N, x_goal)
        constraints.append(e_goal)
        for _ in range(ndx):
            g_lb.append(0)
            g_ub.append(0)

        # Initial state
        x_0 = state_integrate(self.c_model)(x_nom, self.c_dxs[:, 0])
        constraints.append(state_difference(self.c_model)(x0, x_0))
        for _ in range(ndx):
            g_lb.append(0)
            g_ub.append(0)

        # Warm start
        dxs_init = np.vstack([np.zeros(ndx) for _ in range(nodes + 1)])
        us_init = np.vstack([np.zeros(nu) for _ in range(nodes)])
        self.x_init = np.vstack([dxs_init.reshape(-1, 1), us_init.reshape(-1, 1)])

        # Initialize NLP
        self.nlp = {
            "x": casadi.vertcat(casadi.reshape(self.c_dxs, -1, 1), casadi.reshape(self.c_us, -1, 1)),
            "f": obj,
            "g": casadi.vertcat(*constraints)
        }
        self.g_lb = np.array(g_lb)
        self.g_ub = np.array(g_ub)

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

        solver = casadi.nlpsol("solver", "ipopt", self.nlp, opts)

        try:
            self.sol = solver(x0=self.x_init, lbg=self.g_lb, ubg=self.g_ub)
        except:
            print("Solver failed!")
            self.sol = None

        # TODO: check solution
        self._retract_trajectory()

    def _retract_trajectory(self):
        self.hs = []
        self.qs = []
        self.us = []

        x_sol = self.sol["x"]
        x_size = N_STATES - 1 + N_INPUTS
        q_nom = np.array(x_nom)[6:]

        for i in range(nodes):
            x = np.array(x_sol[i * x_size : (i + 1) * x_size])
            self.hs.append(x[:6])

            dq = x[6:N_STATES-1]
            q = pin.integrate(self.model, q_nom, dq)

            self.qs.append(q)
            self.us.append(x[N_STATES-1:])

        x_last = np.array(x_sol[nodes * x_size :])
        assert(len(x_last) == N_STATES-1)
        self.hs.append(x_last[:6])

        dq = x_last[6:]
        q = pin.integrate(self.model, q_nom, dq)
        self.qs.append(q)


def main():
    robot = load(URDF_PATH, SRDF_PATH)
    model = robot.model
    data = robot.data

    robot.q0 = robot.model.referenceConfigurations[REFERENCE_POSE]
    pin.computeAllTerms(model, data, robot.q0, np.zeros(model.nv))
    Ag = data.Ag
    mass = data.mass[0]

    ee_frame_ids = [model.getFrameId(f) for f in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]]

    oc_problem = OptimalControlProblem(
        model,
        Ag,
        mass,
        ee_frame_ids,
    )
    oc_problem.solve(approx_hessian=True)

    qs = oc_problem.qs
    us = oc_problem.us

    print("Initial q: ", qs[0])
    print("Final q: ", qs[-1])

    # Visualize
    robot.initViewer()
    robot.loadViewerModel("pinocchio")
    for _ in range(20):
        for k in range(nodes):
            q = np.array(qs[k])
            v = np.array(us[k])[12:].flatten()
            dq = np.concatenate((np.zeros(6), v))  # zero base velocity
            pin.computeAllTerms(model, data, q, dq)
            v_foot = pin.getFrameVelocity(model, data, ee_frame_ids[0])
            # print("Foot vel: ", v_foot)

            robot.display(q)
            sleep(dt)


if __name__ == "__main__":
    main()
