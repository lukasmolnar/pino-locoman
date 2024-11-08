from time import sleep

import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi

from load import load

URDF_PATH = "b2g_description/urdf/b2g.urdf"
SRDF_PATH = "b2g_description/srdf/b2g.srdf"
REFERENCE_POSE = "standing_with_arm_up"

N_STATES = 32  # 6 centroidal momentum + 7 floating base + 12 leg joints + 7 arm joints
N_JOINTS = 19  # for now including gripper

# Problem parameters
x_goal = [
    0, 0, 0, 0, 0, 0,
    0.5, 0, 0.55, 0, 0, 0, 1,
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


def centroidal_dynamics(model, data, dt):
    h = casadi.SX.sym("h", 6)  # centroidal momentum
    q = casadi.SX.sym("q", model.nq)
    u = casadi.SX.sym("u", N_JOINTS)

    dh = data.hg.vector
    Ag = data.Ag
    Ag_b = Ag[:, :6]
    Ag_j = Ag[:, 6:]

    Ag_b_reg = Ag_b + 1e-6 * casadi.SX.eye(6)
    dq_b = casadi.inv(Ag_b_reg) @ (h - Ag_j @ u)
    dq_j = u

    x = casadi.vertcat(h, q)
    dx = casadi.vertcat(dh, dq_b, dq_j)

    x_next = state_integrate(model)(x, dx * dt)

    return casadi.Function("int_dyn", [x, u], [x_next], ["x", "u"], ["x_next"])


def cost_quadratic_state_error(model):
    dx = casadi.SX.sym("dx", model.nv * 2)

    x_N = state_integrate(model)(x_nom, dx)
    e_goal = state_difference(model)(x_N, x_goal)

    cost = 0.5 * e_goal.T @ e_goal

    return casadi.Function("quad_cost", [dx], [cost], ["dx"], ["cost"])


class OptimalControlProblem:
    def __init__(self, model, terminal_soft_constraint=False):
        self.opti = casadi.Opti()

        self.model = model
        self.data = self.model.createData()

        self.c_model = cpin.Model(self.model)
        self.c_data = self.c_model.createData()

        print(self.c_model)

        ndx = 6 + model.nv
        nu = N_JOINTS

        self.c_dxs = self.opti.variable(ndx, nodes + 1)  # state trajectory
        self.c_us = self.opti.variable(nu, nodes)  # control trajectory

        # Objective function
        obj = 0

        # State & Control regularization
        for i in range(nodes):
            x_i = state_integrate(self.c_model)(x_nom, self.c_dxs[:, i])
            e_reg = state_difference(self.c_model)(x_nom, x_i)
            obj += (
                1e-5 * 0.5 * e_reg.T @ e_reg
                + 1e-5 * 0.5 * self.c_us[:, i].T @ self.c_us[:, i]
            )
        if terminal_soft_constraint:
            obj += 1000 * cost_quadratic_state_error(self.c_model)(self.c_dxs[:, nodes])

        self.opti.minimize(obj)

        # Dynamical constraints
        for i in range(nodes):
            x_i = state_integrate(self.c_model)(x_nom, self.c_dxs[:, i])
            x_i_1 = state_integrate(self.c_model)(x_nom, self.c_dxs[:, i + 1])
            f_x_u = centroidal_dynamics(self.c_model, self.c_data, dt)(
                x_i, self.c_us[:, i]
            )
            gap = state_difference(self.c_model)(f_x_u, x_i_1)
            self.opti.subject_to(gap == [0] * ndx)

        # Control constraints
        self.opti.subject_to(self.opti.bounded(-10, self.c_us, 10))

        # Final constraint
        if not terminal_soft_constraint:
            x_N = state_integrate(self.c_model)(x_nom, self.c_dxs[:, nodes])
            e_goal = state_difference(self.c_model)(x_N, x_goal)
            self.opti.subject_to(e_goal == [0] * ndx)

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
        self.xs = []
        self.us = []
        self.gaps = []

        for idx, (dx_sol, u_sol) in enumerate(
            zip(self.sol.value(self.c_dxs).T, self.sol.value(self.c_us).T)
        ):
            q = pin.integrate(self.model, np.array(x_nom)[6:], dx_sol[6:])
            self.xs.append(q)
            self.us.append(u_sol)

        q = pin.integrate(
            self.model, np.array(x_nom)[6:], self.sol.value(self.c_dxs).T[nodes, 6:]
        )
        self.xs.append(q)

    def _compute_gaps(self):
        self.gaps = {"vector": [np.zeros(self.model.nv)], "norm": [0]}

        nq = self.model.nq

        for idx, (x, u) in enumerate(zip(self.xs, self.us)):
            x_pin = self._simulate_step(x, u)

            gap_q = pin.difference(self.model, x_pin[:nq], self.xs[idx + 1][:nq])
            gap_v = self.xs[idx + 1][nq:] - x_pin[nq:]

            gap = np.concatenate([gap_q, gap_v])
            self.gaps["vector"].append(gap)
            self.gaps["norm"].append(np.linalg.norm(gap))

    def _simulate_step(self, x, u):
        # TODO: dynamics
        h = x[:6]
        q = x[6:]
        dx = np.concatenate((np.zeros(6), u * dt))
        x_next = pin.integrate(self.model, x, dx)

        return x_next


def main():
    robot = load(URDF_PATH, SRDF_PATH)
    model = robot.model
    data = robot.data

    robot.q0 = robot.model.referenceConfigurations[REFERENCE_POSE]
    pin.forwardKinematics(model, data, robot.q0)

    oc_problem = OptimalControlProblem(model, terminal_soft_constraint=False)

    oc_problem.solve(approx_hessian=True)
    xs = np.vstack(oc_problem.xs)

    print("Initial state: ", xs[0])
    print("Final state: ", xs[-1])

    # Visualize
    robot.initViewer()
    robot.loadViewerModel("pinocchio")
    for _ in range(10):
        for k in range(nodes):
            robot.display(xs[k])
            sleep(dt)


if __name__ == "__main__":
    main()
