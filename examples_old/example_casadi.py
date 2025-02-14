from time import sleep

import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca

from load import load

URDF_PATH = "b2g_description/urdf/b2g.urdf"
SRDF_PATH = "b2g_description/srdf/b2g.srdf"
REFERENCE_POSE = "standing_with_arm_up"

N_STATES = 25  # 6 floating base + 12 leg joints + 7 arm joints
N_JOINTS = 19  # for now including gripper

# Problem parameters
x_goal = [0, 0, 0.55, 0, 0, 0, 1,
         0, 0.7, -1.5, 0, 0.7, -1.5,
         0, 0.7, -1.5, 0, 0.7, -1.5,
         0, 0.5, -0.5, 0, 0, 0, 0]
x0 = [0, 0, 0.55, 0, 0, 0, 1,
         0, 0.7, -1.5, 0, 0.7, -1.5,
         0, 0.7, -1.5, 0, 0.7, -1.5,
         0, 0.26, -0.26, 0, 0, 0, 0]

nodes = 80
dt = 0.02

# Other variables
x_nom = [0, 0, 0.55, 0, 0, 0, 1,
         0, 0.7, -1.5, 0, 0.7, -1.5,
         0, 0.7, -1.5, 0, 0.7, -1.5,
         0, 0.26, -0.26, 0, 0, 0, 0]


def state_integrate(model):
    x = ca.SX.sym("x", model.nq)
    dx = ca.SX.sym("dx", model.nv)
    x_next = cpin.integrate(model, x, dx)

    return ca.Function("integrate", [x, dx], [x_next], ["x", "dx"], ["x_next"])


def state_difference(model):
    x0 = ca.SX.sym("x0", model.nq)
    x1 = ca.SX.sym("x1", model.nq)
    x_diff = cpin.difference(model, x0, x1)

    return ca.Function("difference", [x0, x1], [x_diff], ["x0", "x1"], ["x_diff"])


def euler_integration(model, data, dt):
    x = ca.SX.sym("x", model.nq)
    u = ca.SX.sym("u", 19)
    dx = ca.vertcat(np.zeros(6), u * dt)

    # tau = ca.SX.zeros(model.nv)
    # a = cpin.aba(model, data, x, u, tau)
    # dq = u * dt + a * dt**2
    # dx = dq

    x_next = state_integrate(model)(x, dx)

    return ca.Function("int_dyn", [x, u], [x_next], ["x", "u"], ["x_next"])


def cost_quadratic_state_error(model):
    dx = ca.SX.sym("dx", model.nv * 2)

    x_N = state_integrate(model)(x_nom, dx)
    e_goal = state_difference(model)(x_N, x_goal)

    cost = 0.5 * e_goal.T @ e_goal

    return ca.Function("quad_cost", [dx], [cost], ["dx"], ["cost"])


class OptimalControlProblem:
    def __init__(self, model, terminal_soft_constraint=False):
        self.opti = ca.Opti()

        self.model = model
        self.data = self.model.createData()

        self.c_model = cpin.Model(self.model)
        self.c_data = self.c_model.createData()

        print(self.c_model)

        nv = self.c_model.nv
        nu = N_JOINTS

        self.c_dxs = self.opti.variable(nv, nodes + 1)  # state trajectory
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
            f_x_u = euler_integration(self.c_model, self.c_data, dt)(
                x_i, self.c_us[:, i]
            )
            gap = state_difference(self.c_model)(f_x_u, x_i_1)

            self.opti.subject_to(gap == [0] * N_STATES)

        # Control constraints
        self.opti.subject_to(self.opti.bounded(-10, self.c_us, 10))

        # Final constraint
        if not terminal_soft_constraint:
            x_N = state_integrate(self.c_model)(x_nom, self.c_dxs[:, nodes])
            e_goal = state_difference(self.c_model)(x_N, x_goal)
            self.opti.subject_to(e_goal == [0] * N_STATES)

        # Initial state
        x_0 = state_integrate(self.c_model)(x_nom, self.c_dxs[:, 0])
        self.opti.subject_to(state_difference(self.c_model)(x0, x_0) == [0] * N_STATES)

        # Warm start
        self.opti.set_initial(
            self.c_dxs, np.vstack([np.zeros(nv) for _ in range(nodes + 1)]).T
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

        nq = self.model.nq
        nv = self.model.nv

        for idx, (dx_sol, u_sol) in enumerate(
            zip(self.sol.value(self.c_dxs).T, self.sol.value(self.c_us).T)
        ):
            q = pin.integrate(self.model, np.array(x_nom)[:nq], dx_sol[:nv])
            v = dx_sol[nv:]

            self.xs.append(np.concatenate([q, v]))
            self.us.append(u_sol)

        q = pin.integrate(
            self.model, np.array(x_nom)[:nq], self.sol.value(self.c_dxs).T[nodes, :nv]
        )
        v = self.sol.value(self.c_dxs).T[nodes, nv:]
        self.xs.append(np.concatenate([q, v]))

    def _compute_gaps(self):
        self.gaps = {"vector": [np.zeros(self.model.nv * 2)], "norm": [0]}

        nq = self.model.nq
        _nv = self.model.nv

        for idx, (x, u) in enumerate(zip(self.xs, self.us)):
            x_pin = self._simulate_step(x, u)

            gap_q = pin.difference(self.model, x_pin[:nq], self.xs[idx + 1][:nq])
            gap_v = self.xs[idx + 1][nq:] - x_pin[nq:]

            gap = np.concatenate([gap_q, gap_v])
            self.gaps["vector"].append(gap)
            self.gaps["norm"].append(np.linalg.norm(gap))

    def _simulate_step(self, x, u):
        dx = np.concatenate((np.zeros(6), u * dt))
        x_next = pin.integrate(self.model, x, dx)

        # a = pin.aba(self.model, self.data, q, v, tau)
        # dq = v * dt + a * dt**2
        # dv = a * dt
        # q_next = pin.integrate(self.model, q, dq)
        # v_next = v + dv

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
