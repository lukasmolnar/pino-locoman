from time import sleep

import numpy as np
import pinocchio as pin
import casadi

from helpers import *
from centroidal_dynamics import CentroidalDynamics

URDF_PATH = "b2g_description/urdf/b2g.urdf"
SRDF_PATH = "b2g_description/srdf/b2g.srdf"
REFERENCE_POSE = "standing_with_arm_up"

N_JOINTS = 19  # for now including gripper
N_STATES = 13 + N_JOINTS  # 6 centroidal momentum + 7 floating base + joint positions
N_INPUTS = 12 + N_JOINTS  # 4x3 end-effector forces + joint velocities

DEBUG = False  # print info

# Problem parameters
x_goal = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0.55, 0, 0, 0, 1,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.5, -0.5, 0, 0, 0, 0
]
x_init = [
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

contact_feet = ["FR_foot", "RL_foot"]
swing_feet = ["FL_foot", "RR_foot"]

# Nominal state (from which dx is integrated)
x_nom = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0.55, 0, 0, 0, 1,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.26, -0.26, 0, 0, 0, 0
]

assert len(x_goal) == N_STATES
assert len(x_init) == N_STATES
assert len(x_nom) == N_STATES
assert len(x_weights) == N_STATES - 1  # no quaternion


class OptimalControlProblem:
    def __init__(self, model, data, contact_ee_ids, swing_ee_ids):
        self.opti = casadi.Opti()
        self.model = model
        self.data = data

        swing_ee_positions = {}
        for ee_id in swing_ee_ids:
            r_ee = self.data.oMf[ee_id].translation
            swing_ee_positions[ee_id] = r_ee

        self.dyn = CentroidalDynamics(model, contact_ee_ids)

        ndx = N_STATES - 1  # x includes quaternion, dx does not
        nf = 3 * len(contact_ee_ids)  # number of contact forces variables
        nu = nf + N_JOINTS  # total input variables
        mu = 0.5  # friction coefficient

        self.c_dxs = self.opti.variable(ndx, nodes + 1)  # state trajectory
        self.c_us = self.opti.variable(nu, nodes)  # control trajectory
        self.c_dq_b = self.opti.variable(6, nodes + 1)  # base velocity

        # Objective function
        obj = 0

        # Interpolate between start and goal
        x_interpol = np.linspace(x_init, x_goal, nodes + 1)

        # State & Control regularization
        for i in range(nodes + 1):
            x_i = self.dyn.state_integrate()(x_nom, self.c_dxs[:, i])
            e_reg = self.dyn.state_difference()(x_interpol[i], x_i)
            dq_b_i = self.c_dq_b[:, i]
            obj += (
                1e-1 * 0.5 * e_reg.T @ Q @ e_reg
                # + 1e-5 * 0.5 * dq_b_i.T @ dq_b_i
            )

        self.opti.minimize(obj)

        for i in range(nodes):
            # Dynamical constraints
            x = self.dyn.state_integrate()(x_nom, self.c_dxs[:, i])
            x_next = self.dyn.state_integrate()(x_nom, self.c_dxs[:, i + 1])
            x_next_dyn = self.dyn.centroidal_dynamics()(
                x, self.c_us[:, i], self.c_dq_b[:, i]
            )
            gap = self.dyn.state_difference()(x_next, x_next_dyn)
            self.opti.subject_to(gap == [0] * ndx)

            # Base velocity
            h = x[:6]
            q = x[6:]
            dq_j = self.c_us[nf:, i]
            dq_b = self.dyn.get_base_velocity()(h, q, dq_j)
            dq = casadi.vertcat(dq_b, dq_j)
            self.opti.subject_to(self.c_dq_b[:, i] == dq_b)

            # Contact constraints
            for idx, frame_id in enumerate(contact_ee_ids):
                # Friction cone
                f_e = self.c_us[idx * 3 : (idx + 1) * 3, i]
                f_normal = f_e[2]
                f_tangent_square = f_e[0]**2 + f_e[1]**2

                self.opti.subject_to(f_normal >= 0)
                self.opti.subject_to(mu**2 * f_normal**2 >= f_tangent_square)

                # Zero end-effector velocity
                vel_lin = self.dyn.get_frame_velocity(frame_id)(q, dq)
                self.opti.subject_to(vel_lin == [0] * 3)

            # Swing foot trajectory
            for frame_id, ee_pos in swing_ee_positions.items():
                vel = self.dyn.get_frame_velocity(frame_id)(q, dq)
                vel_des = swing_bezier_vel(ee_pos, dt * i, dt * nodes)
                self.opti.subject_to(vel == vel_des)

        # State and input constraints
        self.opti.subject_to(self.opti.bounded(-1, self.c_dxs, 1))
        self.opti.subject_to(self.opti.bounded(-500, self.c_us[:nf, :], 500))
        self.opti.subject_to(self.opti.bounded(-1, self.c_us[nf:, :], 1))

        # Final constraint
        # x_N = state_integrate()(x_nom, self.c_dxs[:, nodes])
        # e_goal = state_difference()(x_N, x_goal)
        # self.opti.subject_to(e_goal == [0] * ndx)

        # Initial state
        x_0 = self.dyn.state_integrate()(x_nom, self.c_dxs[:, 0])
        self.opti.subject_to(self.dyn.state_difference()(x_0, x_init) == [0] * ndx)

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
        self.dq_bs = [] 
        self.gaps = []

        h_nom = np.array(x_nom)[:6]
        q_nom = np.array(x_nom)[6:]

        for idx, (dx_sol, u_sol, dq_b_sol) in enumerate(
            zip(self.sol.value(self.c_dxs).T, self.sol.value(self.c_us).T, self.sol.value(self.c_dq_b).T)
        ):
            dh = dx_sol[:6]
            dq = dx_sol[6:]
            h = h_nom + dh
            q = pin.integrate(self.model, q_nom, dq)
            self.hs.append(h)
            self.qs.append(q)
            self.us.append(u_sol)
            self.dq_bs.append(dq_b_sol)

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
    q0 = model.referenceConfigurations[REFERENCE_POSE]
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    contact_ee_ids = [model.getFrameId(f) for f in contact_feet]
    swing_ee_ids = [model.getFrameId(f) for f in swing_feet]

    oc_problem = OptimalControlProblem(model, data, contact_ee_ids, swing_ee_ids)
    oc_problem.solve(approx_hessian=True)

    hs = np.vstack(oc_problem.hs)
    qs = np.vstack(oc_problem.qs)
    us = np.vstack(oc_problem.us)
    dq_bs = np.vstack(oc_problem.dq_bs)

    print("Initial state: ", qs[0])
    print("Final state: ", qs[-1])

    # Visualize
    robot.initViewer()
    robot.loadViewerModel("pinocchio")
    for _ in range(50):
        for k in range(nodes):
            q = qs[k]
            robot.display(q)
            if DEBUG:
                h = hs[k]
                u = us[k]
                dq_b = dq_bs[k]
                dq = np.concatenate([dq_b, u[12:]]) 
                pin.forwardKinematics(model, data, q, dq)
                pin.computeCentroidalMomentum(model, data)
                pin.updateFramePlacements(model, data)
                v_foot = pin.getFrameVelocity(model, data, contact_ee_ids[0]).vector[:3]
                r_ee = data.oMf[contact_ee_ids[0]].translation
                print("k: ", k)
                print("r_ee: ", r_ee)
                print("com_true: ", data.com[0])
                print("h: ", h)
                print("h_true: ", data.hg.vector)
                print("q: ", q)
                print("u: ", u)
                print("v_foot: ", v_foot)

            sleep(dt)


if __name__ == "__main__":
    main()
