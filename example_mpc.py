from time import sleep

import numpy as np
import pinocchio as pin

from helpers import *
from optimal_control_problem import OptimalControlProblem

# Problem parameters
robot_class = B2(reference_pose="standing")
nodes = 20
dt = 0.02
robot_class.set_gait_sequence(gait="trot", nodes=nodes, dt=dt, arm_task=False)

# MPC parameters
mpc_nodes = 200

# Tracking goal: linear and angular momentum
com_goal = np.array([0.2, 0, 0, 0, 0, 0])

debug = False  # print info


def mpc_loop(ocp, q0, N):
    # Initialize solver
    ocp.init_solver(solver="fatrop", approx_hessian=True)
    x_init = np.concatenate((np.zeros(6), q0))
    ocp.update_initial_state(x_init)
    ocp.update_gait_sequence(shift_idx=0)
    ocp.solve(retract_all=False)

    solve_time = ocp.sol.stats()["t_wall_total"]

    for k in range(1, N):
        x_init = ocp.dyn.state_integrate()(x_init, ocp.dx_prev[1])
        ocp.update_initial_state(x_init)
        ocp.update_gait_sequence(shift_idx=k)
        ocp.warm_start(ocp.dx_prev, ocp.u_prev)
        ocp.solve(retract_all=False)

        solve_time += ocp.sol.stats()["t_wall_total"]

    print("Average solve time: ", solve_time / N)

    return ocp


def main():
    robot = robot_class.robot
    model = robot_class.model
    data = robot_class.data
    q0 = robot_class.q0
    print(model)
    print(q0)

    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    # Setup OCP
    ocp = OptimalControlProblem(
        robot_class=robot_class,
        com_goal=com_goal,
    )
    ocp = mpc_loop(ocp, q0, mpc_nodes)

    qs = ocp.qs
    hs = ocp.hs
    us = ocp.us

    print("Initial h: ", hs[0].T)
    print("Final h: ", hs[-1].T)
    print("Initial q: ", qs[0].T)
    print("Final q: ", qs[-1].T)

    # Visualize
    robot.initViewer()
    robot.loadViewerModel("pinocchio")
    for _ in range(50):
        for k in range(len(qs)):
            q = qs[k]
            robot.display(q)
            if debug:
                h = hs[k]
                u = us[k]
                pin.computeAllTerms(model, data, q, np.zeros(model.nv))
                print("k: ", k)
                print("h: ", h)
                print("q: ", q)
                print("u: ", u)
                print("com: ", data.com[0])

            sleep(dt)


if __name__ == "__main__":
    main()
