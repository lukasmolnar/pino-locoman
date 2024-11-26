from time import sleep

import numpy as np
import pinocchio as pin

from helpers import *
from optimal_control_problem import OptimalControlProblem

# Problem parameters
robot_class = B2()
nodes = 30
dt = 0.05
robot_class.set_gait_sequence(gait="trot", nodes=nodes, dt=dt)

# Tracking goal: x, y, yaw
com_goal = np.array([0, 0, 0])

debug = True  # print info


def main():
    robot = robot_class.robot
    model = robot_class.model
    data = robot_class.data
    q0 = robot_class.q0
    print(model)

    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    oc_problem = OptimalControlProblem(
        robot_class=robot_class,
        com_goal=com_goal,
    )
    oc_problem.solve(approx_hessian=True)

    hs = np.vstack(oc_problem.hs)
    qs = np.vstack(oc_problem.qs)
    us = np.vstack(oc_problem.us)

    print("Initial h: ", hs[0])
    print("Final h: ", hs[-1])
    print("Initial q: ", qs[0])
    print("Final q: ", qs[-1])

    # Visualize
    robot.initViewer()
    robot.loadViewerModel("pinocchio")
    for _ in range(50):
        for k in range(nodes):
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
