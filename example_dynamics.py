from time import sleep

import numpy as np
import pinocchio as pin

from helpers import *
from optimal_control_problem import OptimalControlProblem

# Problem parameters
robot_class = B2G(reference_pose="standing_with_arm_forward")
nodes = 20
dt = 0.02
robot_class.set_gait_sequence(gait="trot", nodes=nodes, dt=dt, arm_task=True)

# Tracking goal: linear and angular momentum
com_goal = np.array([0, 0, 0, 0, 0, 0])

debug = False  # print info


def main():
    robot = robot_class.robot
    model = robot_class.model
    data = robot_class.data
    q0 = robot_class.q0
    print(model)
    print(q0)

    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    oc_problem = OptimalControlProblem(
        robot_class=robot_class,
        com_goal=com_goal,
    )
    oc_problem.solve(solver="fatrop", approx_hessian=True)

    hs = oc_problem.hs
    qs = oc_problem.qs
    us = oc_problem.us

    print("Initial h: ", hs[0].T)
    print("Final h: ", hs[-1].T)
    print("Initial q: ", qs[0].T)
    print("Final q: ", qs[-1].T)

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
