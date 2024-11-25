from time import sleep

import numpy as np
import pinocchio as pin

from helpers import *
from optimal_control_problem import OptimalControlProblem

URDF_PATH = "b2g_description/urdf/b2g.urdf"
SRDF_PATH = "b2g_description/srdf/b2g.srdf"
REFERENCE_POSE = "standing_with_arm_up"

DEBUG = False  # print info

# Problem parameters
nodes = 30
dt = 0.05
gait_sequence = GaitSequence(gait="trot", nodes=nodes, dt=dt)
com_goal = np.array([10, 0, 0])  # x, y, yaw

# Nominal state (from which dx is integrated)
x_nom = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0.55, 0, 0, 0, 1,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.7, -1.5, 0, 0.7, -1.5,
    0, 0.26, -0.26, 0, 0, 0, 0
]


def main():
    robot = load(URDF_PATH, SRDF_PATH)
    model = robot.model
    data = robot.data
    print(model)

    q0 = model.referenceConfigurations[REFERENCE_POSE]
    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    oc_problem = OptimalControlProblem(
        model=model,
        data=data,
        gait_sequence=gait_sequence,
        com_goal=com_goal,
        x_nom=x_nom,
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
            if DEBUG:
                h = hs[k]
                u = us[k]
                print("k: ", k)
                print("h: ", h)
                print("q: ", q)
                print("u: ", u)

            sleep(dt)


if __name__ == "__main__":
    main()
