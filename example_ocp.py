from time import sleep

import numpy as np
import pinocchio as pin

from helpers import *
from optimal_control_problem import OptimalControlProblem

# Problem parameters
robot = B2(reference_pose="standing")
gait_type = "trot"
gait_nodes = 20
ocp_nodes = 15
dt = 0.02

arm_task = False
arm_f_des = np.array([0, 0, -100])
arm_vel_des = np.array([0.2, 0, 0])

# Tracking goal: linear and angular momentum
com_goal = np.array([0.2, 0, 0, 0, 0, 0])

debug = False  # print info


def main():
    robot.set_gait_sequence(gait_type, gait_nodes, dt)
    if type(robot) == B2G and arm_task:
        robot.add_arm_task(arm_f_des, arm_vel_des)
    robot.initialize_weights()

    robot_instance = robot.robot
    model = robot.model
    data = robot.data
    q0 = robot.q0
    print(model)
    print(q0)

    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    # Setup OCP
    ocp = OptimalControlProblem(
        robot=robot,
        nodes=ocp_nodes,
        com_goal=com_goal,
    )
    ocp.init_solver(solver="fatrop", approx_hessian=True)
    x_nom = np.concatenate((np.zeros(6), q0))
    ocp.update_initial_state(x_nom)
    ocp.update_gait_sequence(shift_idx=0)
    ocp.solve(retract_all=True)

    print("Final h: ", ocp.hs[-1].T)
    print("Final q: ", ocp.qs[-1].T)

    # Visualize
    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    for _ in range(50):
        for k in range(ocp_nodes):
            q = ocp.qs[k]
            robot_instance.display(q)
            if debug:
                h = ocp.hs[k]
                u = ocp.us[k]
                pin.computeAllTerms(model, data, q, np.zeros(model.nv))
                print("k: ", k)
                print("h: ", h)
                print("q: ", q)
                print("u: ", u)
                print("com: ", data.com[0])

            sleep(dt)


if __name__ == "__main__":
    main()
