import time
import numpy as np
import pinocchio as pin
import casadi

from helpers import *
from optimal_control_problem import OptimalControlProblem

# Problem parameters
# robot = B2(reference_pose="standing")
robot = B2G(reference_pose="standing_with_arm_forward", ignore_arm=False)
gait_type = "trot"
gait_nodes = 14
ocp_nodes = 10
dt = 0.03

# Only for B2G
arm_f_des = np.array([0, 0, -100])
arm_vel_des = np.array([0, 0, 0])

# Tracking goal: linear and angular momentum
com_goal = np.array([0, 0, 0, 0, 0, 0])

# Compiled solver
compile_solver = False
load_compiled_solver = None # "libcompiled_solver_B2G_N10_dt03.so"

debug = False  # print info


def main():
    robot.set_gait_sequence(gait_type, gait_nodes, dt)
    if type(robot) == B2G and not robot.ignore_arm:
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
    x_init = np.concatenate((np.zeros(6), q0))
    gait_idx = 0

    if load_compiled_solver:
        compiled_solver = casadi.external("compiled_solver", "codegen/lib/" + load_compiled_solver)
        contact_schedule = ocp.gait_sequence.shift_contact_schedule(gait_idx)[:,:ocp_nodes]
        start_time = time.time()
        DX_sol, U_sol = compiled_solver(x_init, contact_schedule, gait_idx)
        end_time = time.time()
        print("DX_sol: ", DX_sol)
        print("U_sol: ", U_sol)
        print("Solve time (ms): ", (end_time - start_time) * 1000)
        return

    else:
        ocp.update_initial_state(x_init)
        ocp.update_gait_sequence(shift_idx=gait_idx)
        ocp.init_solver(solver="fatrop", compile_solver=compile_solver)
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

            time.sleep(dt)


if __name__ == "__main__":
    main()
