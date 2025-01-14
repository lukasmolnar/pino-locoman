import time
import numpy as np
import pinocchio as pin
import casadi

from helpers import *
from optimal_control_problem import OptimalControlProblem

# Problem parameters
# robot = B2(reference_pose="standing")
robot = B2G(reference_pose="standing_with_arm_up", ignore_arm=False)
gait_type = "trot"
gait_nodes = 14
ocp_nodes = 8
dt = 0.03

# Only for B2G
arm_f_des = np.array([0, 0, -100])
arm_vel_des = np.array([0.1, 0, 0])

# Tracking goal: linear and angular momentum
com_goal = np.array([0.1, 0, 0, 0, 0, 0])

# Solver
solver = "osqp"
compile_solver = False
load_compiled_solver = None

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
    )
    ocp.set_com_goal(com_goal)
    ocp.set_arm_task(arm_f_des, arm_vel_des)

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
        ocp.update_contact_schedule(shift_idx=gait_idx)
        ocp.init_solver(solver=solver, compile_solver=compile_solver)
        ocp.solve(retract_all=True)
        print("Solve time (ms):", ocp.solve_time * 1000)

    print("Final h: ", ocp.hs[-1].T)
    print("Final q: ", ocp.qs[-1].T)

    if debug:
        for k in range(ocp_nodes):
            q = ocp.qs[k]
            h = ocp.hs[k]
            u = ocp.us[k]
            pin.computeAllTerms(model, data, q, np.zeros(model.nv))
            print("k: ", k)
            print("h: ", h.T)
            print("q: ", q.T)
            print("u: ", u.T)
            print("com: ", data.com[0])

    # Visualize
    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    for _ in range(50):
        for q in ocp.qs:
            robot_instance.display(q)        
            time.sleep(dt)


if __name__ == "__main__":
    main()
