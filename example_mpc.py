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
arm_f_des = np.array([0, 0, 0])
arm_vel_des = np.array([0.1, 0, 0])

# Tracking goal: linear and angular momentum
com_goal = np.array([0.1, 0, 0, 0, 0, 0])

# MPC
mpc_loops = 50

# Compiled solver
# NOTE: Make sure ocp_nodes and dt are correct!
compile_solver = False
load_compiled_solver = None
# load_compiled_solver = "libsolver_trot_N8_dt03.so"

debug = False  # print info


def mpc_loop(ocp, robot_instance, q0, N):
    x_init = np.concatenate((np.zeros(6), q0))
    solve_times = []

    if load_compiled_solver:
        # Load solver
        compiled_solver = casadi.external("compiled_solver", "codegen/lib/" + load_compiled_solver)
        ocp.hs = []
        ocp.qs = []

        for k in range(N):
            # TODO: Look into warm-starting
            contact_schedule = ocp.gait_sequence.shift_contact_schedule(k)[:,:ocp_nodes]
            start_time = time.time()
            DX_sol, U_sol = compiled_solver(x_init, contact_schedule, k, com_goal, arm_f_des, arm_vel_des)
            end_time = time.time()
            sol_time = end_time - start_time
            solve_times.append(sol_time)

            print("Solve time (ms): ", sol_time * 1000)
            
            x_init = ocp.dyn.state_integrate()(x_init, DX_sol)
            h = np.array(x_init[:6])
            q = np.array(x_init[6:])
            ocp.hs.append(h)
            ocp.qs.append(q)
            robot_instance.display(q)

    else:
        # Initialize solver
        ocp.init_solver(solver="osqp", compile_solver=compile_solver)

        for k in range(N):
            ocp.update_initial_state(x_init)
            ocp.update_contact_schedule(shift_idx=k)
            ocp.warm_start()
            ocp.solve(retract_all=False)
            solve_times.append(ocp.solve_time)

            x_init = ocp.dyn.state_integrate()(x_init, ocp.DX_prev[1])
            robot_instance.display(ocp.qs[-1])  # Display last q

    print("Avg solve time (ms): ", np.average(solve_times) * 1000)
    print("Std solve time (ms): ", np.std(solve_times) * 1000)

    return ocp


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

    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    robot_instance.display(q0)

    # Setup OCP
    ocp = OptimalControlProblem(
        robot=robot,
        nodes=ocp_nodes,
    )
    ocp.set_com_goal(com_goal)
    ocp.set_arm_task(arm_f_des, arm_vel_des)
    ocp = mpc_loop(ocp, robot_instance, q0, mpc_loops)

    print("Final h: ", ocp.hs[-1].T)
    print("Final q: ", ocp.qs[-1].T)

    # Visualize 
    for _ in range(50):
        for k in range(len(ocp.qs)):
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
