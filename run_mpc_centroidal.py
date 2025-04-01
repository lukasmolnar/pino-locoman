import time
import numpy as np
import pinocchio as pin
import casadi as ca

from utils.helpers import *
from optimal_control_problem import *

# Problem parameters
robot = B2(dynamics="whole_body_acc", reference_pose="standing", payload=None)
# robot = B2G(dynamics="centroidal_acc", reference_pose="standing_with_arm_up", ignore_arm=False)
ocp_class = OCPWholeBodyAcc
gait_type = "trot"
gait_period = 0.8
nodes = 14
dt_min = 0.01  # used for simulation
dt_max = 0.08

# Tracking targets
base_vel_des = np.array([0.2, 0, 0, 0, 0, 0.2])  # linear + angular
ext_force_des = np.array([0, 0, 0])
arm_vel_des = np.array([0, 0, 0])

# Swing params
swing_height = 0.07
swing_vel_limits = [0.1, -0.2]

# MPC
mpc_loops = 100

# Solver
solver = "fatrop"
warm_start = True
compile_solver = True
load_compiled_solver = None
# load_compiled_solver = "libsolver_trot_N8_dt03.so"

debug = False  # print info


def mpc_loop(ocp, robot_instance, x_init, N):
    solve_times = []

    if solver == "fatrop" and compile_solver:
        if load_compiled_solver:
            # Load solver
            solver_function = ca.external("compiled_solver", "codegen/lib/" + load_compiled_solver)
        else:
            # Initialize solver and compile it
            ocp.init_solver(solver, warm_start)
            ocp.compile_solver()
            solver_function = ocp.solver_function

        for k in range(N):
            # Get parameters
            t_current = k * dt_min
            ocp.update_initial_state(x_init)
            ocp.update_gait_sequence(t_current)
            contact_schedule = ocp.opti.value(ocp.contact_schedule)
            swing_schedule = ocp.opti.value(ocp.swing_schedule)
            n_contacts = ocp.opti.value(ocp.n_contacts)
            swing_period = ocp.opti.value(ocp.swing_period)

            params = [x_init, dt_min, dt_max, contact_schedule, swing_schedule, n_contacts, swing_period,
                      swing_height, swing_vel_limits, robot.Q_diag, robot.R_diag, base_vel_des]

            if ocp.ext_force_frame:
                params += [ext_force_des]
            if ocp.arm_ee_frame:
                params += [arm_vel_des]
            if warm_start:
                ocp.warm_start()
                x_warm_start = ocp.opti.value(ocp.opti.x, ocp.opti.initial())
                params += [x_warm_start]

            # Solve
            start_time = time.time()
            sol_x = solver_function(*params)
            end_time = time.time()
            sol_time = end_time - start_time
            solve_times.append(sol_time)

            print("Solve time (ms): ", sol_time * 1000)

            # Retract solution and update x_init
            ocp.retract_stacked_sol(sol_x, retract_all=False)
            x_init = ocp.dyn.state_integrate()(x_init, ocp.DX_prev[1])  # TODO: simulate step

            robot_instance.display(ocp.qs[-1])  # Display last q

    else:
        # Initialize solver
        ocp.init_solver(solver=solver, compile_solver=compile_solver)

        for k in range(N):
            t_current = k * dt_min
            ocp.update_initial_state(x_init)
            ocp.update_gait_sequence(t_current)
            if warm_start:
                ocp.warm_start()

            ocp.solve(retract_all=False)
            solve_times.append(ocp.solve_time)

            x_init = ocp.dyn.state_integrate()(x_init, ocp.DX_prev[1])  # TODO: simulate step
            robot_instance.display(ocp.qs[-1])  # Display last q

    print("Avg solve time (ms): ", np.average(solve_times) * 1000)
    print("Std solve time (ms): ", np.std(solve_times) * 1000)

    return ocp


def main():
    robot.set_gait_sequence(gait_type, gait_period)
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
    ocp = ocp_class(robot, nodes)
    ocp.setup_problem()
    ocp.set_time_params(dt_min, dt_max)
    ocp.set_swing_params(swing_height, swing_vel_limits)
    ocp.set_tracking_targets(base_vel_des, ext_force_des, arm_vel_des)
    ocp.set_weights(robot.Q_diag, robot.R_diag)

    x_init = robot.x_init
    ocp = mpc_loop(ocp, robot_instance, x_init, mpc_loops)

    if debug:
        for k in range(len(ocp.qs)):
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
    for _ in range(50):
        for q in ocp.qs:
            robot_instance.display(q)
            time.sleep(dt_min)


if __name__ == "__main__":
    main()
