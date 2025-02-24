import time
import numpy as np
import pinocchio as pin
import casadi as ca
import matplotlib.pyplot as plt

from helpers import *
from ocp_rnea import OCP_RNEA

# Problem parameters
# robot = Go2(reference_pose="standing")
robot = B2G(reference_pose="standing_with_arm_up", ignore_arm=False)
gait_type = "trot"
gait_period = 0.5
nodes = 10
dt_min = 0.02  # used for simulation
dt_max = 0.05

# Only for B2G
arm_f_des = np.array([0, 0, 0])
arm_vel_des = np.array([0.3, 0, 0])

# Tracking goal: linear and angular momentum
com_goal = np.array([0.3, 0, 0, 0, 0, 0])

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
# load_compiled_solver = "libsolver_go2_dt_N12.so"

debug = False  # print info
plot = True


def mpc_loop(ocp, robot_instance, q0, N):
    x_init = np.concatenate((q0, np.zeros(robot.nv)))
    tau_prev = np.zeros(robot.nj)
    solve_times = []
    tau_diffs = []

    if compile_solver or load_compiled_solver:
        if load_compiled_solver:
            # Load solver
            solver_function = ca.external("compiled_solver", "codegen/lib/" + load_compiled_solver)
        else:
            # Initialize solver
            ocp.init_solver(solver, compile_solver, warm_start)
            solver_function = ocp.solver_function

        # Warm start (dual variables)
        # lam_g_warm_start = ocp.opti.value(ocp.opti.lam_g, ocp.opti.initial())

        for k in range(N):
            # Get parameters
            t_current = k * dt_min
            ocp.update_initial_state(x_init)
            ocp.update_previous_torques(tau_prev)
            ocp.update_gait_sequence(t_current)
            contact_schedule = ocp.opti.value(ocp.contact_schedule)
            swing_schedule = ocp.opti.value(ocp.swing_schedule)
            n_contacts = ocp.opti.value(ocp.n_contacts)

            params = [x_init, tau_prev, dt_min, dt_max, contact_schedule, swing_schedule, n_contacts,
                      robot.Q_diag, robot.R_diag, robot.W_diag, com_goal, swing_height, swing_vel_limits]

            if ocp.arm_ee_id:
                params += [arm_f_des, arm_vel_des]
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
            ocp._retract_stacked_sol(sol_x, retract_all=False)
            x_pred = ocp.dyn.state_integrate()(x_init, ocp.DX_prev[1])
            x_init = ocp._simulate_step(x_init, ocp.U_prev[0])
            x_diff = x_pred - x_init
            print("x_diff norm: ", np.linalg.norm(x_diff))

            tau_0 = ocp._get_torque_sol(idx=0)
            tau_diff = np.linalg.norm(tau_0 - tau_prev)
            tau_diffs.append(tau_diff)
            tau_prev = ocp._get_torque_sol(idx=1)  # update with next idx

            # lam_g_warm_start = sol_lam_g
            robot_instance.display(ocp.qs[-1])  # Display last q

    else:
        # Initialize solver
        ocp.init_solver(solver, compile_solver, warm_start)

        for k in range(N):
            t_current = k * dt_min
            ocp.update_initial_state(x_init)
            ocp.update_previous_torques(tau_prev)
            ocp.update_gait_sequence(t_current)
            if warm_start:
                ocp.warm_start()

            ocp.solve(retract_all=False)
            solve_times.append(ocp.solve_time)

            x_init = ocp._simulate_step(x_init, ocp.U_prev[0])
            robot_instance.display(ocp.qs[-1])  # Display last q

    print("Avg solve time (ms): ", np.average(solve_times) * 1000)
    print("Std solve time (ms): ", np.std(solve_times) * 1000)

    print("Avg tau diff: ", np.average(tau_diffs))
    print("Std tau diff: ", np.std(tau_diffs))

    T = sum([ocp.opti.value(dt) for dt in ocp.dts])
    print("Horizon length (s): ", T)

    return ocp


def main():
    robot.set_gait_sequence(gait_type, gait_period)
    if type(robot) == B2G and not robot.ignore_arm:
        robot.add_arm_task(arm_f_des, arm_vel_des)
    robot.initialize_weights(dynamics="rnea")

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
    ocp = OCP_RNEA(robot, nodes)
    ocp.set_time_params(dt_min, dt_max)
    ocp.set_com_goal(com_goal)
    ocp.set_swing_params(swing_height, swing_vel_limits)
    ocp.set_arm_task(arm_f_des, arm_vel_des)
    ocp.set_weights(robot.Q_diag, robot.R_diag, robot.W_diag)
    ocp = mpc_loop(ocp, robot_instance, q0, mpc_loops)

    if debug:
        for k in range(len(ocp.qs)):
            q = ocp.qs[k]
            v = ocp.vs[k]
            a = ocp.accs[k]
            tau = ocp.taus[k]
            forces = ocp.forces[k]
            print("q: ", q.T)
            print("v: ", v.T)
            print("tau: ", tau.T)

            # RNEA
            pin.framesForwardKinematics(model, data, q)
            f_ext = [pin.Force(np.zeros(6)) for _ in range(model.njoints)]
            for idx, frame_id in enumerate(robot.ee_ids):
                joint_id = model.frames[frame_id].parentJoint
                translation_joint_to_contact_frame = model.frames[frame_id].placement.translation
                rotation_world_to_joint_frame = data.oMi[joint_id].rotation.T
                f_world = forces[idx * 3 : (idx + 1) * 3].flatten()
                print(f"force {frame_id}: {f_world}")

                f_lin = rotation_world_to_joint_frame @ f_world
                f_ang = np.cross(translation_joint_to_contact_frame, f_lin)
                f = np.concatenate((f_lin, f_ang))
                f_ext[joint_id] = pin.Force(f)

            tau_rnea = pin.rnea(model, data, q, v, a, f_ext)

            tau_total = np.concatenate((np.zeros(6), tau.flatten()))
            print("tau gap: ", tau_total - tau_rnea)

    if plot:
        # Plot q, v, tau
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        labels = ["FL hip", "FL thigh", "FL calf", "FR hip", "FR thigh", "FR calf",
                  "RL hip", "RL thigh", "RL calf", "RR hip", "RR thigh", "RR calf"]

        axs[0].set_title("Joint positions (q)")
        for j in range(12):
            # Ignore base (quaternion)
            axs[0].plot([q[7 + j] for q in ocp.qs], label=labels[j])
        axs[0].legend()

        axs[1].set_title("Joint velocities (v)")
        for j in range(12):
            # Ignore base
            axs[1].plot([v[6 + j] for v in ocp.vs], label=labels[j])
        axs[1].legend()

        axs[2].set_title("Joint torques (tau)")
        for j in range(12):
            axs[2].plot([tau[j] for tau in ocp.taus], label=labels[j])
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    # Visualize
    for _ in range(50):
        for q in ocp.qs:
            robot_instance.display(q)
            time.sleep(dt_min)

if __name__ == "__main__":
    main()