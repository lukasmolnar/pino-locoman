import time
import numpy as np
import pinocchio as pin
import casadi as ca
import matplotlib.pyplot as plt

from utils.robot import *
from optimization import make_ocp
from ocp_args import OCP_ARGS

# Parameters
robot = B2(reference_pose="standing", payload=None)
# robot = B2G(reference_pose="standing_with_arm_up", ignore_arm=False)
dynamics ="whole_body_rnea"
gait_type = "trot"
gait_period = 0.8
nodes = 14
dt_min = 0.01  # used for simulation
dt_max = 0.08

# Tracking targets
base_vel_des = np.array([0.2, 0, 0, 0, 0, 0])  # linear + angular
ext_force_des = np.array([0, 0, 0])
arm_vel_des = np.array([0, 0, 0])

# Swing params
swing_height = 0.07
swing_vel_limits = [0.1, -0.2]

# MPC
mpc_loops = 100

# Solver
solver = "osqp"
warm_start = True
compile_solver = False
load_compiled_solver = None
# load_compiled_solver = "libsolver_b2_wb_aj_N14.so"

debug = False  # print info
plot = False


def mpc_loop(ocp, robot_instance):
    x_init = ocp.x_nom
    solve_times = []
    if dynamics == "whole_body_rnea":
        tau_prev = np.zeros(robot.nj)  # previous torque solution

    if solver == "fatrop" and compile_solver:
        if load_compiled_solver:
            # Load solver
            solver_function = ca.external("compiled_solver", "codegen/lib/" + load_compiled_solver)
        else:
            # Initialize solver and compile it
            ocp.init_solver(solver, warm_start)
            ocp.compile_solver()
            solver_function = ocp.solver_function

        # Warm start (dual variables)
        # lam_g_warm_start = ocp.opti.value(ocp.opti.lam_g, ocp.opti.initial())

        # Fixed parameters
        Q_diag = ocp.opti.value(ocp.Q_diag)
        R_diag = ocp.opti.value(ocp.R_diag)
        if dynamics == "whole_body_rnea":
            W_diag = ocp.opti.value(ocp.W_diag)  # weights on previous torque solution

        for k in range(mpc_loops):
            # Update parameters
            t_current = k * dt_min
            ocp.update_initial_state(x_init)
            ocp.update_gait_sequence(t_current)
            contact_schedule = ocp.opti.value(ocp.contact_schedule)
            swing_schedule = ocp.opti.value(ocp.swing_schedule)
            n_contacts = ocp.opti.value(ocp.n_contacts)
            swing_period = ocp.opti.value(ocp.swing_period)

            params = [x_init, dt_min, dt_max, contact_schedule, swing_schedule, n_contacts,
                      swing_period, swing_height, swing_vel_limits, Q_diag, R_diag, base_vel_des]

            if ocp.ext_force_frame:
                params += [ext_force_des]
            if ocp.arm_ee_frame:
                params += [arm_vel_des]
            if warm_start:
                ocp.warm_start()
                x_warm_start = ocp.opti.value(ocp.opti.x, ocp.opti.initial())
                params += [x_warm_start]
            if dynamics == "whole_body_rnea":
                params += [tau_prev, W_diag]

            # Solve
            start_time = time.time()
            sol_x = solver_function(*params)
            end_time = time.time()
            sol_time = end_time - start_time
            solve_times.append(sol_time)

            print("Solve time (ms): ", sol_time * 1000)

            # Retract solution and update x_init
            ocp.retract_stacked_sol(sol_x, retract_all=False)
            x_init = ocp.dyn.state_integrate()(x_init, ocp.DX_prev[1])
            if dynamics == "whole_body_rnea":
                tau_prev = ocp.get_tau_sol(i=1)  # solution for the next time step

            robot_instance.display(ocp.q_sol[-1])  # display last q

    else:
        # Initialize params
        ocp.update_initial_state(x_init)
        ocp.update_gait_sequence(t_current=0)
        if dynamics == "whole_body_rnea":
            ocp.update_previous_torques(tau_prev)
        ocp.update_solver_params(warm_start)

        # Initialize solver
        ocp.init_solver()

        for k in range(mpc_loops):
            # Update parameters
            t_current = k * dt_min
            ocp.update_initial_state(x_init)
            ocp.update_gait_sequence(t_current)
            if warm_start:
                ocp.warm_start()
            if dynamics == "whole_body_rnea":
                ocp.update_previous_torques(tau_prev)
            ocp.update_solver_params(warm_start)

            # Solve
            ocp.solve(retract_all=False)
            solve_times.append(ocp.solve_time)

            # Update x_init
            x_init = ocp.dyn.state_integrate()(x_init, ocp.DX_prev[1])
            robot_instance.display(ocp.q_sol[-1])  # display last q

    print("Avg solve time (ms): ", np.average(solve_times) * 1000)
    print("Std solve time (ms): ", np.std(solve_times) * 1000)

    return ocp


def main():
    # Initialize robot
    robot.set_gait_sequence(gait_type, gait_period)
    robot_instance = robot.robot
    model = robot.model
    data = robot.data
    q0 = robot.q0
    print("Model: ", model)
    print("q0: ", q0)

    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    robot_instance.display(q0)

    # Setup OCP
    default_args = OCP_ARGS[dynamics]
    ocp = make_ocp(
        dynamics=dynamics,
        default_args=default_args,
        robot=robot,
        nodes=nodes,
        solver=solver,
    )
    ocp.set_time_params(dt_min, dt_max)
    ocp.set_swing_params(swing_height, swing_vel_limits)
    ocp.set_tracking_targets(base_vel_des, ext_force_des, arm_vel_des)

    # Run MPC
    ocp = mpc_loop(ocp, robot_instance)

    T = sum([ocp.opti.value(dt) for dt in ocp.dts])
    print("Horizon length (s): ", T)

    if debug:
        print("************** DEBUG **************")
        tau_diffs = []
        tau_b_norms = []
        tau_j_sol = []
        for k in range(len(ocp.q_sol)):
            q = ocp.q_sol[k].flatten()
            v = ocp.v_sol[k].flatten()
            a = ocp.a_sol[k].flatten()
            forces = ocp.forces_sol[k].flatten()

            ee_frames = ocp.foot_frames.copy()
            if ocp.ext_force_frame:
                ee_frames.append(ocp.ext_force_frame)

            # Evaluate EOM
            M = pin.crba(model, data, q)
            nle = pin.nonLinearEffects(model, data, q, v)
            tau_ext = np.zeros(M.shape[0])
            for idx, frame_id in enumerate(ee_frames):
                f_world = forces[idx * 3 : (idx + 1) * 3]
                J_c = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
                J_c_lin = J_c[:3, :]
                tau_ext += J_c_lin.T @ f_world

            tau_all = M @ a + nle - tau_ext

            # RNEA
            pin.framesForwardKinematics(model, data, q)
            f_ext = [pin.Force(np.zeros(6)) for _ in range(model.njoints)]
            for idx, frame_id in enumerate(ee_frames):
                joint_id = model.frames[frame_id].parentJoint
                translation_joint_to_contact_frame = model.frames[frame_id].placement.translation
                rotation_world_to_joint_frame = data.oMi[joint_id].rotation.T
                f_world = forces[idx * 3 : (idx + 1) * 3]

                f_lin = rotation_world_to_joint_frame @ f_world
                f_ang = np.cross(translation_joint_to_contact_frame, f_lin)
                f = np.concatenate((f_lin, f_ang))
                f_ext[joint_id] = pin.Force(f)

            # Both RNEA functions work!
            tau_rnea = pin.rnea(model, data, q, v, a, f_ext)
            # tau_rnea = pin.rnea(model, data, q, v, a) - tau_ext

            tau_diff = tau_all - tau_rnea
            tau_b = tau_all[:6]
            tau_j = tau_all[6:]
            tau_diffs.append(np.linalg.norm(tau_diff))
            tau_b_norms.append(np.linalg.norm(tau_b))
            tau_j_sol.append(tau_j)
        
        print("Avg tau_diff: ", np.mean(tau_diffs))
        print("Std tau_diff: ", np.std(tau_diffs))
        print("Avg tau_b_norm: ", np.mean(tau_b_norms))
        print("Std tau_b_norm: ", np.std(tau_b_norms))

        if plot:
            # Plot q, v, tau
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            labels = ["FL hip", "FL thigh", "FL calf", "FR hip", "FR thigh", "FR calf",
                    "RL hip", "RL thigh", "RL calf", "RR hip", "RR thigh", "RR calf"]

            axs[0].set_title("Joint positions (q)")
            for j in range(12):
                # Ignore base (quaternion)
                axs[0].plot([q[7 + j] for q in ocp.q_sol], label=labels[j])
            axs[0].legend()

            axs[1].set_title("Joint velocities (v)")
            for j in range(12):
                # Ignore base
                axs[1].plot([v[6 + j] for v in ocp.v_sol], label=labels[j])
            axs[1].legend()

            axs[2].set_title("Joint torques (tau)")
            for j in range(12):
                axs[2].plot([tau[j] for tau in tau_j_sol], label=labels[j])
            axs[2].legend()

            plt.tight_layout()
            plt.show()

    # Visualize 
    for _ in range(50):
        for q in ocp.q_sol:
            robot_instance.display(q)
            time.sleep(dt_min)


if __name__ == "__main__":
    main()
