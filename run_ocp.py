import time
import numpy as np
import pinocchio as pin

from utils.robot import *
from optimization import make_ocp
from ocp_args import OCP_ARGS

# Parameters
# robot = B2(reference_pose="standing", payload=None)
robot = B2G(reference_pose="standing_with_arm_up", ignore_arm=False)
dynamics = "whole_body_rnea"
gait_type = "trot"
gait_period = 0.8
nodes = 16
tau_nodes = 3  # add torque limits for this many nodes
dt_min = 0.015  # used for simulation
dt_max = 0.06

# Tracking targets
base_vel_des = np.array([0.2, 0, 0, 0, 0, 0])  # linear + angular
ext_force_des = np.array([0, 0, 0])
arm_vel_des = np.array([0.2, 0, 0])

# Swing params
swing_height = 0.07
swing_vel_limits = [0.1, -0.2]

# Solver
solver = "osqp"
compile_solver = False

debug = False  # print info


def main():
    robot.set_gait_sequence(gait_type, gait_period)
    robot_instance = robot.robot
    model = robot.model
    data = robot.data
    q0 = robot.q0
    print(model)
    print(q0)

    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    # Setup OCP
    default_args = OCP_ARGS[dynamics]
    ocp = make_ocp(
        dynamics=dynamics,
        default_args=default_args,
        robot=robot,
        solver=solver,
        nodes=nodes,
        tau_nodes=tau_nodes,
    )
    ocp.set_time_params(dt_min, dt_max)
    ocp.set_swing_params(swing_height, swing_vel_limits)
    ocp.set_tracking_targets(base_vel_des, ext_force_des, arm_vel_des)

    x_init = ocp.x_nom
    t_current = 0
    if dynamics == "whole_body_rnea":
        tau_prev = np.zeros(robot.nj)  # previous torque solution
        ocp.update_previous_torques(tau_prev)

    ocp.update_initial_state(x_init)
    ocp.update_gait_sequence(t_current)
    ocp.init_solver()

    if solver == "fatrop" and compile_solver:
        ocp.compile_solver(warm_start=False)

        # Evaluate solver function that was compiled
        contact_schedule = ocp.opti.value(ocp.contact_schedule)
        swing_schedule = ocp.opti.value(ocp.swing_schedule)
        n_contacts = ocp.opti.value(ocp.n_contacts)
        swing_period = ocp.opti.value(ocp.swing_period)
        Q_diag = ocp.opti.value(ocp.Q_diag)
        R_diag = ocp.opti.value(ocp.R_diag)
        if dynamics == "whole_body_rnea":
            W_diag = ocp.opti.value(ocp.W_diag)  # weights on previous torque solution

        params = [x_init, dt_min, dt_max, contact_schedule, swing_schedule, n_contacts,
                  swing_period, swing_height, swing_vel_limits, Q_diag, R_diag, base_vel_des]

        if ocp.ext_force_frame:
            params += [ext_force_des]
        if ocp.arm_ee_frame:
            params += [arm_vel_des]
        if dynamics == "whole_body_rnea":
            params += [tau_prev, W_diag]

        start_time = time.time()
        sol_x = ocp.solver_function(*params)
        end_time = time.time()
        ocp.solve_time = end_time - start_time

        ocp.retract_stacked_sol(sol_x, retract_all=True)
    else:
        ocp.solve(retract_all=True)

    print("Solve time (ms):", ocp.solve_time * 1000)

    T = sum([ocp.opti.value(dt) for dt in ocp.dts])
    print("Horizon length (s): ", T)

    if debug:
        print("************** DEBUG **************")
        tau_diffs = []
        tau_b_norms = []
        tau_j_sol = []
        for k in range(nodes):
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

    # Visualize
    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    for _ in range(50):
        for k in range(nodes):
            q = ocp.q_sol[k]
            robot_instance.display(q)      
            dt = ocp.opti.value(ocp.dts[k])  
            time.sleep(dt)
        time.sleep(1)

if __name__ == "__main__":
    main()
