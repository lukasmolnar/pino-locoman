import time
import numpy as np
import pinocchio as pin

from helpers import *
from ocp_rnea import OCP_RNEA

# Problem parameters
robot = Go2(reference_pose="standing")
# robot = B2G(reference_pose="standing_with_arm_up", ignore_arm=False)
gait_type = "trot"
gait_period = 0.5
nodes = 14
dt = 0.03
dt_sim = 0.01

# Only for B2G
arm_f_des = np.array([0, 0, 0])
arm_vel_des = np.array([0.3, 0, 0])

# Tracking goal: linear and angular momentum
com_goal = np.array([0.3, 0, 0, 0, 0, 0])

# Swing params
swing_height = 0.07
swing_vel_limits = [0.1, -0.2]

# Solver
solver = "fatrop"
compile_solver = True

debug = False  # print info


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

    # Setup OCP
    ocp = OCP_RNEA(robot, nodes)
    ocp.set_time_params(dt, dt_sim)
    ocp.set_com_goal(com_goal)
    ocp.set_swing_params(swing_height, swing_vel_limits)
    ocp.set_arm_task(arm_f_des, arm_vel_des)
    ocp.set_weights(robot.Q_diag, robot.R_diag, robot.W_diag)

    x_init = np.concatenate((q0, np.zeros(model.nv)))
    tau_prev = np.zeros(robot.nj)
    t_current = 0

    ocp.update_initial_state(x_init)
    ocp.update_previous_torques(tau_prev)
    ocp.update_gait_sequence(t_current)
    ocp.init_solver(solver, compile_solver, warm_start=False)

    if compile_solver:
        # Evaluate solver function that was compiled
        contact_schedule = ocp.opti.value(ocp.contact_schedule)
        swing_schedule = ocp.opti.value(ocp.swing_schedule)
        n_contacts = ocp.opti.value(ocp.n_contacts)

        params = [x_init, tau_prev, dt, dt_sim, contact_schedule, swing_schedule, n_contacts, robot.Q_diag,
                  robot.R_diag, robot.W_diag, com_goal, swing_height, swing_vel_limits]
        if ocp.arm_ee_id:
            params += [arm_f_des, arm_vel_des]

        start_time = time.time()
        sol_x = ocp.solver_function(*params)
        end_time = time.time()
        ocp.solve_time = end_time - start_time

        ocp._retract_stacked_sol(sol_x, retract_all=True)
    else:
        ocp.solve(retract_all=True)

    print("Solve time (ms):", ocp.solve_time * 1000)

    if debug:
        for k in range(nodes):
            q = ocp.qs[k]
            v = ocp.vs[k]
            tau = ocp.taus[k]
            forces = ocp.fs[k]
            print("k: ", k)
            print("q: ", q.T)
            print("v: ", v.T)
            print("tau: ", tau.T)

            if k == nodes - 1:
                break

            v_next = ocp.vs[k + 1]
            if k == 0:
                a = (v_next - v) / dt_sim
            else:
                a = (v_next - v) / dt

            # RNEA
            pin.framesForwardKinematics(model, data, q)
            f_ext = [pin.Force(np.zeros(6)) for _ in range(model.njoints)]
            for idx, frame_id in enumerate(robot.ee_ids):
                joint_id = model.frames[frame_id].parentJoint
                translation_joint_to_contact_frame = model.frames[frame_id].placement.translation
                rotation_world_to_joint_frame = data.oMi[joint_id].rotation.T
                f_world = forces[idx * 3 : (idx + 1) * 3]
                print(f"force {frame_id}: {f_world}")

                f_lin = rotation_world_to_joint_frame @ f_world
                f_ang = np.cross(translation_joint_to_contact_frame, f_lin)
                f = np.concatenate((f_lin, f_ang))
                f_ext[joint_id] = pin.Force(f)

            tau_rnea = pin.rnea(model, data, q, v, a, f_ext)

            tau_total = np.concatenate((np.zeros(6), tau))
            print("tau gap: ", tau_total - tau_rnea)

    # Visualize
    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    for _ in range(100):
        for k in range(nodes):
            q = ocp.qs[k]
            robot_instance.display(q)
            if k == 0:
                time.sleep(dt_sim)
            else:
                time.sleep(dt)
        time.sleep(1)

if __name__ == "__main__":
    main()