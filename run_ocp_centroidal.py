import time
import numpy as np
import pinocchio as pin
import casadi as ca

from utils.helpers import *
from optimal_control_problem import OCP_Centroidal

# Problem parameters
# robot = B2(dynamics="centroidal", reference_pose="standing")
robot = B2G(dynamics="centroidal", reference_pose="standing_with_arm_up", ignore_arm=False)
gait_type = "trot"
gait_period = 0.5
nodes = 14
dt_min = 0.02  # used for simulation
dt_max = 0.05

# Tracking target: Base velocity (and arm task for B2G)
base_vel_des = np.array([0.3, 0, 0, 0, 0, 0])  # linear + angular
arm_f_des = np.array([0, 0, 0])
arm_vel_des = np.array([0.3, 0.3, 0])

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
    robot.initialize_weights()

    robot_instance = robot.robot
    model = robot.model
    data = robot.data
    q0 = robot.q0
    print(model)
    print(q0)

    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    # Setup OCP
    ocp = OCP_Centroidal(robot, nodes)
    ocp.setup_problem()
    ocp.set_time_params(dt_min, dt_max)
    ocp.set_swing_params(swing_height, swing_vel_limits)
    ocp.set_tracking_target(base_vel_des, arm_f_des, arm_vel_des)
    ocp.set_weights(robot.Q_diag, robot.R_diag)

    x_init = np.concatenate((np.zeros(6), q0))
    t_current = 0

    ocp.update_initial_state(x_init)
    ocp.update_gait_sequence(t_current)
    ocp.init_solver(solver, warm_start=False)

    if solver == "fatrop" and compile_solver:
        ocp.compile_solver()

        # Evaluate solver function that was compiled
        contact_schedule = ocp.opti.value(ocp.contact_schedule)
        swing_schedule = ocp.opti.value(ocp.swing_schedule)
        n_contacts = ocp.opti.value(ocp.n_contacts)
        swing_period = ocp.opti.value(ocp.swing_period)

        params = [x_init, dt_min, dt_max, contact_schedule, swing_schedule, n_contacts, swing_period,
                  swing_height, swing_vel_limits, robot.Q_diag, robot.R_diag, base_vel_des]
        if ocp.arm_ee_id:
            params += [arm_f_des, arm_vel_des]

        start_time = time.time()
        sol_x = ocp.solver_function(*params)
        end_time = time.time()
        ocp.solve_time = end_time - start_time

        ocp.retract_stacked_sol(sol_x, retract_all=True)
    else:
        ocp.solve(retract_all=True)

    print("Solve time (ms):", ocp.solve_time * 1000)

    if debug:
        for k in range(nodes):
            q = ocp.qs[k]
            h = ocp.hs[k]
            u = ocp.us[k]

            forces = u[:robot.nf]
            v_j = u[robot.nf:]
            v_b = ocp.dyn.get_base_velocity()(h, q, v_j)
            v_b = np.array(v_b).flatten()
            v = np.concatenate((v_b, v_j))
            pin.computeAllTerms(model, data, q, v)

            print("k: ", k)
            print("h: ", h.T)
            print("q: ", q.T)
            print("h true: ", data.hg / data.mass[0])

            # RNEA
            if k < nodes - 1:
                h_next = ocp.hs[k + 1]
                q_next = ocp.qs[k + 1]
                v_j_next = ocp.us[k + 1][robot.nf:]
                v_b_next = ocp.dyn.get_base_velocity()(h_next, q_next, v_j_next)
                v_b_next = np.array(v_b_next).flatten()
                v_next = np.concatenate((v_b_next, v_j_next))
            else:
                break
            a = (v_next - v) / dt_min

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
            print("tau rnea: ", tau_rnea)

    # Visualize
    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    for _ in range(50):
        for k in range(nodes):
            q = ocp.qs[k]
            robot_instance.display(q)      
            dt = ocp.opti.value(ocp.dts[k])  
            time.sleep(dt)
        time.sleep(1)

if __name__ == "__main__":
    main()
