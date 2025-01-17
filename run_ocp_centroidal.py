import time
import numpy as np
import pinocchio as pin
import casadi

from helpers import *
from ocp_centroidal import OCP_Centroidal

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

debug = True  # print info


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
    ocp = OCP_Centroidal(
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

    if debug:
        for k in range(ocp_nodes):
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
            if k < ocp_nodes - 1:
                h_next = ocp.hs[k + 1]
                q_next = ocp.qs[k + 1]
                v_j_next = ocp.us[k + 1][robot.nf:]
                v_b_next = ocp.dyn.get_base_velocity()(h_next, q_next, v_j_next)
                v_b_next = np.array(v_b_next).flatten()
                v_next = np.concatenate((v_b_next, v_j_next))
            else:
                break
            a = (v_next - v) / dt

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
        for q in ocp.qs:
            robot_instance.display(q)        
            time.sleep(dt)


if __name__ == "__main__":
    main()
