import time
import numpy as np
import pinocchio as pin

from helpers import *
from ocp_rnea import OptimalControlRNEA

# Problem parameters
# robot = B2(reference_pose="standing")
robot = B2G(reference_pose="standing_with_arm_up", ignore_arm=False)
gait_type = "stand"
gait_nodes = 14
ocp_nodes = 8
dt = 0.03

# Only for B2G
arm_f_des = np.array([0, 0, 0])
arm_vel_des = np.array([0, 0, 0])

# Tracking goal: linear and angular momentum
com_goal = np.array([0, 0, 0, 0, 0, 0])

# MPC
mpc_loops = 10

# Compiled solver
compile_solver = False
load_compiled_solver = None

debug = False  # print info


def mpc_loop(ocp, robot_instance, q0, N):
    x_init = np.concatenate((q0, np.zeros(robot_instance.model.nv)))
    solve_times = []

    # Initialize solver
    ocp.init_solver(solver="osqp", compile_solver=compile_solver)

    for k in range(N):
        ocp.update_initial_state(x_init)
        ocp.update_gait_sequence(shift_idx=k)
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
    ocp = OptimalControlRNEA(
        robot=robot,
        nodes=ocp_nodes,
    )
    ocp.set_com_goal(com_goal)
    ocp.set_arm_task(arm_f_des, arm_vel_des)
    ocp = mpc_loop(ocp, robot_instance, q0, mpc_loops)

    if debug:
        for k in range(ocp_nodes):
            q = ocp.qs[k].T
            v = ocp.vs[k].T
            tau = ocp.taus[k].T
            forces = ocp.fs[k].T
            print("q: ", q)
            print("v: ", v)
            print("tau: ", tau)

            if k < ocp_nodes - 1:
                v_next = ocp.vs[k + 1].T
            else:
                break
            a = (v_next - v) / dt
            print("a: ", a)

            # RNEA
            pin.framesForwardKinematics(model, data, q.flatten())
            f_ext = [pin.Force(np.zeros(6)) for _ in range(model.njoints)]
            for idx, frame_id in enumerate(robot.ee_ids):
                joint_id = model.frames[frame_id].parentJoint
                translation_joint_to_contact_frame = model.frames[frame_id].placement.translation
                rotation_world_to_joint_frame = data.oMi[joint_id].rotation.T
                f_world = forces[idx * 3 : (idx + 1) * 3]
                f_lin = rotation_world_to_joint_frame @ f_world
                f_ang = np.cross(translation_joint_to_contact_frame, f_lin)
                f = np.concatenate((f_lin, f_ang))
                f_ext[joint_id] = pin.Force(f)
                print("f_ext: ", f)
            tau_rnea = pin.rnea(model, data, q.flatten(), v.flatten(), a.flatten(), f_ext)
            print("tau rnea: ", tau_rnea)

            tau_total = np.concatenate((np.zeros(6), tau))
            print("tau gap: ", tau_total - tau_rnea)

    # Visualize
    for _ in range(50):
        for k in range(len(ocp.qs)):
            q = ocp.qs[k]
            robot_instance.display(q)
            time.sleep(dt)

if __name__ == "__main__":
    main()