import time
import numpy as np
import pinocchio as pin

from helpers import *
from ocp_rnea import OCP_RNEA

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

# Compiled solver
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
    ocp = OCP_RNEA(
        robot=robot,
        nodes=ocp_nodes,
    )
    ocp.set_com_goal(com_goal)
    ocp.set_arm_task(arm_f_des, arm_vel_des)

    x_init = np.concatenate((q0, np.zeros(model.nv)))
    gait_idx = 0

    ocp.update_initial_state(x_init)
    ocp.update_contact_schedule(shift_idx=gait_idx)
    ocp.init_solver(solver=solver, compile_solver=compile_solver)
    ocp.solve(retract_all=True)

    if debug:
        for k in range(ocp_nodes):
            q = ocp.qs[k]
            v = ocp.vs[k]
            tau = ocp.taus[k]
            forces = ocp.fs[k]
            print("q: ", q.T)
            print("v: ", v.T)
            print("tau: ", tau.T)

            if k < ocp_nodes - 1:
                v_next = ocp.vs[k + 1]
            else:
                break
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
            print("tau rnea: ", tau_rnea)

            tau_total = np.concatenate((np.zeros(6), tau))
            print("tau gap: ", tau_total - tau_rnea)

    print("Solve time (ms):", ocp.solve_time * 1000)

    # Visualize
    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    for _ in range(100):
        for k in range(ocp_nodes):
            q = ocp.qs[k]
            robot_instance.display(q)
            time.sleep(dt)

if __name__ == "__main__":
    main()