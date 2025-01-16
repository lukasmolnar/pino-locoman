import time
import numpy as np
import pinocchio as pin

from helpers import *
from ocp_rnea import OptimalControlRNEA

# Problem parameters
# robot = B2(reference_pose="standing")
robot = B2G(reference_pose="standing_with_arm_up", ignore_arm=False)
gait_type = "trot"
gait_nodes = 14
ocp_nodes = 8
dt = 0.03

# Only for B2G
arm_f_des = np.array([0, 0, 0])
arm_vel_des = np.array([0, 0, 0])

# Tracking goal: linear and angular momentum
com_goal = np.array([0, 0, 0, 0, 0, 0])

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
    ocp = OptimalControlRNEA(
        robot=robot,
        nodes=ocp_nodes,
    )
    ocp.set_com_goal(com_goal)
    ocp.set_arm_task(arm_f_des, arm_vel_des)

    x_init = np.concatenate((q0, np.zeros(model.nv)))
    gait_idx = 0

    ocp.update_initial_state(x_init)
    ocp.update_gait_sequence(shift_idx=gait_idx)
    ocp.init_solver(solver=solver, compile_solver=compile_solver)
    ocp.solve(retract_all=True)

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

            # Separate external forces
            # tau_rnea = pin.rnea(model, data, q.flatten(), v.flatten(), a.flatten())
            # tau_ext = np.zeros(model.nv)
            # for idx, frame_id in enumerate(robot.ee_ids):
            #     J = pin.computeFrameJacobian(model, data, q.flatten(), frame_id, pin.LOCAL_WORLD_ALIGNED)
            #     J_lin = J[:3]
            #     f_ext = forces[idx * 3 : (idx + 1) * 3]
            #     tau_ext += J_lin.T @ f_ext
            # if robot.arm_ee_id:
            #     J = pin.computeFrameJacobian(model, data, q.flatten(), robot.arm_ee_id, pin.LOCAL_WORLD_ALIGNED)
            #     J_lin = J[:3]
            #     f_ext = forces[-3:]
            #     tau_ext += J_lin.T @ f_ext
            # tau_rnea -= tau_ext

            tau_total = np.concatenate((np.zeros(6), tau))
            print("tau gap: ", tau_total - tau_rnea)

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