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
ocp_nodes = 10
dt = 0.03

# Only for B2G
arm_f_des = np.array([0, 0, 0])
arm_vel_des = np.array([0.1, 0, 0])

# Tracking goal: linear and angular momentum
com_goal = np.array([0.1, 0, 0, 0, 0, 0])

# Compiled solver
compile_solver = False
load_compiled_solver = None

debug = False  # print info


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
    ocp.init_solver(solver="fatrop", compile_solver=compile_solver)
    ocp.solve(retract_all=True)

    for k in range(ocp_nodes):
        q = ocp.qs[k].T
        v = ocp.vs[k].T
        tau = ocp.taus[k].T
        f = ocp.fs[k].T
        print("q: ", q)
        print("v: ", v)
        print("tau: ", tau)
        print("f: ", f)

        # RNEA
        tau_rnea = pin.rnea(model, data, q0, np.zeros(model.nv), np.zeros(model.nv))
        print("tau_rnea: ", tau_rnea)

        tau_ext = np.zeros(model.nv)
        pin.computeAllTerms(model, data, q.flatten(), v.flatten())
        pin.updateFramePlacements(model, data)
        for idx, frame_id in enumerate(robot.ee_ids):
            J = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
            J_lin = J[:3]
            f_e = f[idx * 3 : (idx + 1) * 3]
            tau_ext += J_lin.T @ f_e

        if robot.arm_ee_id:
            arm_vel = ocp.dyn.get_frame_velocity(robot.arm_ee_id)(q, v)
            print("arm_vel: ", arm_vel)
        print("tau_ext: ", tau_ext)

    # Visualize
    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    for _ in range(50):
        for k in range(ocp_nodes):
            q = ocp.qs[k]
            robot_instance.display(q)
            time.sleep(dt)

if __name__ == "__main__":
    main()