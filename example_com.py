from time import sleep

import numpy as np
import pinocchio as pin

from helpers import load

URDF_PATH = "b2g_description/urdf/b2g.urdf"
SRDF_PATH = "b2g_description/srdf/b2g.srdf"
REFERENCE_POSE = "standing_with_arm_up"

robot = load(URDF_PATH, SRDF_PATH)
model = robot.model
data = robot.data

robot.q0 = robot.model.referenceConfigurations[REFERENCE_POSE]
pin.forwardKinematics(model, data, robot.q0)

lfFoot, rfFoot, lhFoot, rhFoot = "FL_foot", "FR_foot", "RL_foot", "RR_foot"

foot_frames = [lfFoot, rfFoot, lhFoot, rhFoot]
foot_frame_ids = [robot.model.getFrameId(frame_name) for frame_name in foot_frames]
foot_joint_ids = [
    robot.model.frames[robot.model.getFrameId(frame_name)].parent
    for frame_name in foot_frames
]
print(foot_frame_ids)
print(foot_joint_ids)
pin.forwardKinematics(model, data, robot.q0)
pin.framesForwardKinematics(model, data, robot.q0)

constraint_models = []

for j, frame_id in enumerate(foot_frame_ids):
    contact_model_lf1 = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_3D,
        robot.model,
        foot_joint_ids[j],
        robot.model.frames[frame_id].placement,
        0,
        data.oMf[frame_id],
    )

    constraint_models.extend([contact_model_lf1])

robot.initViewer()
robot.loadViewerModel("pinocchio")
robot.display(robot.q0)

constraint_datas = [cm.createData() for cm in constraint_models]

q = robot.q0.copy()

pin.computeAllTerms(model, data, q, np.zeros(model.nv))
kkt_constraint = pin.ContactCholeskyDecomposition(model, constraint_models)
constraint_dim = sum([cm.size() for cm in constraint_models])
N = 10000
eps = 1e-10
mu = 0.0
# q_sol = (q[:] + np.pi) % np.pi - np.pi
q_sol = q.copy()
robot.display(q_sol)

# Bring CoM between the two feet.

mass = data.mass[0]


def squashing(model, data, q_in):
    q = q_in.copy()
    y = np.ones((constraint_dim))

    N_full = 200

    # Decrease CoMz
    com_drop_amp = 0.2
    pin.computeAllTerms(model, data, q, np.zeros(model.nv))
    com_base = data.com[0].copy()
    kp = 1.0
    speed = 1.0

    def com_des(k):
        return com_base - np.array(
            [
                0.0,
                0.0,
                np.abs(com_drop_amp * np.sin(2.0 * np.pi * k * speed / (N_full))),
            ]
        )

    for k in range(N):
        pin.computeAllTerms(model, data, q, np.zeros(model.nv))
        pin.computeJointJacobians(model, data, q)
        com_act = data.com[0].copy()
        com_err = com_act - com_des(k)
        kkt_constraint.compute(model, data, constraint_models, constraint_datas, mu)
        constraint_value = np.concatenate(
            [cd.c1Mc2.translation for cd in constraint_datas]
        )
        # constraint_value = np.concatenate([pin.log6(cd.c1Mc2) for cd in constraint_datas])
        J = np.vstack(
            [
                pin.getFrameJacobian(
                    model, data, cm.joint1_id, cm.joint1_placement, cm.reference_frame
                )[:3, :]
                for cm in constraint_models
            ]
        )
        primal_feas = np.linalg.norm(constraint_value, np.inf)
        dual_feas = np.linalg.norm(J.T.dot(constraint_value + y), np.inf)
        print("primal_feas:", primal_feas)
        print("dual_feas:", dual_feas)
        # if primal_feas < eps and dual_feas < eps:
        #    print("Convergence achieved")
        #    break
        print("constraint_value:", np.linalg.norm(constraint_value))
        print("com_error:", np.linalg.norm(com_err))
        rhs = np.concatenate(
            [-constraint_value - y * mu, kp * mass * com_err, np.zeros(model.nv - 3)]
        )
        dz = kkt_constraint.solve(rhs)
        dy = dz[:constraint_dim]
        dq = dz[constraint_dim:]
        alpha = 1.0
        q = pin.integrate(model, q, -alpha * dq)
        y -= alpha * (-dy + y)
        robot.display(q)
        sleep(0.05)
    return q


q_new = squashing(model, data, robot.q0)