from time import sleep

import numpy as np
from numpy.linalg import norm, solve
import pinocchio as pin

from helpers import load

URDF_PATH = "b2g_description/urdf/b2g.urdf"
SRDF_PATH = "b2g_description/srdf/b2g.srdf"
REFERENCE_POSE = "standing_with_arm_up"

robot = load(URDF_PATH, SRDF_PATH)
model = robot.model
data = robot.data
print(model)

robot.q0 = robot.model.referenceConfigurations[REFERENCE_POSE]
pin.forwardKinematics(model, data, robot.q0)
print(robot.q0)

robot.initViewer()
robot.loadViewerModel("pinocchio")
robot.display(robot.q0)

q = robot.q0.copy()

# Tracking task
N_TASKS = 2
ID_body = 1
ID_ee = 19
oMdes_body = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.55]))  # TODO: adapt body pose
oMdes_ee = pin.SE3(np.eye(3), np.array([0.5, 0.0, 1.0]))

eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12

i = 0
while True:
    pin.forwardKinematics(model, data, q)
    errors = []
    jacobians = []

    # EE task
    iMd_ee = data.oMi[ID_ee].actInv(oMdes_ee)
    err_ee = pin.log(iMd_ee).vector  # in joint frame
    J_ee = pin.computeJointJacobian(model, data, q, ID_ee)  # in joint frame
    J_ee = J_ee[:, 6:]  # ignore floating base
    J_ee = -np.dot(pin.Jlog6(iMd_ee.inverse()), J_ee)
    errors.append(err_ee)
    jacobians.append(J_ee)

    # Body task
    iMd_body = data.oMi[ID_body].actInv(oMdes_body)
    err_body = pin.log(iMd_body).vector  # in joint frame
    J_body = pin.computeJointJacobian(model, data, q, ID_body)  # in joint frame
    J_body = J_body[:, 6:]  # ignore floating base
    J_body = -np.dot(pin.Jlog6(iMd_body.inverse()), J_body)
    errors.append(err_body)
    jacobians.append(J_body)

    # Combine tasks
    err = np.concatenate(errors)
    J = np.vstack(jacobians)

    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break

    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6*N_TASKS), err))
    v = np.concatenate((np.zeros(6), v))  # add floating base zeros
    q = pin.integrate(model, q, v * DT)
    if not i % 10:
        print("%d: error = %s" % (i, err.T))
    i += 1

if success:
    print("Convergence achieved!")
else:
    print(
        "\nWarning: the iterative algorithm has not reached convergence to the desired precision"
    )

print("\nresult: %s" % q.flatten().tolist())
print("\nfinal error: %s" % err.T)

robot.display(q)
sleep(10)