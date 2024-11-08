from time import sleep

import numpy as np
from numpy.linalg import norm, solve
import pinocchio as pin

from load import load

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

# Track end effector position through IK
JOINT_ID = 19
oMdes = pin.SE3(np.eye(3), np.array([0.5, 0.0, 1.0]))

eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12

i = 0
while True:
    pin.forwardKinematics(model, data, q)
    iMd = data.oMi[JOINT_ID].actInv(oMdes)
    err = pin.log(iMd).vector  # in joint frame
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pin.computeJointJacobian(model, data, q, JOINT_ID)  # in joint frame
    J = J[:, 7:]  # ignore floating base
    J = -np.dot(pin.Jlog6(iMd.inverse()), J)
    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    v = np.concatenate((np.zeros(7), v))  # add floating base zeros
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