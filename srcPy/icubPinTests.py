import sys
import os
import casadi as ca
from utils import *
from nmpcArm import armNMPC
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
# print(dir(cpin)) -> prints callable functions


def main():
    """ Option structs """
    dt = 0.05
    ocpParams = {
            'H': 50,
            'r': 1,
            'q': 7
            }

    armParams = {
            "path": os.getcwd() + '/../conf/model.urdf',
            "joints": ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_prosup', 'wrist_pitch', 'wrist_yaw'],
            "arm": 'l_'
            }

    icubCtrler = armNMPC(armParameters=armParams, method='inverse', Ts=dt, ocpParameters=ocpParams)

    """
    Control Loop
    """
    model, data = icubCtrler.pinModelandData()
    """ Print Arm Model Data """
    with open('leftArmData.txt', 'w+') as f:
        # Joints with their names
        f.write("JOINTS:\n")
        for i in range(model.njoints):
            f.write(f"  {i}: {model.names[i]} ({model.joints[i].shortname()})\n")
        
        # Frames with details
        f.write("\nFRAMES:\n")
        for i in range(model.nframes):
            frame = model.frames[i]
            f.write(f"  {i}: {frame.name} (type={frame.type}, parentJoint={frame.parentJoint})\n")
    f.close()
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    print(cmodel.njoints)

    q = ca.SX.sym('q', model.nq)
    v = ca.SX.sym('v', model.nv)
    a = ca.SX.sym('a', model.nv)
    f_ee = ca.SX.sym('f', 6)
    f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(model.njoints)]
    f_ext[6] = cpin.Force(f_ee)  # End-effector is joint index 6

    # f_ext = ca.horzcat(ca.DM.zeros((cmodel.nv, cmodel.njoints - 1)), f)
    tau = cpin.rnea(cmodel, cdata, q, v, a, f_ext)
    cpin.computeRNEADerivatives(cmodel, cdata, q, v, a, f_ext)

    du_dq = cdata.dtau_dq
    du_dv = cdata.dtau_dv
    du_da = cdata.M

    hk_rnea = ca.Function('RNEA', [q, v, a, f_ee], [tau], ['q', 'v', 'a', 'f_ee'], ['tau'])

    q_i = pin.neutral(model)
    pin.forwardKinematics(model, data, q_i)
    pin.updateFramePlacements(model, data)
    hand_str = armParams['arm'] + 'hand'
    hand_id = model.getFrameId(hand_str)


if __name__ == "__main__":
    main()
