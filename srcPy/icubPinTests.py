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
    # with open('leftArmData.txt', 'w+') as f:
    #     # Joints with their names
    #     f.write("JOINTS:\n")
    #     for i in range(model.njoints):
    #         f.write(f"  {i}: {model.names[i]} ({model.joints[i].shortname()})\n")
    #     
    #     # Frames with details
    #     f.write("\nFRAMES:\n")
    #     for i in range(model.nframes):
    #         frame = model.frames[i]
    #         f.write(f"  {i}: {frame.name} (type={frame.type}, parentJoint={frame.parentJoint})\n")
    # f.close()
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

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

    """
    Rotating the Right Hand to have 'z' inwards
    """
    rot_x_pi = pin.rpy.rpyToMatrix(np.array([-np.pi/2, 0, 0]))
    rot_c1 = pin.rpy.rpyToMatrix(np.array([0, -np.pi/2, -np.pi/2]))
    rot_c2 = pin.rpy.rpyToMatrix(np.array([np.pi/2, 0, 0]))
    H_world = pin.SE3(np.eye(3), np.zeros((3, )))
    H_object = pin.SE3(np.eye(3), np.ones((3, )))
    H_l = pin.SE3(rot_c1, np.array([0, -3, 0]))
    H_r = pin.SE3(rot_c2, np.array([0, 3, 0]))
    fc_r = np.array([0, 0, 5, 0, 0, 0])
    fc_l = np.array([0, 0, 5, 0, 0, 0])
    print(f'fc_r: {fc_r}')
    print(f'fc_l: {fc_l}')
    fo_r = H_r.dualAction @ fc_r
    fo_l = H_l.dualAction @ fc_l
    # print(fo_r)
    # print(fo_l)
    # print(fo_r + fo_l)

    """ goc1 -1 """
    H_o_cl = H_object.inverse() * (H_object * H_l) 
    H_o_cr = H_object.inverse() * (H_object * H_r) 

    H_cl_o = H_o_cl.inverse()
    H_cr_o = H_o_cr.inverse()

    fo_l = H_cl_o.dualAction @ fc_l
    fo_r = H_cr_o.dualAction @ fc_r
    print(f'fo_r: {fo_r}')
    print(f'fo_l: {fo_l}')
    print(f'fo_l + fo_r: {fo_r + fo_l}')
    sys.exit()

    """ 3D plot"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    plotCoordinateFrame(ax, np.array(H_world), name='world')
    plotCoordinateFrame(ax, np.array(H_object), name='object')
    plotCoordinateFrame(ax, np.array(H_object * H_r), name='c_right')
    plotCoordinateFrame(ax, np.array(H_object * H_l), name='c_left')
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
