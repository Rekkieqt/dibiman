import sys
import os
import casadi as ca
import copy
from scipy.io import savemat
from utils import *
from nmpcReArmed import armNMPC
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
# print(dir(cpin)) -> prints callable functions


def main():
    """ Option structs """
    left = 'l_'
    right = 'r_'
    dt = 0.05
    ocpParams = {
            'r' : 1,
            'q' : 7,
            'H': 50,
            'Ts': dt
            }

    rightArm = {
            "path": os.getcwd() + '/../conf/model.urdf',
            "joints": ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_prosup', 'wrist_pitch'], # r_wrist_yaw
            "prefix": right
            }
    leftArm = copy.deepcopy(rightArm)
    leftArm["prefix"] = left
    armParams = [rightArm, leftArm]

    objParams = {
            "path": os.getcwd() + '/../conf/testBox.urdf',
            }

    iBee = armNMPC(armParameters=armParams, Ts=dt, ocpParameters=ocpParams, objectParameters=objParams)
    nq = iBee.models[0].nq
    nv = iBee.models[0].nv

    """
    Initial Configuration

    """

    # Hand Ids
    rightHandID = iBee.models[0].getFrameId(right + 'hand')
    leftHandID = iBee.models[1].getFrameId(left + 'hand')

    qi_l = pin.neutral(iBee.models[1])
    qi_r = pin.neutral(iBee.models[0])
    vi_r = np.zeros([nq, ])
    vi_l = vi_r

    pin.forwardKinematics(iBee.models[0], iBee.datas[0], qi_r)
    pin.updateFramePlacements(iBee.models[0], iBee.datas[0])
    Ree_r = iBee.datas[0].oMf[rightHandID].rotation.copy()
    p_r_arm = iBee.datas[0].oMf[rightHandID].translation.copy()

    pin.forwardKinematics(iBee.models[1], iBee.datas[1], qi_l)
    pin.updateFramePlacements(iBee.models[1], iBee.datas[1])
    Ree_l = iBee.datas[1].oMf[leftHandID].rotation.copy()
    p_l_arm = iBee.datas[1].oMf[leftHandID].translation.copy()
    
    """
    Object Parameters (test)

    """
    p_w_o = .5 * (p_r_arm + p_l_arm)
    object_frame = pin.SE3(np.eye(3), p_w_o)

    # SE(3) l_hand -> object c.o.m.
    r_hand_to_obj_H = iBee.datas[0].oMi[6].inverse() * object_frame
    r_hand_to_obj_frame = pin.Frame('r_hand_to_obj',
                                    iBee.models[0].frames[rightHandID].parentJoint,
                                    rightHandID,
                                    r_hand_to_obj_H,
                                    pin.FrameType.OP_FRAME
                                    )
    iBee.models[0].addFrame(r_hand_to_obj_frame)
    rightObjectID = iBee.models[0].getFrameId('r_hand_to_obj')

    # SE(3) l_hand -> object c.o.m.
    l_hand_to_obj_H = iBee.datas[1].oMi[6].inverse() * object_frame
    l_hand_to_obj_frame = pin.Frame('l_hand_to_obj',
                                    iBee.models[1].frames[leftHandID].parentJoint,
                                    leftHandID,
                                    l_hand_to_obj_H,
                                    pin.FrameType.OP_FRAME
                                    )
    iBee.models[1].addFrame(l_hand_to_obj_frame)
    leftObjectID = iBee.models[1].getFrameId('l_hand_to_obj')

    iBee.createDatas()
    pin.forwardKinematics(iBee.models[0], iBee.datas[0], qi_r)
    pin.updateFramePlacements(iBee.models[0], iBee.datas[0])
    pin.forwardKinematics(iBee.models[1], iBee.datas[1], qi_l)
    pin.updateFramePlacements(iBee.models[1], iBee.datas[1])
    iBee.setFrameIDs(rightObjectID, leftObjectID)

    """
    Inverse Kinematics - Joint Reference and Torque Stationarity

    """
    p_r_arm_ref = p_r_arm + np.array([0, 0.05, 0.05])
    p_l_arm_ref = p_l_arm + np.array([0, 0.05, 0.05])

    p_w_obj_ref = iBee.datas[0].oMf[rightObjectID].translation + np.array([0.0, 0.0, 0.01])
    rpy = np.array([0, np.pi/20, 0])
    p_R_obj_ref = iBee.datas[0].oMf[rightObjectID].rotation @ pin.rpy.rpyToMatrix(rpy)
    H_obj_ref = pin.SE3(p_R_obj_ref, p_w_obj_ref)
    print(H_obj_ref.translation)
    target = 'pos'

    qf_r = iBee.inverseKinematics(iBee.models[0], qi_r, H_obj_ref, rightObjectID, target=target)
    print(qf_r)
    xf_r = np.concatenate((qf_r, vi_r), axis=0)
    tau_r_ref = pin.rnea(iBee.models[0], iBee.datas[0], qf_r, vi_r, vi_r)

    qf_l = iBee.inverseKinematics(iBee.models[1], qi_l, H_obj_ref, leftObjectID, target=target)
    print(qf_l)
    xf_l = np.concatenate((qf_l, vi_l), axis=0)
    tau_l_ref = pin.rnea(iBee.models[1], iBee.datas[1], qf_l, vi_l, vi_l)

    pin.forwardKinematics(iBee.models[0], iBee.datas[0], qf_r)
    pin.updateFramePlacements(iBee.models[0], iBee.datas[0])
    print(iBee.datas[0].oMf[rightObjectID])


    pin.forwardKinematics(iBee.models[1], iBee.datas[1], qf_l)
    pin.updateFramePlacements(iBee.models[1], iBee.datas[1])
    print(iBee.datas[1].oMf[leftObjectID])

    """
    Simulation variables
    """

    T = 5
    N = int(T/dt)
    t = np.linspace(0, T, N+1)

    # Joint position and velocity
    x_r = [None] * (N + 1)
    x_l = [None] * (N + 1)
    
    # Joint Positions
    q_r = [None] * (N + 1)
    q_l = [None] * (N + 1)

    # Joint Velocities
    v_r = [None] * (N + 1)
    v_l = [None] * (N + 1)

    # Input Torques
    u_l = [None] * N
    u_r = [None] * N

    # Forces
    f_l = [None] * N
    f_r = [None] * N

    # Spatial twist (w.r.t. WORLD)
    twist_r = [None] * (N + 1)
    twist_l = [None] * (N + 1)

    # Arm vectors
    p_r = np.zeros((3, N + 1))
    p_l = np.zeros((3, N + 1))

    """ Initial state """
    x_r[0] = np.concatenate((qi_r, vi_r), axis=0)
    x_l[0] = np.concatenate((qi_l, vi_l), axis=0)

    p_r[:, 0] = p_r_arm
    p_l[:, 0] = p_l_arm

    q_r[0] = qi_r
    q_l[0] = qi_l

    v_r[0] = vi_r
    v_l[0] = vi_l

    twist_r[0] = vi_r
    twist_l[0] = vi_l

    for i in range(N):
        # Solve OCP
        try:
            u_r[i], u_l[i] = iBee.solve(x_r[i], x_l[i], xf_r, xf_l, tau_r_ref, tau_l_ref)

        except Exception as e:
            raise

        # Update the models
        x_r[i + 1] = np.squeeze(iBee.Fk_Forward_r(x_r[i], u_r[i]))
        x_l[i + 1] = np.squeeze(iBee.Fk_Forward_l(x_l[i], u_l[i]))

        q_r[i + 1] = x_r[i + 1][0:nq].copy()
        q_l[i + 1] = x_l[i + 1][0:nq].copy()

        v_r[i + 1] = x_r[i + 1][nq:].copy()
        v_l[i + 1] = x_l[i + 1][nq:].copy()

        # Update frame velocities and placements
        pin.forwardKinematics(iBee.models[0], iBee.datas[0], x_r[i + 1][:nq], x_r[i + 1][nq:])
        twist_r[i + 1] = pin.getFrameVelocity(iBee.models[0], iBee.datas[0], rightHandID, pin.WORLD)
        pin.updateFramePlacements(iBee.models[0], iBee.datas[0])
        p_r[:, i + 1] = iBee.datas[0].oMf[rightHandID].translation.copy()

        pin.forwardKinematics(iBee.models[1], iBee.datas[1], x_l[i + 1][:nq], x_l[i + 1][nq:])
        twist_l[i + 1] = pin.getFrameVelocity(iBee.models[1], iBee.datas[1], leftHandID, pin.WORLD)
        pin.updateFramePlacements(iBee.models[1], iBee.datas[1])
        p_l[:, i + 1] = iBee.datas[1].oMf[leftHandID].translation.copy()

    try:
        wrench_r, wrench_l = iBee.objForceSolver(objParams) 
    except Exception as e:
        raise

    print(f'right wrench :{wrench_r}')
    print(f'left wrench :{wrench_l}')

    print(f'right q :{q_r[-1]}')
    print(f'left q :{q_l[-1]}')
    print(f'right q_ref :{qf_r}')
    print(f'left q_ref :{qf_l}')
    print(f'right p :{p_r[:, -1]}')
    print(f'left p :{p_l[:, -1]}')
    print(f'right p_ref :{p_r_arm_ref}')
    print(f'left p_ref :{p_l_arm_ref}')
    print(f'Distance in the end {np.linalg.norm(p_r[:, -1] - p_l[:, -1])}')

    # --------------PLOTS-----------
    try:
        import matplotlib.pyplot as plt

        plotTraj({
            'x':q_r,
            'xref':xf_r[:nq],
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Angles [rad]',
            'title': 'Joint Reference Tracking (right)',
            })

        plotTraj({
            'x':q_l,
            'xref':xf_l[:nq],
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Angles [rad]',
            'title': 'Joint Reference Tracking (left)',
            })

        plotTraj({
            'x':v_r,
            'xref':xf_r[nq:],
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Velocities [rad/s]',
            'title': 'Joint Reference Tracking (right)',
            })

        plotTraj({
            'x':v_l,
            'xref':xf_l[nq:],
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Velocities [rad/s]',
            'title': 'Joint Reference Tracking (left)',
            })

        plotTraj({
            'x':u_r,
            'xref':tau_r_ref,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Torques [Nm]',
            'title': 'Torque Reference Tracking (right)',
            })

        plotTraj({
            'x':u_l,
            'xref':tau_l_ref,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Torques [Nm]',
            'title': 'Torque Reference Tracking (left)',
            })

        plotErr({
            'x':np.linalg.norm(p_r - p_l, axis=0),
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Distance [m]',
            'title': 'End-Effector Distance during trajectory'
            })

        color_scheme_r = {
            'line': 'teal',
            'start': 'limegreen',
            'end': 'coral',
            }

        color_scheme_l = {
            'line': 'darkorange',
            'start': 'gold',
            'end': 'crimson',
        }       

        plot3D({
            'x':[p_r, p_l],
            'xlabel': ['Right Arm', 'Left Arm'],
            'title': 'iCub Arms Control',
            'colors': [color_scheme_r, color_scheme_l],
            'frames': None
            })

        plotTraj({
            'x':twist_r,
            'xref':vi_r,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Twist Velocities',
            'title': 'Right End-Effector Velocities'
            })

        plotTraj({
            'x':twist_l,
            'xref':vi_r,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Twist Velocities',
            'title': 'Left End-Effector Velocities'
            })

        plotTraj({
            'x':np.vstack((wrench_r, wrench_l)),
            'xref':vi_r,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Spatial Wrenches (right, left)',
            'title': 'Forces applied at the End-Effectors'
            })

    except ImportError as err:
        print(
            "Error while initializing the viewer. "
            "It seems you should install Python meshcat"
        )
        raise err


if __name__ == "__main__":
    main()
