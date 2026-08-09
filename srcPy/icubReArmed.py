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
    
    # Desired arms location
    p_r_arm_ref = p_r_arm + np.array([0, 0, 0.01])
    p_l_arm_ref = p_l_arm + np.array([0, 0, 0.01])

    Hf_r = pin.SE3(Ree_r, p_r_arm_ref)
    qf_r = iBee.inverseKinematics(iBee.models[0], qi_r, p_r_arm_ref, right, target='pos')
    xf_r = np.concatenate((qf_r, vi_r), axis=0)
    print(qf_r)
    tau_r_ref = pin.rnea(iBee.models[0], iBee.datas[0], qf_r, vi_r, vi_r)

    Hf_l = pin.SE3(Ree_l, p_l_arm_ref)
    qf_l = iBee.inverseKinematics(iBee.models[1], qi_l, p_l_arm_ref, left, target='pos')
    xf_l = np.concatenate((qf_l, vi_l), axis=0)
    print(qf_l)
    tau_l_ref = pin.rnea(iBee.models[1], iBee.datas[1], qf_l, vi_l, vi_l)

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
    twist_r = np.zeros((6, N + 1))
    twist_l = np.zeros((6, N + 1))

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

    for i in range(N):

        # Solve OCP
        try:
            # u_r[i], f_r[i], u_l[i], f_l[i] = iBee.solve(q_r[i], q_l[i], v_r[i], v_l[i], qf_r, qf_l, tau_r_ref, tau_l_ref)
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

        # Update placements
        pin.forwardKinematics(iBee.models[0], iBee.datas[0], x_r[i + 1][:nq], x_r[i + 1][nq:])
        twist_r[:, i + 1] = pin.getFrameVelocity(iBee.models[0], iBee.datas[0], rightHandID, pin.WORLD)
        pin.updateFramePlacements(iBee.models[0], iBee.datas[0])
        p_r[:, i + 1] = iBee.datas[0].oMf[rightHandID].translation.copy()

        pin.forwardKinematics(iBee.models[1], iBee.datas[1], x_l[i + 1][:nq], x_l[i + 1][nq:])
        twist_l[:, i + 1] = pin.getFrameVelocity(iBee.models[1], iBee.datas[1], leftHandID, pin.WORLD)
        pin.updateFramePlacements(iBee.models[1], iBee.datas[1])
        p_l[:, i + 1] = iBee.datas[1].oMf[leftHandID].translation.copy()


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

        plotErr({
            'x':np.linalg.norm(twist_r - twist_l, axis=0),
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Euclidian l2 norm of the difference of spatial twists left and right',
            'title': 'End-Effector Difference between spatial twists'
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

        plot3D({
            'x':[twist_r, twist_l],
            'xlabel': ['Right Arm Twist', 'Left Arm Twist'],
            'title': 'iCub Arms Twists',
            'colors': [color_scheme_r, color_scheme_l],
            'frames': None
            })

    except ImportError as err:
        print(
            "Error while initializing the viewer. "
            "It seems you should install Python meshcat"
        )
        raise err


if __name__ == "__main__":
    main()
