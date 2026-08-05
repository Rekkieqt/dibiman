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
            'pl': None,
            'pr': None
            }

    iBee = armNMPC(armParameters=armParams, Ts=dt, ocpParameters=ocpParams)
    nq = iBee.models[0].nq
    nv = iBee.models[0].nv

    """
    Initial Configuration
    """

    # Arms initial
    p_r_arm = np.array([-0.24295, 0.189132, 0.0184444])
    p_l_arm = np.array([-0.243011, -0.189058, 0.0187189])
    R_r_arm = np.array([
        [0.862048, 0.21001, -0.46127],
        [-0.316008, 0.93426, -0.165219],
        [0.396248, 0.288192, 0.871741]
    ])
    R_l_arm = np.array([
        [-0.862038, 0.209971, -0.461305],
        [-0.316018, -0.934241, 0.165305],
        [-0.396261, 0.28828, 0.871706]
    ])
    Hi_r = pin.SE3(R_r_arm, p_r_arm)
    Hi_l = pin.SE3(R_l_arm, p_l_arm)

    # Hand Ids
    rightHandID = iBee.models[0].getFrameId(right + 'hand')
    leftHandID = iBee.models[1].getFrameId(left + 'hand')

    # Object
    # vector p to object c.o.m. frame
    pObj = (p_l_arm + p_r_arm)/2

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
    
    # Init Solver
    # Ree_l_to_contact = Ree_l.T @ iBee.R1
    # Ree_r_to_contact = Ree_r.T @ iBee.R2
    # iBee.initSolver(objParams, Ree_l_to_contact, Ree_r_to_contact)

    # Desired object location
    pRefObj = pObj + np.array([0, 0, 0.05])

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

    # Input Torques
    u_l = [None] * N
    u_r = [None] * N

    # Forces
    f_l = [None] * N
    f_r = [None] * N

    # Object velocity
    v_obj = [None] * (N + 1)

    # Arm vectors
    p_r = np.zeros((3, N + 1))
    p_l = np.zeros((3, N + 1))

    """ Initial state """
    x_r[0] = np.concatenate((qi_r, vi_r), axis=0)
    x_l[0] = np.concatenate((qi_l, vi_l), axis=0)

    v_obj[0] = vi_l

    p_r[:, 0] = p_r_arm
    p_l[:, 0] = p_l_arm
    

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

        # Update placements
        pin.forwardKinematics(iBee.models[0], iBee.datas[0], x_r[i + 1][:nq])
        pin.updateFramePlacements(iBee.models[0], iBee.datas[0])
        p_r[:, i + 1] = iBee.datas[0].oMf[rightHandID].translation.copy()

        pin.forwardKinematics(iBee.models[1], iBee.datas[1], x_l[i + 1][:nq])
        pin.updateFramePlacements(iBee.models[1], iBee.datas[1])
        p_l[:, i + 1] = iBee.datas[1].oMf[leftHandID].translation.copy()


    extract_Data = {
            'p_r' : p_r,
            'x_r' : x_r,
            'u_r' : u_r
            }
    savemat('output.mat', extract_Data)
    # --------------PLOTS-----------
    try:
        import matplotlib.pyplot as plt

        plotTraj({
            'x':x_r,
            'xref':xf_r,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Angles [rad]',
            'title': 'Joint Reference Tracking (right)',
            })

        plotTraj({
            'x':x_l,
            'xref':xf_l,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Angles [rad]',
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

        # plotTraj({
        #     'x':v_r,
        #     'xref':vi_r,
        #     't':t,
        #     'xlabel': 'Time [s]',
        #     'ylabel': 'Joint Velocities [rad/s]',
        #     'title': 'Joint Velocities (right)',
        #     })

        # plotTraj({
        #     'x':v_l,
        #     'xref':vi_l,
        #     't':t,
        #     'xlabel': 'Time [s]',
        #     'ylabel': 'Joint Velocities [rad/s]',
        #     'title': 'Joint Velocities (left)',
        #     })

        # plotTraj({
        #     'x':f_r,
        #     'xref':vi_r,
        #     't':t,
        #     'xlabel': 'Time [s]',
        #     'ylabel': 'Right Contact Wrench',
        #     'title': 'Contact Forces applied (right)',
        #     })

        # plotTraj({
        #     'x':f_l,
        #     'xref':vi_l,
        #     't':t,
        #     'xlabel': 'Time [s]',
        #     'ylabel': 'Left Contact Wrench',
        #     'title': 'Contact Forces applied (left)',

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

    except ImportError as err:
        print(
            "Error while initializing the viewer. "
            "It seems you should install Python meshcat"
        )
        raise err


if __name__ == "__main__":
    main()
