import sys
import os
import casadi as ca
from utils import *
from nmpcBiMan import NMPC
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
            'H': 10,
            'Ts': dt
            }

    armParams = {
            "path": os.getcwd() + '/../conf/model.urdf',
            "joints": ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_prosup', 'wrist_pitch']
            }

    objParams = {
            "path": os.getcwd() + '/../conf/testBox.urdf',
            'pl': None,
            'pr': None
            }

    icubBi = NMPC(armParameters=armParams, ocpParameters=ocpParams)

    """
    Initial Configuration
    """
    lz = 0.2;

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
    rightHandID = icubBi.modelR.getFrameId(right + 'hand')
    leftHandID = icubBi.modelL.getFrameId(left + 'hand')

    # Object
    # vector p to object c.o.m. frame
    pObj = (p_l_arm + p_r_arm)/2

    qi_l = pin.randomConfiguration(icubBi.modelL)
    qi_r = pin.randomConfiguration(icubBi.modelR)
    qi_l = icubBi.inverseKinematics(icubBi.modelL, qi_l, Hi_l, left)
    qi_r = icubBi.inverseKinematics(icubBi.modelR, qi_r, Hi_r, right)
    vi_r = np.zeros([icubBi.modelL.nv, ])
    vi_l = vi_r

    pin.forwardKinematics(icubBi.modelR, icubBi.dataR, qi_r)
    pin.updateFramePlacements(icubBi.modelR, icubBi.dataR)
    Ree_r = icubBi.dataR.oMf[rightHandID].rotation.copy()
    print(icubBi.dataR.oMf[rightHandID])
    print(qi_r)
    objParams['pr'] = icubBi.dataR.oMf[rightHandID].translation.copy()

    pin.forwardKinematics(icubBi.modelL, icubBi.dataL, qi_l)
    pin.updateFramePlacements(icubBi.modelL, icubBi.dataL)
    Ree_l = icubBi.dataL.oMf[leftHandID].rotation.copy()
    print(icubBi.dataL.oMf[leftHandID])
    print(qi_l)
    objParams['pl'] = icubBi.dataR.oMf[leftHandID].translation.copy()
    
    # Init Solver
    Ree_l_to_contact = Ree_l.T @ icubBi.R1
    Ree_r_to_contact = Ree_r.T @ icubBi.R2
    icubBi.initSolver(objParams, Ree_l_to_contact, Ree_r_to_contact)

    # Desired object location
    pRefObj = pObj + np.array([0, 0, 0.2])

    # Desired arms location
    p_r_arm_ref = p_r_arm + np.array([0, 0, 0.1])
    p_l_arm_ref = p_l_arm + np.array([0, 0, 0.1])

    Hf_r = pin.SE3(R_r_arm, p_r_arm_ref)
    Hf_l = pin.SE3(R_l_arm, p_l_arm_ref)
    qf_l = icubBi.inverseKinematics(icubBi.modelL, qi_l, Hf_l, left)
    qf_r = icubBi.inverseKinematics(icubBi.modelR, qi_r, Hf_r, right)

    tau_r_ref = pin.rnea(icubBi.modelR, icubBi.dataR, qf_r, vi_r, vi_r)
    tau_l_ref = pin.rnea(icubBi.modelL, icubBi.dataL, qf_l, vi_l, vi_l)

    sys.exit()

    """
    Simulation variables
    """

    T = 1
    N = int(T/dt)
    t = np.linspace(0, T, N+1)

    # Joint position
    q_l = [None] * (N + 1)
    v_l = [None] * (N + 1)

    # Joint velocity
    q_r = [None] * (N + 1)
    v_r = [None] * (N + 1)

    # Input Torques
    u_l = [None] * N
    u_r = [None] * N

    # Forces
    f_l = [None] * N
    f_r = [None] * N

    # Object velocity
    v_obj = [None] * (N + 1)

    # Arm vectors
    p_r = [None] * (N + 1)
    p_l = [None] * (N + 1)

    """ Initial state """
    q_l[0] = qi_l
    q_r[0] = qi_r

    v_l[0] = vi_l
    v_r[0] = vi_r

    v_obj[0] = vi_l

    p_r[0] = p_r_arm
    p_l[0] = p_l_arm

    for i in range(N):

        # Solve OCP
        try:
            # u_r[i], f_r[i], u_l[i], f_l[i] = icubBi.solve(q_r[i], q_l[i], v_r[i], v_l[i], qf_r, qf_l, tau_r_ref, tau_l_ref)
            u_r[i], u_l[i] = icubBi.solve(q_r[i], q_l[i], v_r[i], v_l[i], qf_r, qf_l, tau_r_ref, tau_l_ref)

        except Exception as e:
            print(icubBi.opti.debug.value)
            raise

        # Update the models
        q_r[i + 1], v_r[i + 1] = icubBi.forwardDynamics(icubBi.modelR, Ree_r_to_contact)(q_r[i], v_r[i], u_r[i])
        q_l[i + 1], v_l[i + 1] = icubBi.forwardDynamics(icubBi.modelL, Ree_l_to_contact)(q_l[i], v_l[i], u_l[i])
        q_r[i + 1] = np.squeeze(q_r[i + 1])
        q_l[i + 1] = np.squeeze(q_l[i + 1])
        v_r[i + 1] = np.squeeze(v_r[i + 1])
        v_l[i + 1] = np.squeeze(v_l[i + 1])

        # Update placements
        pin.forwardKinematics(icubBi.modelR, icubBi.dataR, q_r[i + 1].copy())
        pin.updateFramePlacements(icubBi.modelR, icubBi.dataR)
        p_r[i + 1] = icubBi.dataR.oMf[rightHandID].translation.copy()

        pin.forwardKinematics(icubBi.modelL, icubBi.dataL, q_l[i + 1].copy())
        pin.updateFramePlacements(icubBi.modelL, icubBi.dataL)
        p_l[i + 1] = icubBi.dataL.oMf[leftHandID].translation.copy()


    # --------------PLOTS-----------
    try:
        import matplotlib.pyplot as plt

        plotTraj({
            'x':q_r,
            'xref':qf_r,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Angles [rad]',
            'title': 'Joint Reference Tracking (right)',
            })

        plotTraj({
            'x':q_l,
            'xref':qf_l,
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

        plotTraj({
            'x':v_r,
            'xref':vi_r,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Velocities [rad/s]',
            'title': 'Joint Velocities (right)',
            })

        plotTraj({
            'x':v_l,
            'xref':vi_l,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Velocities [rad/s]',
            'title': 'Joint Velocities (left)',
            })

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
        #     })

        plot3D({
            'x':[p_r, p_l],
            'xlabel': ['Right Arm', 'Left Arm'],
            'title': 'iCub Arms Control',
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
