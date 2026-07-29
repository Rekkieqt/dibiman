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
            'H': 50,
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
    # Object
    # vector p to object c.o.m. frame
    pObj = np.array([-.2, 0, .1])
    lz = 0.2;

    # Arms initial
    p_r_arm = pObj + np.array([0, lz/2, 0])
    p_l_arm = pObj + np.array([0, -lz/2, 0])
    rpy_r = np.array([0, 0, np.pi/2])
    rpy_l = rpy_r
    Hi_r = pin.SE3(pin.rpy.rpyToMatrix(rpy_r), p_r_arm)
    Hi_l = pin.SE3(pin.rpy.rpyToMatrix(rpy_l), p_l_arm)

    qi_l = pin.randomConfiguration(icubBi.modelL)
    qi_r = pin.randomConfiguration(icubBi.modelR)
    qi_l = icubBi.inverseKinematics(icubBi.modelL, qi_l, Hi_l, left)
    qi_r = icubBi.inverseKinematics(icubBi.modelR, qi_r, Hi_r, right)
    vi_r = np.zeros([icubBi.modelL.nv, ])
    vi_l = vi_r

    pin.forwardKinematics(icubBi.modelR, icubBi.dataR, qi_r)
    pin.updateFramePlacements(icubBi.modelR, icubBi.dataR)
    Ree_r = icubBi.dataR.oMi[6].rotation.copy()
    print(icubBi.dataR.oMi[6])
    objParams['pr'] = icubBi.dataR.oMi[6].translation.copy()

    pin.forwardKinematics(icubBi.modelL, icubBi.dataL, qi_l)
    pin.updateFramePlacements(icubBi.modelL, icubBi.dataL)
    Ree_l = icubBi.dataL.oMi[6].rotation.copy()
    print(icubBi.dataL.oMi[6])
    objParams['pl'] = icubBi.dataR.oMi[6].translation.copy()

    # Init Solver
    Ree_l_to_contact = Ree_l.T @ icubBi.R1
    Ree_r_to_contact = Ree_r.T @ icubBi.R2
    icubBi.initSolver(objParams, Ree_l_to_contact, Ree_r_to_contact)

    # Desired object location
    pRefObj = pObj + np.array([-0.05, 0.05, 0.1])

    # Desired arms location
    p_r_arm_ref = pRefObj + np.array([0, lz/2, 0])
    p_l_arm_ref = pRefObj + np.array([0, -lz/2, 0])

    Hf_r = pin.SE3(pin.rpy.rpyToMatrix(rpy_r), p_r_arm_ref)
    Hf_l = pin.SE3(pin.rpy.rpyToMatrix(rpy_r), p_l_arm_ref)
    qf_l = icubBi.inverseKinematics(icubBi.modelL, qi_l, Hf_l, left)
    qf_r = icubBi.inverseKinematics(icubBi.modelR, qi_r, Hf_r, right)

    tau_r_ref = pin.rnea(icubBi.modelR, icubBi.dataR, qf_l, vi_r, vi_r)
    tau_l_ref = pin.rnea(icubBi.modelL, icubBi.dataL, qf_l, vi_l, vi_l)

    """
    Simulation variables
    """
    sys.exit()

    T = 8
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

    """ Initial state """
    q_l[0] = qi_l
    q_r[0] = qi_r

    v_l[0] = vi_l
    v_r[0] = vi_r

    for i in range(N):
        # Solve OCP
        u_r[i], u_l[i] = icubBi.solve(q_r[i], q_l[i], v_r[i], v_l[i], qf_r, qf_l, tau_r_ref, tau_l_ref)

        # Update the models
        q_r[i + 1], v_r[i + 1] = icubBi


    # --------------PLOTS-----------
    try:
        import matplotlib.pyplot as plt
        plotTraj({
            'x':q,
            'xref':qref,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Angles [rad]',
            'title': 'Joint Reference Tracking',
            })

        plotTraj({
            'x':u,
            'xref':uref,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Torques [Nm]',
            'title': 'Torque Reference Tracking',
            })

        plotTraj({
            'x':v,
            'xref':v0,
            't':t,
            'xlabel': 'Time [s]',
            'ylabel': 'Joint Velocities [rad/s]',
            'title': 'Joint Velocities',
            })

        plot3D({
            'x':p,
            'xlabel': 'End-Effector Trajectory',
            'title': 'iCub Arm Control',
            'frames': frames
            })

    except ImportError as err:
        print(
            "Error while initializing the viewer. "
            "It seems you should install Python meshcat"
        )
        raise err


if __name__ == "__main__":
    main()
