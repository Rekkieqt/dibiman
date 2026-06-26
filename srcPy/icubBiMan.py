import sys
import os
import casadi as ca
from utils import *
from nmpcArm import NMPC
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
            "joints": ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_prosup', 'wrist_pitch'],
            "arm": 'r_'
            }

    icubCtrler = NMPC(armParameters=armParams, method='inverse', Ts=dt, ocpParameters=ocpParams)

    """
    Control Loop
    """
    model, data = icubCtrler.pinModelandData()
    nv = model.nv
    nq = model.nq
    nu = nv
    q0 = pin.randomConfiguration(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    v0 = np.zeros([nv, ])
    x0 = np.concatenate((q0, v0), axis=0)
    qref = pin.randomConfiguration(model) 
    uref = pin.rnea(model, data, qref, v0, v0)
    xref = np.concatenate((qref, v0), axis=0)

    T = 8
    N = int(T/dt)
    t = np.linspace(0, T, N+1)
    x = [None] * (N + 1)
    q = [None] * (N + 1)
    frames = []
    p = np.zeros((3, N + 1))
    u = [None] * N

    x[0] = x0
    q[0] = x[0][0:nq]
    p[:, 0] = data.oMi[6].translation.copy()
    frames.append(data.oMi[6].copy())

    for i in range(N):
        # Plant simulation Control and warmstart
        u[i] = icubCtrler.solve(x[i], xref, uref)
        # u[i] = np.squeeze(M(x[i], xref, uref))
        x[i + 1] = np.squeeze(icubCtrler.plantModel()(x[i], u[i]))

        # Exctracting the trajectory
        q[i + 1] = x[i + 1][:nq]
        pin.forwardKinematics(model, data, q[i + 1])
        pin.updateFramePlacements(model, data)
        p[:, i + 1] = data.oMi[6].translation.copy()

    # Extracting velocities
    frames.append(data.oMi[6].copy())
    v = [None] * (N + 1)
    v = [xi[-nv:] for xi in x]

    # Printing last result and references
    # print('States, last measured vs ref')
    # print(x[-1])
    # print(xref)
    # print('Torques, last measured vs ref')
    # print(u[-1])
    # print(uref)

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
