import sys
import os

import casadi as ca
from utils import *
from pathlib import Path
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin


def getBoxModel(path):
    """
    Get Model from urdf
    """
    model = pin.buildModelFromUrdf(path, pin.JointModelFreeFlyer())
    # model_path = Path(os.environ.get("pinRobotDir"))
    # urdf_filename = model_path / "hector_description/robots/quadrotor_base.urdf"
    # model = pin.buildModelFromUrdf(urdf_filename, pin.JointModelFreeFlyer())
    data = model.createData()
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    return model, data, cmodel, cdata


def actuation_model(nu):
    u1 = ca.SX.sym("u1", nu)
    u2 = ca.SX.sym("u2", nu)
    u = ca.vertcat(u1, u2)
    tau = u1 + u2

    return ca.Function("act_model", [u], [tau], ["u"], ["tau"])


def state_integrate(model):
    q = ca.SX.sym("dq", model.nq)
    dq = ca.SX.sym("q", model.nv)
    v = ca.SX.sym("v", model.nv)
    dv = ca.SX.sym("dv", model.nv)

    q_next = cpin.integrate(model, q, dq)
    v_next = v + dv

    dx = ca.vertcat(dq, dv)
    x = ca.vertcat(q, v)
    x_next = ca.vertcat(q_next, v_next)

    return ca.Function("integrate", [x, dx], [x_next], ["x", "dx"], ["x_next"])


def state_difference(model):
    q0 = ca.SX.sym("q0", model.nq)
    q1 = ca.SX.sym("q1", model.nq)
    v0 = ca.SX.sym("v0", model.nv)
    v1 = ca.SX.sym("v1", model.nv)

    q_diff = cpin.difference(model, q0, q1) # of dimension 6, because quaternions only have 3 dof
    v_diff = v1 - v0

    x0 = ca.vertcat(q0, v0)
    x1 = ca.vertcat(q1, v1)
    x_diff = ca.vertcat(q_diff, v_diff) # dimension 6

    return ca.Function("difference", [x0, x1], [x_diff], ["x0", "x1"], ["x_diff"])


def euler_integration(model, data, dt=0.05):
    nu = model.nv
    u1 = ca.SX.sym('u1', nu)
    u2 = ca.SX.sym('u2', nu)
    u = ca.vertcat(u1, u2)

    # tau = actuation_model(nu)(u)
    tau = ca.SX.sym('u', model.nv)

    q = ca.SX.sym("q", model.nq)
    v = ca.SX.sym("v", model.nv)

    a = cpin.aba(model, data, q, v, tau)

    dq = v * dt + a * dt**2
    dv = a * dt

    x = ca.vertcat(q, v)
    dx = ca.vertcat(dq, dv)
    x_next = state_integrate(model)(x, dx)

    return ca.Function("int_dyn", [x, tau], [x_next], ["x", "u1"], ["x_next"])


def dynSolver(F, model, data, solOpts):
    opti = ca.Opti()
    r = solOpts['r'] # input cost
    q = solOpts['q'] # state cost
    H = solOpts['H'] # horizon length

    nx = F.size_in(0)[0] - 1
    nxF = F.size_in(0)[0]
    nu = F.size_in(1)[0]
    # np = F.size_in(2)[0]

    R = r * ca.DM.eye(nu)
    Q = q * ca.DM.eye(nx)

    x0 = opti.parameter(nxF, )
    xrf = opti.parameter(nxF, )
    xnom = opti.parameter(nxF, )
    # p = opti.parameter(np, )

    X = []
    U = []
    for k in range(H):
        X.append(opti.variable(nx))
        U.append(opti.variable(nu))
    X.append(opti.variable(nx))

    # Objective function
    obj = 0

    # Initial state
    x_0 = state_integrate(model)(xnom, X[0])
    opti.subject_to(state_difference(model)(x0, x_0) == [0] * nx)

    # State & Control regularization
    for k in range(H):
        x_i = state_integrate(model)(xnom, X[k])
        e_reg = state_difference(model)(xnom, x_i)
        obj += ca.mtimes([e_reg.T, Q, e_reg])
        obj += ca.mtimes([U[k].T, R, U[k]])

    opti.minimize(obj)

    # Dynamical constraints
    for k in range(H):
        x_i = state_integrate(model)(xnom, X[k])
        f_x_u = euler_integration(model, data)(x_i, U[k])
        x_i_1 = state_integrate(model)(xnom, X[k + 1])
        gap = state_difference(model)(f_x_u, x_i_1)
        opti.subject_to(gap == [0] * nx)

    # Solver options
    opts = {
            'print_time': 0,
            'expand': True,
            'debug': True,
            # 'structure_detection': 'auto',
            'fatrop.print_level': 0
            # 'jit': True,
            # 'jit_options': {'flags': '-O3', 'verbose': True},
            # 'fatrop.tolerance': 1e-3,
            # 'fatrop.max_iter': 200
            # 'ipopt.hessian_approximation': 'limited-memory',
            # 'ipopt.print_level': 0,
            # 'ipopt.tol': 1e-3
            }

    opti.solver('fatrop', opts)

    M = opti.to_function('NMPC', [x0, xrf, xnom], [U[0]])

    return M


def main():
    objPath = os.getcwd() + '/../conf/testBox.urdf'
    model, data, cmodel, cdata = getBoxModel(objPath)

    dt = 0.1
    H = 2
    nq = cmodel.nq
    nv = cmodel.nv
    F = euler_integration(cmodel, cdata, dt)
    nx = F.size_in(0)[0]
    M = dynSolver(F, cmodel, cdata, {
        'H': H,
        'r': 1,
        'q': 3
        })

    """
    Control Loop
    """
    # initial state
    Hi = pin.SE3.Random()
    Hivec = pin.SE3ToXYZQUAT(Hi)
    v0 = np.zeros([nv, ])
    x0 = np.concatenate((Hivec, v0), axis=0)

    # reference state
    Hf = pin.SE3.Random()
    Hfvec = pin.SE3ToXYZQUAT(Hf)
    vf = v0
    xf = np.concatenate((Hfvec, vf), axis=0)

    # xnom
    xnom = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    T = 8
    Ts = dt
    N = int(T/Ts)
    t = np.linspace(0, T, N+1)
    x = [None] * (N + 1)
    p = np.zeros((3, N+1))
    u = [None] * N
    x[0] = x0
    p[:, 0] = Hi.translation

    # Homogenous matrices
    frames = []
    frames.append(np.array(Hi))

    for i in range(N):
        u[i] = np.squeeze(M(x[i], xf, xnom))
        x[i + 1] = np.squeeze(F(x[i], u[i]))
        p[:, i + 1] = pin.XYZQUATToSE3(x[i + 1][0:nq]).translation
        # Xpast, Upast = Xsolved.copy(), Usolved.copy()

    # Printing last result and references
    print('States, last measured vs ref')
    print(x[-1])
    print(xf)
    qf = x[-1][0:nq]
    print(pin.difference(model, qf, Hfvec))
    print('Controls, force and torque')
    print(u[-1])
    frames.append(np.array(pin.XYZQUATToSE3(x[-1][0:nq])))

    # --------------PLOTS-----------
    try:
        import matplotlib.pyplot as plt
        
        plotData = {
                'x': x,
                'xref': xf,
                't': t,
                'xlabel': 'time [s]',
                'ylabel': 'Translation and Rotation of the Object Frame',
                'title': 'Rigid Body OCP for Forces and Torques Applied'
                }

        plotTraj(plotData)
        plot3D({
            'title': 'Box Trajectory',
            'xlabel': 'Pos',
            'x': p,
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
