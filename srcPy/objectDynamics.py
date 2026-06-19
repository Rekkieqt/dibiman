import sys
import os

import casadi as ca
from utils import *
import hppfcl as fcl
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin


def getObjModel(params):
    """
    Get Model from urdf
    """
    modelpath = params['path']

    model = pin.buildModelFromUrdf(modelpath, pin.JointModelFreeFlyer())

    data = model.createData()
    '''
    print(list(model.names))
    universe
    root_joint
    '''

    return model, data


def getCasFunc(obj, objData, intOpts):
    '''
    Hard coded mass and Inertia >>>
    InMat = intOpts['InertiaMat']
    In = np.array([
        [InMat[0], 0, 0],
        [0, InMat[1], 0],
        [0, 0, InMat[2]]
        ])

    m = intOpts['mass'] * np.eye(nn)
    '''
    # dimensions
    nn = 3
    nu = 3

    properties = obj.inertias[1]
    In = properties.inertia
    m = properties.mass * ca.DM.eye(nn)
    g = np.array([[0], [0], [-9.8]])

    # Cross Product Matrix format
    p = ca.SX.sym('p', nn)
    pX = ca.vertcat(
            ca.horzcat(0, -p[2], p[1]),
            ca.horzcat(p[2], 0, -p[0]),
            ca.horzcat(-p[1], p[0], 0)
        )
    xProd = ca.Function('xProduct', [p], [pX], ['inVec'], ['outMat'])

    p1 = ca.SX.sym('p1', nn)
    p2 = ca.SX.sym('p2', nn)

    """
    Declaring the acting inputs, f1, f2 and n1 and n2
    """
    f1 = ca.SX.sym("f1", nu)
    n1 = ca.SX.sym("n1", nu)
    f2 = ca.SX.sym("f2", nu)
    n2 = ca.SX.sym("n2", nu)

    """
    State variables, r = (r, rd) and theta = (theta, omega)
    """
    r = ca.SX.sym("r", nn)
    dr = ca.SX.sym("dr", nn)
    theta = ca.SX.sym("theta", nn)
    omega = ca.SX.sym("omega", nn)

    """
    Get the Newton-Euler dynamic equations
    """
    dt = intOpts['tf']

    # Rotation Dynamics
    alpha = ca.inv(In) @ (n1 + n2 + ca.cross(p1, f1) + ca.cross(p2, f2) - xProd(omega) @ (In @ omega))
    dphi = ca.vertcat(omega, alpha)
    phi = ca.vertcat(theta, omega)
    u = ca.vertcat(n1, f1, n2, f2)
    # frot = ca.Function('alpha', [omega, u, p1, p2], [da], ['omega', 'u', 'p1', 'p2'], ['omega'])

    # Translation Dynamics
    ddr = ca.inv(m) @ ((f1 + f2) + g)
    dx = ca.vertcat(dr, ddr)
    x = ca.vertcat(r, dr)
    # fpos = ca.Function('ddr', [x, u], [dx], ['x', 'u'], ['dx'])

    """
    Explicit Euler integration
    """

    omegak = omega + alpha * dt
    thetak = cpin.integrate(obj, theta, omegak * dt)
    phik = ca.vertcat(thetak, omegak)

    xk = x + ddr * dt
    miuk = ca.vertcat(xk, phik)
    miu0 = ca.vertcat(x, phi)
    Fk = ca.Function('dyn', [miu0, u, p1, p2], [miuk], ['miu0', 'u', 'p1', 'p2'], ['miu1'])

    return FkRot, FkX


def getNLP(F, solOpts):
    opti = ca.Opti()
    H = solOpts['H']
    nx = solOpts['nx']
    nu = solOpts['nu']
    r = solOpts['r']
    q = solOpts['q']

    R = r * np.eye(nu)
    Q = q * np.eye(nx)

    x = opti.variable(nx, H + 1)
    u = opti.variable(nu, H)
    x0 = opti.parameter(nx, )
    xrf = opti.parameter(nx, )
    urf = opti.parameter(nu, )

    obj = 0
    for i in range(H):
        obj += ca.mtimes([(x[:, i + 1] - xrf).T, Q, x[:, i + 1] - xrf])
        obj += ca.mtimes([(u[:, i] - urf).T, R, u[:, i] - urf])

    opti.minimize(obj)

    for k in range(H):
      opti.subject_to(x[:, k + 1]==F(x[:, k], u[:, k]))

    opti.subject_to(x[:, 0]==x0)

    opti
    opts = {
            'print_time': 0,
            'ipopt.print_level': 0
            }

# solver options
    opti.solver('ipopt', opts)

    M = opti.to_function('NMPC', [x0, xrf, urf], [u[:, 0]])

    return M


def main():
    objPath = os.getcwd() + '/../conf/testBox.urdf'
    # model = pin.buildModelFromUrdf(model_path)
    H = 20
    nx = 12
    nu = 12
    nn = 3
    pinOpts = {
            "path": objPath
            }

    model, data = getObjModel(pinOpts)
    intOpts = {
            'mass': 2,
            'InertiaMat': [.533, .533, .533],
            't0': 0,
            'tf': 0.1,
            'H': H
            }

    F, xNext = getCasFunc(model, data, intOpts)
    sys.exit(1)
    solOpts = {
            'H': H,
            'r': 1,
            'q': 3,
            'nx': nx,
            'nu': nu
            }

    M = getNLP(F, solOpts)

    """
    Control Loop
    """
    # initial state
    r0 = np.zeros([nn, ])
    v0 = np.zeros([nn, ])
    th0 = np.zeros([nn, ])
    w0 = np.zeros([nn, ])
    x0 = np.concatenate((r0, v0, th0, w0), axis=0)

    # reference state
    rf = np.ones([nn, ])
    vf = v0
    thf = np.pi/3 * np.ones([nn, ])
    wf = w0
    xf = np.concatenate((rf, vf, thf, wf), axis=0)
    uf = np.zeros([nu, ])

    T = 15
    Ts = intOpts['tf']
    N = int(T/Ts)
    t = np.linspace(0, T, N+1)
    x = [None] * (N + 1)
    u = [None] * N
    x[0] = x0

    for i in range(N):
        u[i] = np.squeeze(M(x[i], xf, uf))
        x[i + 1] = np.squeeze(F(x[i], u[i]))


    # --------------PLOTS-----------
    try:
        import matplotlib.pyplot as plt
        
        plotData = {
                'x': x,
                'xref': xf,
                't': t
                'xlabel': 'time [s]',
                'ylabel': 'Translation and Rotation of the Object Frame',
                'title': 'Rigid Body OCP for Forces and Torques Applied'
                }

        plotTraj(plotData)

    except ImportError as err:
        print(
            "Error while initializing the viewer. "
            "It seems you should install Python meshcat"
        )
        raise err


if __name__ == "__main__":
    main()
