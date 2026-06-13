import sys
import os

import casadi
from casadi import *
from plotTraj import *
import hppfcl as fcl
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin


def getObjModel(params):
    """
    Get Model from sdf
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
    print(inertia.mass)
    print(inertia.lever)
    print(inertia.inertia)

    # dimensions
    nn = 3
    nu = 3

    properties = obj.inertias[1]
    In = properties.inertia
    # p0 = properties.lever
    m = properties.mass * np.eye(nn)

    g = np.array([[0], [0], [-9.8]])

    p1 = np.array([1, 1, 1])
    P1 = np.array([
        [0, -p1[2], p1[1]],
        [p1[2], 0, -p1[0]],
        [-p1[1], p1[0], 0]
        ])

    p2 = np.array([1, -1, 1])
    P2 = np.array([
        [0, -p2[2], p2[1]],
        [p2[2], 0, -p2[0]],
        [-p2[1], p2[0], 0]
        ])

    """
    Declaring the acting inputs, f1, f2 and n1 and n2
    """
    f1 = casadi.SX.sym("f1", nu)
    n1 = casadi.SX.sym("n1", nu)
    f2 = casadi.SX.sym("f2", nu)
    n2 = casadi.SX.sym("n2", nu)

    """
    State variables, r = (r, rd) and theta = (theta, omega)
    """
    r = casadi.SX.sym("r", nn)
    rd = casadi.SX.sym("rd", nn)
    theta = casadi.SX.sym("theta", nn)
    omega = casadi.SX.sym("omega", nn)

    omegaX = casadi.vertcat(
        casadi.horzcat(0, -omega[2], omega[1]),
        casadi.horzcat(omega[2], 0, -omega[0]),
        casadi.horzcat(-omega[1], omega[0], 0)
        )

    
    """
    Get the Newton-Euler dynamics ODE
    """
    rdd = casadi.inv(m) @ (f1 + f2) + g
    # rdd = [r, rdd]
    alpha = casadi.inv(In) @ (n1 + n2 + P1 @ f1 + P2 @ f2 - omegaX @ (In @ omega))

    # thetadd = [theta, thetadd]
    u = casadi.vertcat(f1, f2, n1, n2)
    transX = casadi.vertcat(rd, rdd)
    rotW = casadi.vertcat(omega, alpha)
    objODE = casadi.vertcat(transX, rotW)
    rotS = casadi.vertcat(theta, omega)
    transS = casadi.vertcat(r, rd)
    x = casadi.vertcat(transS, rotS)

    fObj = casadi.Function('fObj', [x, u], [objODE], ['x', 'u'], ['objAcc'])
    print(fObj)

    """
    Integrate the Object dynamics
    """
    tf = intOpts['tf']
    t0 = intOpts['t0']
    H = intOpts['H']

    dae = {'x': x,
           'p': u,
           'ode': objODE}

    opts = {
            'simplify': True,
            'number_of_finite_elements': 4
            }

    intg = casadi.integrator('rk4', 'rk', dae, t0, tf, opts)
    intRes = intg(x0=x, p=u)
    x_next = intRes['xf']
    F = casadi.Function('objRint', [x, u], [x_next], ['x', 'u'], ['x_plus'])
    print(F)
    bN = F.mapaccum(H)

    return F, bN


def getNLP(F, solOpts):
    opti = casadi.Opti()
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
        obj += mtimes([(x[:, i + 1] - xrf).T, Q, x[:, i + 1] - xrf])
        obj += mtimes([(u[:, i] - urf).T, R, u[:, i] - urf])

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
                'N': N,
                'xref': xf,
                't': t
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
