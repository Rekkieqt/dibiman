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
    # model = pin.buildModelFromUrdf(path, pin.JointModelFreeFlyer())
    model_path = Path(os.environ.get("pinRobotDir"))
    urdf_filename = model_path / "hector_description/robots/quadrotor_base.urdf"
    model = pin.buildModelFromUrdf(urdf_filename, pin.JointModelFreeFlyer())

    data = model.createData()

    # model = pin.Model()
    # # Default gravity is already (0, 0, -9.81, 0, 0, 0); shown here for clarity.

    # box_size = 1.0  # edge length of each cube [m]
    # box_half = box_size / 2.0
    # box_mass = 1.0  # mass of each cube [kg]

    # box_inertia = pin.Inertia.FromBox(box_mass, box_size, box_size, box_size)

    # # Cube 1 - free-flyer joint branching from the universe (joint 0)
    # joint_id = model.addJoint(
    #     0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "box"
    # )
    # model.appendBodyToJoint(joint_id, box_inertia, pin.SE3.Identity())
    # box_frame = pin.Frame("box_frame", joint_id, 0, pin.SE3.Identity(), pin.FrameType.BODY)
    # model.addFrame(box_frame)

    # data = model.createData()
    # print(list(model.names))

    return model, data

def actuation_model():
    u1 = ca.SX.sym("u1", 6)
    u2 = ca.SX.sym("u2", 6)
    tau = u1 + u2

    return ca.Function("act_model", [u1, u2], [tau], ["f1/n1", "f2/n2"], ["tau"])


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

    q_diff = cpin.difference(model, q0, q1)
    v_diff = v1 - v0

    x0 = ca.vertcat(q0, v0)
    x1 = ca.vertcat(q1, v1)
    x_diff = ca.vertcat(q_diff, v_diff)

    return ca.Function("difference", [x0, x1], [x_diff], ["x0", "x1"], ["x_diff"])


def euler_integration(model, data, dt):
    nu = 6
    u1 = ca.SX.sym('u1', nu)
    u2 = ca.SX.sym('u2', nu)

    # tau = actuation_model()(u1, u2)
    tau = ca.SX.sym('u', model.nv)

    q = ca.SX.sym("q", model.nq)
    v = ca.SX.sym("v", model.nv)

    a = cpin.aba(model, data, q, v, tau)

    dq = v * dt + a * dt**2
    dv = a * dt

    x = ca.vertcat(q, v)
    dx = ca.vertcat(dq, dv)
    x_next = state_integrate(model)(x, dx)

    return ca.Function("int_dyn", [x, u], [x_next], ["x", "u"], ["x_next"])


def getCasFunc(obj, objData, intOpts):
    # dimensions
    cobj = cpin.Model(obj)
    nq = cobj.nq
    nv = cobj.nv
    nu = 3

    properties = obj.inertias[1]
    In = properties.inertia
    m = properties.mass * ca.DM.eye(nu)
    g = np.array([[0], [0], [-9.8]])

    # Cross Product Matrix format
    p = ca.SX.sym('p', nu)
    pX = ca.vertcat(
            ca.horzcat(0, -p[2], p[1]),
            ca.horzcat(p[2], 0, -p[0]),
            ca.horzcat(-p[1], p[0], 0)
        )
    xProd = ca.Function('xProduct', [p], [pX], ['inVec'], ['outMat'])

    p1 = ca.SX.sym('p1', nu)
    p2 = ca.SX.sym('p2', nu)

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
    r = ca.SX.sym("r", nu)
    dr = ca.SX.sym("dr", nu)
    theta = ca.SX.sym("theta", nq)
    omega = ca.SX.sym("omega", nu)

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
    thetak = cpin.integrate(theta, omegak * dt)
    phik = ca.vertcat(thetak, omegak)

    xk = x + ddr * dt
    miuk = ca.vertcat(xk, phik)
    miu0 = ca.vertcat(x, phi)
    Fk = ca.Function('dyn', [miu0, u, p1, p2], [miuk], ['miu0', 'u', 'p1', 'p2'], ['miu1'])

    return FkRot, FkX


def dynSolver(F, model, solOpts):
    opti = ca.Opti()
    r = solOpts['r'] # input cost
    q = solOpts['q'] # state cost
    H = solOpts['H'] # horizon length

    nq = F.size_in(0)[0]
    nu = F.size_in(1)[0]

    R = r * ca.DM.eye(nu)
    Q = q * ca.DM.eye(nq)

    x0 = opti.parameter(nq, )
    xrf = opti.parameter(nq, )
    urf = opti.parameter(nu, )

    X = []
    U = []
    for k in range(H):
        X.append(opti.variable(nq))
        U.append(opti.variable(nu))
    X.append(opti.variable(nq))

    # Cost function
    obj = 0
    for i in range(H):
        obj += ca.mtimes([(X[i + 1] - xrf).T, Q, X[i + 1] - xrf])
        obj += ca.mtimes([(U[i] - urf).T, R, U[i] - urf])

    opti.minimize(obj)

    # Subject to the model/ discrete function
    opti.subject_to(state_difference(model)(X[0], x0) == [0] * nx)
    for k in range(H):
        opti.subject_to(X[k + 1] == F(X[k], U[k]))

    opts = {
            'print_time': 0,
            'expand': True,
            'jit': True,
            'structure_detection': 'auto',
            'fatrop.print_level': 0
            # 'jit_options': {'flags': '-O3', 'verbose': True},
            # 'fatrop.tolerance': 1e-3,
            # 'fatrop.max_iter': 200
            # 'ipopt.hessian_approximation': 'limited-memory',
            # 'ipopt.print_level': 0,
            # 'ipopt.tol': 1e-3
            }

    opti.solver('fatrop', opts)

    M = opti.to_function('NMPC', [x0, xrf, urf], [U[0]])

    return M


def main():
    objPath = os.getcwd() + '/../conf/testBox.urdf'
    model, data = getBoxModel(objPath)

    dt = 0.1
    n = 3
    F = euler_integration(model, data, dt)
    M = dynSolver(F, model, {
        'H': 20,
        'r': 1,
        'q': 3
        })

    """
    Control Loop
    """
    # initial state
    Hi = pin.SE3.Random()
    r0 = Hi.translation
    v0 = np.zeros([n, ])
    th0 = pin.Quaternion(Hi.rotation)
    w0 = np.zeros([n, ])
    x0 = np.concatenate((r0, v0, th0, w0), axis=0)

    # reference state
    Hf = pin.SE3.Random()
    rf = Hf.translation
    vf = v0
    thf = pin.Quaternion(Hf.rotation)
    wf = w0
    xf = np.concatenate((rf, vf, thf, wf), axis=0)
    uf = np.zeros([n, ])

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
                't': t,
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
