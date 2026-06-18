import sys
import os
import casadi
from casadi import *
from plotTraj import *
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
# print(dir(cpin)) -> prints callable functions


def inverseModel(model, modOpts):
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    tf = modOpts['tf']
    """
    Acceleration and Torque as inputs to the system

    """
    q = casadi.SX.sym("q", cmodel.nq)
    a = casadi.SX.sym("a", cmodel.nv)
    v = casadi.SX.sym("v", cmodel.nv)

    # Simplified dynamics
    x = casadi.vertcat(q, v)
    dx = casadi.vertcat(v, a)

    qk = q + tf * v
    vk = v + tf * a
    xk = casadi.vertcat(qk, vk)
    Fk = casadi.Function('Fk', [x, a], [xk], ['x', 'a'], ['xk'])

    # RNEA Function
    tau = cpin.rnea(cmodel, cdata, q, v, a)
    hk = casadi.Function('rnea', [x, a], [tau], ['x', 'a'], ['tau'])

    return Fk, hk


def getArmModel(params):
    """
    Get Model from urdf
    """
    modelpath = params['path']
    arm = params['arm']
    jointsToFree = params['joints']
    jointsToFree = [arm + jnts for jnts in jointsToFree]
    jointsToFree.append('universe')

    fullModel = pin.buildModelFromUrdf(modelpath)
    allJoints = list(fullModel.names)
    jointsToLock = [item for item in allJoints if item not in jointsToFree]
    jointsToLockIDs = []

    for jn in jointsToLock:
        jointsToLockIDs.append(fullModel.getJointId(jn))

    initConf = np.zeros([len(allJoints) - 1, 1])
    model = cpin.buildReducedModel(fullModel, jointsToLockIDs, initConf)
    data = model.createData()

    return model, data


def forwardModel(model, modOpts):
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    tau = casadi.SX.sym("tau", cmodel.nv)
    q = casadi.SX.sym("q", cmodel.nq)
    v = casadi.SX.sym("v", cmodel.nv)
    xd = casadi.SX.sym("xd", cmodel.nv)
    x = casadi.SX.sym("x", cmodel.nq)

    # ABA
    qdd = cpin.aba(cmodel, cdata, q, v, tau)  # ODE format of the manipulator dynamics
    cpin.computeABADerivatives(cmodel, cdata, q, v, tau)

    # ABA Derivatives
    ddq_dq = cdata.ddq_dq
    ddq_dv = cdata.ddq_dv
    ddq_dtau = cdata.Minv

    # Construct ABA Jacobian
    df_dq = casadi.vertcat(np.zeros((model.nv, model.nv)), ddq_dq)
    df_dv = casadi.vertcat(np.eye(model.nv), ddq_dv)
    df_dx = casadi.horzcat(df_dq, df_dv)

    df_du = casadi.vertcat(np.zeros((model.nv, model.nv)), ddq_dtau)
    abaJac = casadi.horzcat(df_dx, df_du)

    # Define f(x) model
    x = casadi.vertcat(q, v)
    u = tau
    dx = casadi.vertcat(v, qdd)
    fJ = casadi.Function('jac_xd_f', [x, u, xd], [abaJac])

    # qdd_f = casadi.Function('qdd_f', [q, v, tau], [qdd], ['q', 'v', 'tau'], ['qdd'])
    xd_f = casadi.Function('xd_f', [x, u], [dx], ['x', 'u'], ['dx'],
                            {"custom_jacobian": fJ, "jac_penalty": 0}).expand()

    Fk = integrator(xd_f, modOpts)

    """
        COST FUNCTION FOR FORWARD KINEMATICS
        # FK q -> XYZ , Quaternion
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)

        H = cdata.oMf[6]
        H = cpin.SE3ToXYZQUAT(H)
        Hf = casadi.Function('Hf', [q], [H], ['q'], ['eH'])
    """
    return Fk


def integrator(F, modOpts):

    """
    Integrate the dynamics
    """
    tf = modOpts['tf']
    t0 = modOpts['t0']

    method = modOpts['method']
    x0 = SX.sym('x0', F.size_in(0))
    u = SX.sym('u', F.size_in(1))
    xk = x0
    
    if method == 'rk4':

        # Fixed step Runge-Kutta 4 integrator
        M = 4 # RK4 steps per interval
        DT = tf/M
        for j in range(M):
            k1 = F(xk, u)
            k2 = F(xk + DT/2 * k1, u)
            k3 = F(xk + DT/2 * k2, u)
            k4 = F(xk + DT * k3, u)
            xk = xk + DT/6 * (k1 + 2 * k2 + 2 * k3 + k4)

        Fk = Function('Fk', [x0, u], [xk],['x0','u'],['xf']).expand()

    elif method == 'euler':

        xk = xk + tf * F(x0, u)
        Fk = Function('Fk', [x0, u], [xk],['x0','u'],['xf']).expand()

    return Fk


def rneaOCP(F, h, solOpts):
    opti = casadi.Opti()
    r = solOpts['r']
    q = solOpts['q']
    H = solOpts['H']

    # na = F.size_in(1)[0]
    nx = F.size_in(0)[0]
    na = F.size_in(1)[0]
    nu = h.size_out(0)[0]

    R = r * casadi.DM.eye(nu)
    Qx = q * casadi.DM.eye(nx)
    Qa = casadi.DM.eye(na)

    x0 = opti.parameter(nx, ) # q, v, a
    xrf = opti.parameter(nx, ) # q, v, a
    urf = opti.parameter(nu, ) # tau = RNEA(q, v, a), when a=0, v=0, q=qref

    X = []
    U = []
    A = []
    for k in range(H):
        X.append(opti.variable(nx))
        U.append(opti.variable(nu))
        A.append(opti.variable(na))
    X.append(opti.variable(nx))
    A.append(opti.variable(na))

    # Cost function
    obj = 0
    for i in range(H):
        # obj += mtimes([(X[i + 1] - xrf).T, Qx, X[i + 1] - xrf])
        obj += mtimes([(X[i + 1] - xrf).T, Qx, X[i + 1] - xrf])
        obj += mtimes([(A[i]).T, Qa, A[i]])
        obj += mtimes([(U[i] - urf).T, R, U[i] - urf])

    opti.minimize(obj)

    # Subject to the model/ descrete function
    opti.subject_to(X[0] == x0)
    for k in range(H):
        opti.subject_to(X[k+1] == F(X[k], A[k]))
        opti.subject_to(U[k] == h(X[k], A[k]))

    opts = {
            'print_time': 1,
            'expand': True,
            'debug': True,
            'jit': True,
            # 'jit_options': {'flags': '-O3', 'verbose': True},
            'fatrop.print_level': 0,
            # 'fatrop.tolerance': 1e-3,
            # 'fatrop.max_iter': 200
            'structure_detection': 'auto'
            # 'ipopt.hessian_approximation': 'limited-memory',
            # 'ipopt.print_level': 0,
            # 'ipopt.tol': 1e-3
            }

    opti.solver('fatrop', opts)

    M = opti.to_function('NMPC', [x0, xrf, urf], [U[0]])
    # M = M.map(2, 'openmp')

    return M


def abaOCP(F, solOpts):
    opti = casadi.Opti()
    r = solOpts['r'] # input cost
    q = solOpts['q'] # state cost
    H = solOpts['H'] # horizon length

    nq = F.size_in(0)[0]
    nu = F.size_in(1)[0]

    R = r * casadi.DM.eye(nu)
    Q = q * casadi.DM.eye(nq)

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
        obj += mtimes([(X[i + 1] - xrf).T, Q, X[i + 1] - xrf])
        obj += mtimes([(U[i] - urf).T, R, U[i] - urf])

    opti.minimize(obj)

    # Subject to the model/ discrete function
    opti.subject_to(X[0] == x0)
    for k in range(H):
        opti.subject_to(X[k + 1] == F(X[k], U[k]))

    opts = {
            'print_time': 1,
            # 'expand': True,
            # 'jit': True,
            # 'jit_options': {'flags': '-O3', 'verbose': True},
            # 'fatrop.print_level': 0,
            # 'fatrop.tolerance': 1e-3,
            # 'fatrop.max_iter': 200
            # 'structure_detection': 'auto'
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.print_level': 0,
            'ipopt.tol': 1e-3
            }

    opti.solver('ipopt', opts)

    M = opti.to_function('NMPC', [x0, xrf, urf], [U[0]])
    # M = M.map(2, 'openmp')

    return M


def main():
    model_path = os.getcwd() + '/../conf/model.urdf'
    jointsToFree = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_prosup', 'wrist_pitch'] #, 'wrist_yaw']
    """
    Option structs
    """
    pinOpts = {
            "path": model_path,
            "joints": jointsToFree,
            "arm": 'r_'
            }
    modOpts = {
            't0': 0,
            'tf': 0.05,
            'method': 'euler'
            }
    model, data = getArmModel(pinOpts)
    Fsim = forwardModel(model, modOpts)
    F, h = inverseModel(model, modOpts)

    solOpts = {
            'H': 15,
            'r': 1,
            'q': 7
            }

    M = rneaOCP(F, h, solOpts)
    # M = abaOCP(Fsim, solOpts)

    """
    Control Loop
    """
    nv = model.nv
    nq = model.nv
    nu = model.nv
    q0 = pin.randomConfiguration(model)
    v0 = np.zeros([nv, ])
    x0 = np.concatenate((q0, v0), axis=0)
    qref = pin.randomConfiguration(model) # np.random.normal(loc=0.0, scale=.3, size=nq) # 
    uref = pin.rnea(model, data, qref, v0, v0)
    xref = np.concatenate((qref, v0), axis=0) # for RNEA

    T = 5
    Ts = modOpts['tf']
    N = int(T/Ts)
    t = np.linspace(0, T, N+1)
    x = [None] * (N + 1)
    u = [None] * N
    x[0] = x0

    for i in range(N):
        u[i] = np.squeeze(M(x[i], xref, uref))
        x[i + 1] = np.squeeze(Fsim(x[i], u[i]))

    # Printing last result and references
    print('States, last measured vs ref')
    print(x[-1])
    print(xref)
    print('Torques, last measured vs ref')
    print(u[-1])
    print(uref)

    # --------------PLOTS-----------
    try:
        import matplotlib.pyplot as plt
        plotData = {
                'x': x,
                'u': u,
                'xref': xref,
                'uref': uref,
                't': t,
                'N': N
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
