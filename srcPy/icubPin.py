import sys
import os
import casadi as ca
from plotUtils import *
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
    q = ca.SX.sym("q", cmodel.nq)
    a = ca.SX.sym("a", cmodel.nv)
    v = ca.SX.sym("v", cmodel.nv)

    # Simplified dynamics
    x = ca.vertcat(q, v)
    dx = ca.vertcat(v, a)

    qk = q + tf * v
    vk = v + tf * a
    xk = ca.vertcat(qk, vk)
    Fk = ca.Function('Fk', [x, a], [xk], ['x', 'a'], ['xk']).expand()

    # RNEA Function
    tau = cpin.rnea(cmodel, cdata, q, v, a)
    hk = ca.Function('rnea', [x, a], [tau], ['x', 'a'], ['tau']).expand()

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

    tau = ca.SX.sym("tau", cmodel.nv)
    q = ca.SX.sym("q", cmodel.nq)
    v = ca.SX.sym("v", cmodel.nv)
    xd = ca.SX.sym("xd", cmodel.nv)
    x = ca.SX.sym("x", cmodel.nq)

    # ABA
    qdd = cpin.aba(cmodel, cdata, q, v, tau)  # ODE format of the manipulator dynamics
    cpin.computeABADerivatives(cmodel, cdata, q, v, tau)

    # ABA Derivatives
    ddq_dq = cdata.ddq_dq
    ddq_dv = cdata.ddq_dv
    ddq_dtau = cdata.Minv

    # Construct ABA Jacobian
    df_dq = ca.vertcat(np.zeros((model.nv, model.nv)), ddq_dq)
    df_dv = ca.vertcat(np.eye(model.nv), ddq_dv)
    df_dx = ca.horzcat(df_dq, df_dv)

    df_du = ca.vertcat(np.zeros((model.nv, model.nv)), ddq_dtau)
    abaJac = ca.horzcat(df_dx, df_du)

    # Define f(x) model
    x = ca.vertcat(q, v)
    u = tau
    dx = ca.vertcat(v, qdd)
    fJ = ca.Function('jac_xd_f', [x, u, xd], [abaJac])

    # qdd_f = ca.Function('qdd_f', [q, v, tau], [qdd], ['q', 'v', 'tau'], ['qdd'])
    xd_f = ca.Function('xd_f', [x, u], [dx], ['x', 'u'], ['dx'],
                            {"custom_jacobian": fJ, "jac_penalty": 0}).expand()

    Fk = integrator(xd_f, modOpts)

    """
        COST FUNCTION FOR FORWARD KINEMATICS
        # FK q -> XYZ , Quaternion
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)

        H = cdata.oMf[6]
        H = cpin.SE3ToXYZQUAT(H)
        Hf = ca.Function('Hf', [q], [H], ['q'], ['eH'])
    """
    return Fk


def integrator(F, modOpts):

    """
    Integrate the dynamics
    """
    tf = modOpts['tf']
    t0 = modOpts['t0']

    method = modOpts['method']
    x0 = ca.SX.sym('x0', F.size_in(0))
    u = ca.SX.sym('u', F.size_in(1))
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

        Fk = ca.Function('Fk', [x0, u], [xk],['x0','u'],['xf']).expand()

    elif method == 'euler':

        xk = xk + tf * F(x0, u)
        Fk = ca.Function('Fk', [x0, u], [xk],['x0','u'],['xf']).expand()

    return Fk


def rneaOCP(F, h, solOpts):
    opti = ca.Opti()
    r = solOpts['r']
    q = solOpts['q']
    H = solOpts['H']

    # na = F.size_in(1)[0]
    nx = F.size_in(0)[0]
    na = F.size_in(1)[0]
    nu = h.size_out(0)[0]

    R = r * ca.DM.eye(nu)
    Qx = q * ca.DM.eye(nx)
    Qa = ca.DM.eye(na)

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
        # obj += ca.mtimes([(X[i + 1] - xrf).T, Qx, X[i + 1] - xrf])
        obj += ca.mtimes([(X[i + 1] - xrf).T, Qx, X[i + 1] - xrf])
        obj += ca.mtimes([(A[i]).T, Qa, A[i]])
        obj += ca.mtimes([(U[i] - urf).T, R, U[i] - urf])

    opti.minimize(obj)

    # Subject to the model/ descrete function
    opti.subject_to(X[0] == x0)
    for k in range(H):
        opti.subject_to(X[k+1] == F(X[k], A[k]))
        opti.subject_to(U[k] == h(X[k], A[k]))

    opts = {
            'print_time': 0,
            'expand': True,
            # 'debug': True,
            # 'jit': True,
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
    nu = model.nq
    q0 = pin.randomConfiguration(model)
    v0 = np.zeros([nv, ])
    x0 = np.concatenate((q0, v0), axis=0)
    qref = pin.randomConfiguration(model) 
    uref = pin.rnea(model, data, qref, v0, v0)
    xref = np.concatenate((qref, v0), axis=0)

    T = 5
    Ts = modOpts['tf']
    N = int(T/Ts)
    t = np.linspace(0, T, N+1)
    x = [None] * (N + 1)
    q = [None] * (N + 1)
    p = [None] * (N)
    u = [None] * N
    x[0] = x0

    for i in range(N):
        u[i] = np.squeeze(M(x[i], xref, uref))
        x[i + 1] = np.squeeze(Fsim(x[i], u[i]))

        # Exctracting the trajectory
        q[i] = x[i][:nq]
        pin.forwardKinematics(model, data, q[i])
        pin.updateFramePlacements(model, data)
        p[i] = data.oMi[6].translation

    # Extracting last position
    q[-1] = x[-1][:nq]
    
    # Extracting velocities
    v = [None] * (N + 1)
    v = [xi[-nv:] for xi in x]

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
            'title': 'iCub Arm Control'
            })

    except ImportError as err:
        print(
            "Error while initializing the viewer. "
            "It seems you should install Python meshcat"
        )
        raise err


if __name__ == "__main__":
    main()
