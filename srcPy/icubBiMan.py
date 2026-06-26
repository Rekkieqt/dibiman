import sys
import os
import casadi as ca
from utils import *
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

    # Both Functions Stacked
    # FhStack = ca.Function('hStack', [x, a], [xk, tau]).expand()

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
    fJ = ca.Function('jac_dx_f', [x, u, xd], [abaJac])

    # qdd_f = ca.Function('qdd_f', [q, v, tau], [qdd], ['q', 'v', 'tau'], ['qdd'])
    dx_f = ca.Function('dx_f', [x, u], [dx], ['x', 'u'], ['dx'],
                            {"custom_jacobian": fJ, "jac_penalty": 0}).expand()

    Fk = integrator(dx_f, modOpts)

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
    
    # testX = ca.hcat(X[:-1])
    # testA = ca.hcat(A[:-1])
    # allDyn, allU = hstack.map(H, 'openmp')(testX, testA)

    # Cost function
    obj = 0
    for i in range(H):
        obj += ca.mtimes([(X[i + 1] - xrf).T, Qx, X[i + 1] - xrf])
        obj += ca.mtimes([(A[i]).T, Qa, A[i]])
        obj += ca.mtimes([(U[i] - urf).T, R, U[i] - urf])

    opti.minimize(obj)

    # Subject to the model/ descrete function
    opti.subject_to(X[0] == x0)
    for k in range(H):
        opti.subject_to(X[k+1] == F(X[k], A[k]))
        opti.subject_to(U[k] == h(X[k], A[k]))
        # opti.subject_to(X[k+1] == allDyn[k])
        # opti.subject_to(U[k] == allU[k])

    opts = {
            'print_time': 1,
            'expand': True,
            # 'debug': True,
            'jit': True,
            # 'jit_options': {'flags': '-O3', 'verbose': True},
            # 'fatrop.print_level': 0,
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

    return opti, M

def set_initial(opti, Xinit, Ainit, Uinit) -> None:
    for k in len(X):
        opti.set_initial(X[k + 1], Xinit[k + 1])
        opti.set_initial(U[k], Uinit[k])
        opti.set_initial(A[k], Ainit[k])


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
        obj += ca.mtimes([(X[i + 1] - xrf).T, Q, X[i + 1] - xrf])
        obj += ca.mtimes([(U[i] - urf).T, R, U[i] - urf])

    opti.minimize(obj)

    # Subject to the model/ discrete function
    opti.subject_to(X[0] == x0)
    for k in range(H):
        opti.subject_to(X[k + 1] == F(X[k], U[k]))

    opts = {
            'print_time': 1,
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

    opti, M = rneaOCP(F, h, solOpts)
    # M = abaOCP(Fsim, solOpts)

    """
    Control Loop
    """
    nv = model.nv
    nq = model.nv
    nu = model.nq
    q0 = pin.randomConfiguration(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    v0 = np.zeros([nv, ])
    x0 = np.concatenate((q0, v0), axis=0)
    qref = pin.randomConfiguration(model) 
    uref = pin.rnea(model, data, qref, v0, v0)
    xref = np.concatenate((qref, v0), axis=0)

    T = 8
    Ts = modOpts['tf']
    N = int(T/Ts)
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
        opti.set_value(x0, x0)
        opti.set_value(xrf, xref)
        opti.set_value(urf, uref)
        sol = opti.solve()
        Xinit, Uinit, Ainit = sol.value(X), sol.value(U), sol.value(A)
        u[i] = Uinit[0]
        set_initial(opti, Xinit, Ainit, Uinit)
        # u[i] = np.squeeze(M(x[i], xref, uref))
        x[i + 1] = np.squeeze(Fsim(x[i], u[i]))

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
