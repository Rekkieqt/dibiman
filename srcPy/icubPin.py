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
    dt = modOpts['tf']

    # u = casadi.SX.sym("tau", cmodel.nv)
    # rnea = casadi.SX.sym("rnea", cmodel.nv)

    q = casadi.SX.sym("q", cmodel.nq)
    a = casadi.SX.sym("a", cmodel.nv)
    v = casadi.SX.sym("v", cmodel.nv)

    qk = casadi.SX.sym("qd", cmodel.nq)
    vk = casadi.SX.sym("vd", cmodel.nv)
    ak = casadi.SX.sym("a", cmodel.nv)

    qk = q + vk * dt #cpin.integrate(model, q, vk) # for floating base
    vk = v + ak * dt

    x = casadi.vertcat(q, v, a)
    xk = casadi.vertcat(qk, vk, ak)

    Fk = casadi.Function('xk', [q, v, a], [qk, vk, ak] ['qk', 'vk', 'ak'], ['qk+1', 'vk+1', 'ak+1']).expand()

    # RNEA
    rnea = cpin.rnea(cmodel, cdata, q, v, a)
    cpin.computeRNEADerivatives(cmodel, cdata, q, v, a)
    hk = casadi.Function('rnea', [q, v, a], [rnea],['q', 'v', 'a'], ['rnea']).expand()

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
    a = casadi.SX.sym("a", cmodel.nq)
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
        # f = Function('f', [x, u], [xdot, L])

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


def getNLPsolver(F, solOpts):
    opti = casadi.Opti()
    r = solOpts['r']
    q = solOpts['q']
    H = solOpts['H']
    # h = modOpts['hk']

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

    # Subject to the model/ descrete function

    opti.subject_to(X[0]==x0)

    for k in range(H):
      opti.subject_to(X[k + 1]==F(X[k], U[k]))
      # opti.subject_to(hk(X[k]==U[k]))
      

    opti
    opts = {
            'print_time': 1,
            'expand': True,
            'jit': True,
            # 'jit_options': {'flags': '-O3', 'verbose': True},
            # 'ipopt.hessian_approximation': 'limited-memory',
            # 'ipopt.print_level': 0,
            # 'ipopt.tol': 1e-3
            # 'fatrop.tol': 1e-3,
            # 'fatrop.max_iter': 200,
            'fatrop.print_level': 0,
            'structure_detection': 'auto'
            }

    opti.solver('fatrop', opts)

    M = opti.to_function('NMPC', [x0, xrf, urf], [U[0]])
    # M = M.map(2, 'openmp')

    return M


def main():
    model_path = os.getcwd() + '/../conf/model.urdf'
    # model = pin.buildModelFromUrdf(model_path)
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
    F = forwardModel(model, modOpts)
    # F, h = inverseModel(model, modOpts)

    solOpts = {
            'H': 15,
            'r': 1,
            'q': 7
            # 'hk': h
            }

    M = getNLPsolver(F, solOpts)

    """
    Control Loop
    """
    nq = model.nv
    nv = model.nv
    nu = model.nv
    q0 = pin.randomConfiguration(model)
    v0 = np.zeros([nv, ])
    x0 = np.concatenate((q0, v0), axis=0)
    qref = pin.randomConfiguration(model) # np.random.normal(loc=0.0, scale=.3, size=nq) # 
    uref = pin.rnea(model, data, qref, v0, v0)
    xref = np.concatenate((qref, v0), axis=0)

    T = 5
    Ts = modOpts['tf']
    N = int(T/Ts)
    t = np.linspace(0, T, N+1)
    x = [None] * (N + 1)
    u = [None] * N
    x[0] = x0


    for i in range(N):
        u[i] = np.squeeze(M(x[i], xref, uref))
        x[i + 1] = np.squeeze(F(x[i], u[i]))

    # Printing last result and references
    # print(x[-1])
    # print(u[-1])
    # print(xref)
    # print(uref)

    # --------------PLOTS-----------
    try:
        import matplotlib.pyplot as plt
        plotData = {
                'x': x,
                'xref': xref,
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
