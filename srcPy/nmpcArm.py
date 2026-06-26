import sys
import os
import casadi as ca
from utils import *
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
# print(dir(cpin)) -> prints callable functions

class NMPC:
    def __init__(self, armParameters, method, Ts, ocpParameters):
        self.method = method
        self.model, self.data = self.getArmModel(armParameters) 
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        self.nq = self.cmodel.nq
        self.nv = self.cmodel.nv
        self.na = self.nq
        self.nu = self.na
        self.nx = self.nq + self.nv

        self.dt = Ts
        self.H = ocpParameters['H']
        if method == 'inverse':
            self.inverseModel()
        elif method == 'forward':
            self.forwardModel()

        self.optimizer = ca.Opti()
        self.solverVariablesInit()

        self.solverOptions = {
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


    def getArmModel(self, params):
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
        model = pin.buildReducedModel(fullModel, jointsToLockIDs, initConf)
        data = model.createData()

        return model, data

    def inverseModel(self) -> None:
        """
        Acceleration and Torque as inputs to the system

        """
        # Dynamic variables
        q = ca.SX.sym("q", self.nq)
        v = ca.SX.sym("v", self.nv)
        a = ca.SX.sym("a", self.na)

        # Simplified dynamics
        x = ca.vertcat(q, v)
        dx = ca.vertcat(v, a)

        qk = q + self.dt * v
        vk = v + self.dt * a
        xk = ca.vertcat(qk, vk)
        self.Fk = ca.Function('Fk', [x, a], [xk], ['x', 'a'], ['xk']).expand()

        # RNEA Function
        tau = cpin.rnea(self.cmodel, self.cdata, q, v, a)
        self.hk = ca.Function('rnea', [x, a], [tau], ['x', 'a'], ['tau']).expand()

        # Both Functions Stacked
        # FhStack = ca.Function('hStack', [x, a], [xk, tau]).expand()

    def forwardModel(self) -> None:
        tau = ca.SX.sym("tau", self.nu)
        q = ca.SX.sym("q", self.nq)
        v = ca.SX.sym("v", self.nv)

        # ABA
        ddq = cpin.aba(self.cmodel, self.cdata, q, v, tau)  # ODE format of the manipulator dynamics
        cpin.computeABADerivatives(self.cmodel, self.cdata, q, v, tau)

        # ABA Derivatives
        ddq_dq = self.cdata.ddq_dq
        ddq_dv = self.cdata.ddq_dv
        ddq_dtau = self.cdata.Minv

        # Construct ABA Jacobian
        df_dq = ca.vertcat(np.zeros((self.nv, self.nv)), ddq_dq)
        df_dv = ca.vertcat(np.eye(self.nv), ddq_dv)
        df_dx = ca.horzcat(df_dq, df_dv)

        df_du = ca.vertcat(np.zeros((self.nv, self.nv)), ddq_dtau)
        abaJac = ca.horzcat(df_dx, df_du)

        # Define f(x) model
        x = ca.vertcat(q, v)
        u = tau
        dx = ca.vertcat(v, ddq)
        fJ = ca.Function('jac_dx_f', [x, u, dx], [abaJac])

        # qdd_f = ca.Function('qdd_f', [q, v, tau], [qdd], ['q', 'v', 'tau'], ['qdd'])
        dx_f = ca.Function('dx_f', [x, u], [dx], ['x', 'u'], ['dx'],
                                {"custom_jacobian": fJ, "jac_penalty": 0})
        xk = x
        xk = xk + self.dt * dx_f(x, u)
        self.Fk_Forward = ca.Function('Fk', [x, u], [xk], ['x0', 'u'], ['xf']).expand()
        # self.Fk = integrator(dx_f, modOpts)

    def rneaSolver(self, params) -> None:
        r = params['r']
        q = params['q']
        H = params['H']

        self.A = []
        for k in range(H):
            self.X.append(self.optimizer.variable(self.nx))
            self.U.append(self.optimizer.variable(self.nu))
            self.A.append(self.optimizer.variable(self.na))
        self.X.append(self.optimizer.variable(self.nx))
        self.A.append(self.optimizer.variable(self.na))
        
        # testX = ca.hcat(X[:-1])
        # testA = ca.hcat(A[:-1])
        # allDyn, allU = hstack.map(H, 'openmp')(testX, testA)

        # Cost function
        R = r * ca.DM.eye(self.nu)
        Qx = q * ca.DM.eye(self.nx)
        Qa = ca.DM.eye(self.na)

        obj = 0
        for i in range(H):
            obj += ca.mtimes([(self.X[i + 1] - self.xrf).T, Qx, self.X[i + 1] - self.xrf])
            obj += ca.mtimes([(self.U[i] - self.urf).T, R, self.U[i] - self.urf])
            obj += ca.mtimes([(self.A[i]).T, Qa, self.A[i]])

        self.optimizer.minimize(obj)
        """ Add terminal cost later... """

        # Subject to the model/ descrete function
        self.optimizer.subject_to(self.X[0] == self.x0)
        for k in range(H):
            self.optimizer.subject_to(self.X[k+1] == self.Fk(self.X[k], self.A[k]))
            self.optimizer.subject_to(self.U[k] == self.hk(self.X[k], self.A[k]))
            # opti.subject_to(X[k+1] == allDyn[k])
            # opti.subject_to(U[k] == allU[k])

        self.optimizer.solver('fatrop', self.solverOptions)

    def solverVariablesInit(self) -> None:
        self.x0 = self.optimizer.parameter(self.nx, )
        self.xrf = self.optimizer.parameter(self.nx, )
        self.urf = self.optimizer.parameter(self.nu, )
        self.X = []
        self.U = []
        self.Xinit = [np.zeros((self.nx, )) for _ in range(self.H + 1)]
        self.Ainit = [np.zeros((self.na, )) for _ in range(self.H)]
        self.Uinit = [np.zeros((self.nu, )) for _ in range(self.H)]

    def abaSolver(self, params):
        r = params['r'] # input cost
        q = params['q'] # state cost
        H = params['H'] # horizon length

        R = r * ca.DM.eye(self.nu)
        Q = q * ca.DM.eye(self.nx)

        for k in range(H):
            self.X.append(self.optimizer.variable(self.nx))
            self.U.append(self.optimizer.variable(self.nu))
        self.X.append(self.optimizer.variable(self.nx))

        # Cost function
        obj = 0
        for i in range(H):
            obj += ca.mtimes([(self.X[i + 1] - self.xrf).T, Q, self.X[i + 1] - self.xrf])
            obj += ca.mtimes([(self.U[i] - self.urf).T, R, self.U[i] - self.urf])

        self.optimizer.minimize(obj)

        # Subject to the model/ discrete function
        self.optimizer.subject_to(self.X[0] == self.x0)
        for k in range(H):
            self.optimizer.subject_to(self.X[k + 1] == self.Fk_Forward(self.X[k], self.U[k]))

        self.optimizer.solver('fatrop', self.solverOptions)

    def cost_function(self, x, xref, w):
        loss = w * (x - xref).T @ (x - xref)
        return loss

    def set_initial(self) -> None:
        for k in range(self.H):
            self.optimizer.set_initial(self.X[k + 1], self.Xinit[k + 1])
            self.optimizer.set_initial(self.A[k], self.Ainit[k])
            self.optimizer.set_initial(self.U[k], self.Uinit[k])

    def solve(self, x0, xRef, uRef):
        self.optimizer.set_value(self.x0, x0)
        self.optimizer.set_value(self.xrf, xRef)
        self.optimizer.set_value(self.urf, uRef)

        # Warm Start
        self.set_initial()

        # Solve
        solution = self.optimizer.solve()
        self.Uinit = solution.value(self.U) 
        self.Ainit = solution.value(self.A) 
        self.Xinit = solution.value(self.X) 

        return self.Uinit[0]
