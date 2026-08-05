import sys
import os
import casadi as ca
from utils import *
import numpy as np
from numpy.linalg import norm, solve
import pinocchio as pin
import pinocchio.casadi as cpin
# print(dir(cpin)) -> prints callable functions

class armNMPC:
    def __init__(self, armParameters, Ts, ocpParameters):
        # Loading the model of the arm from full body URDF
        self.models = []
        for arm in armParameters:
            self.models.append(self.getArmModel(arm))
        self.datas = []
        for model in self.models:
            self.datas.append(model.createData())

        self.nq = self.models[0].nq
        self.nv = self.models[0].nv
        self.na = self.nq
        self.nu = self.na
        self.nx = self.nq + self.nv

        # Model Dynamics
        self.dt = Ts
        self.H = ocpParameters['H']
        # self.Fk_Forward = self.forwardModel(self.cmodel, self.cdata)
        self.Fk_Inverse = self.inverseModel(self.nq)

        self.Fk_Forward_r = self.forwardModel(self.models[0])
        self.Fk_Forward_l = self.forwardModel(self.models[1])
        # self.Fk_Inverse, self.hk_rnea = self.inverseModel(self.models[0])

        # Solver Initialization
        self.optimizer = ca.Opti()
        self.solverVariablesInit()
        self.solverOptions = {
                'print_time': 1,
                'expand': True,
                # 'debug': True,
                # 'jit': True,
                # 'jit_options': {'flags': '-O2', 'verbose': False},
                # 'fatrop.print_level': 0,
                # 'fatrop.tolerance': 1e-3,
                # 'fatrop.max_iter': 200
                # 'structure_detection': 'auto'
                # 'ipopt.hessian_approximation': 'limited-memory',
                # 'ipopt.print_level': 0,
                # 'ipopt.tol': 1e-3
                }
        self.rneaSolver(ocpParameters)

    def getArmModel(self, params):
        """
        Get Model from urdf
        """
        modelpath = params['path']
        arm = params['prefix']
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

        return model

    def inverseModel(self, nq):
        """
        Acceleration and Torque as inputs to the system

        """

        # Dynamic variables
        q = ca.SX.sym("q", nq)
        v = ca.SX.sym("v", nq)
        a = ca.SX.sym("a", nq)

        # Simplified dynamics
        x = ca.vertcat(q, v)
        dx = ca.vertcat(v, a)

        qk = q + self.dt * v
        vk = v + self.dt * a
        xk = ca.vertcat(qk, vk)
        Fk_Inverse = ca.Function('Fk', [x, a], [xk], ['x', 'a'], ['xk']).expand()

        return Fk_Inverse

    def rnea(self, model):
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        # Dynamic variables
        q = ca.SX.sym("q", cmodel.nq)
        v = ca.SX.sym("v", cmodel.nv)
        a = ca.SX.sym("a", cmodel.nv)

        # RNEA Function
        tau = cpin.rnea(cmodel, cdata, q, v, a)
        cpin.computeRNEADerivatives(cmodel, cdata, q, v, a)

        # RNEA Derivatives
        du_dq = cdata.dtau_dq
        du_dv = cdata.dtau_dv
        du_da = cdata.M

        rneaJacobian = ca.horzcat(du_dq, du_dv, du_da)
        x = ca.vertcat(q, v)

        # Define f(x) model
        rneaJac = ca.Function('jac_rnea', [x, a], [rneaJacobian])

        hk_rnea = ca.Function('rnea', [x, a], [tau], ['x', 'a'], ['tau'],
                                   {'custom_jacobian': rneaJac, 'jac_penalty': 0}).expand()

        return hk_rnea

    def forwardModel(self, model):

        cmodel = cpin.Model(model)
        cdata = cmodel.createData()

        u = ca.SX.sym("tau", cmodel.nv)
        q = ca.SX.sym("q", cmodel.nq)
        v = ca.SX.sym("v", cmodel.nv)

        # ABA
        ddq = cpin.aba(cmodel, cdata, q, v, u)  # ODE format of the manipulator dynamics
        cpin.computeABADerivatives(cmodel, cdata, q, v, u)

        # ABA Derivatives
        ddq_dq = cdata.ddq_dq
        ddq_dv = cdata.ddq_dv
        ddq_dtau = cdata.Minv

        # Construct ABA Jacobian
        df_dq = ca.vertcat(np.zeros((cmodel.nv, cmodel.nv)), ddq_dq)
        df_dv = ca.vertcat(np.eye(cmodel.nv), ddq_dv)
        df_dx = ca.horzcat(df_dq, df_dv)

        df_du = ca.vertcat(np.zeros((cmodel.nv, cmodel.nv)), ddq_dtau)
        abaJacobian = ca.horzcat(df_dx, df_du)

        # Define f(x) model
        x = ca.vertcat(q, v)
        dx = ca.vertcat(v, ddq)
        fJ = ca.Function('jac_dx_f', [x, u], [abaJacobian])

        # qdd_f = ca.Function('qdd_f', [q, v, tau], [qdd], ['q', 'v', 'tau'], ['qdd'])
        dx_f = ca.Function('dx_f', [x, u], [dx], ['x', 'u'], ['dx'],
                           {"custom_jacobian": fJ, "jac_penalty": 0})
        xk = x
        xk = xk + self.dt * dx_f(x, u)
        return ca.Function('Fk', [x, u], [xk], ['x0', 'u'], ['xf']).expand()
        # self.Fk = integrator(dx_f, modOpts)

    def rneaSolver(self, params) -> None:
        r = params['r']
        q = params['q']
        H = params['H']

        for k in range(H):
            self.Xr.append(self.optimizer.variable(self.nx))
            self.Ur.append(self.optimizer.variable(self.nu))
            self.Ar.append(self.optimizer.variable(self.na))
            self.Xl.append(self.optimizer.variable(self.nx))
            self.Ul.append(self.optimizer.variable(self.nu))
            self.Al.append(self.optimizer.variable(self.na))
        self.Xr.append(self.optimizer.variable(self.nx))
        self.Ar.append(self.optimizer.variable(self.na))
        self.Xl.append(self.optimizer.variable(self.nx))
        self.Al.append(self.optimizer.variable(self.na))
        
        # Cost function
        R = r * ca.DM.eye(self.nu)
        Qx = q * ca.DM.eye(self.nx)
        Qa = ca.DM.eye(self.na)

        obj = 0
        for i in range(H):
            obj += ca.mtimes([(self.Xr[i + 1] - self.xrfr).T, Qx, self.Xr[i + 1] - self.xrfr])
            obj += ca.mtimes([(self.Ur[i] - self.urfr).T, R, self.Ur[i] - self.urfr])
            obj += ca.mtimes([(self.Ar[i]).T, Qa, self.Ar[i]])

            obj += ca.mtimes([(self.Xl[i + 1] - self.xrfl).T, Qx, self.Xl[i + 1] - self.xrfl])
            obj += ca.mtimes([(self.Ul[i] - self.urfl).T, R, self.Ul[i] - self.urfl])
            obj += ca.mtimes([(self.Al[i]).T, Qa, self.Al[i]])

        self.optimizer.minimize(obj)
        """ Add terminal cost later... """

        # Subject to the model/ descrete function
        self.optimizer.subject_to(self.Xr[0] == self.x0r)
        self.optimizer.subject_to(self.Xl[0] == self.x0l)
        for k in range(H):
            self.optimizer.subject_to(self.Xr[k+1] == self.Fk_Inverse(self.Xr[k], self.Ar[k]))
            self.optimizer.subject_to(self.Ur[k] == self.rnea(self.models[0])(self.Xr[k], self.Ar[k]))

            self.optimizer.subject_to(self.Xl[k+1] == self.Fk_Inverse(self.Xl[k], self.Al[k]))
            self.optimizer.subject_to(self.Ul[k] == self.rnea(self.models[1])(self.Xl[k], self.Al[k]))

        self.optimizer.solver('ipopt', self.solverOptions)

    def solverVariablesInit(self) -> None:
        self.x0r = self.optimizer.parameter(self.nx, )
        self.xrfr = self.optimizer.parameter(self.nx, )
        self.urfr = self.optimizer.parameter(self.nu, )
        self.Xr = []
        self.Ur = []
        self.Ar = []

        self.x0l = self.optimizer.parameter(self.nx, )
        self.xrfl = self.optimizer.parameter(self.nx, )
        self.urfl = self.optimizer.parameter(self.nu, )
        self.Xl = []
        self.Ul = []
        self.Al = []
        self.Xinit = [np.zeros((self.nx, )) for _ in range(self.H + 1)]
        self.Ainit = [np.zeros((self.na, )) for _ in range(self.H)]
        self.Uinit = [np.zeros((self.nu, )) for _ in range(self.H)]

    def cost_function(self, x, xref, w):
        loss = w * (x - xref).T @ (x - xref)
        return loss

    def set_initial(self) -> None:
        for k in range(self.H):
            self.optimizer.set_initial(self.Xr[k + 1], self.Xinit[k + 1])
            self.optimizer.set_initial(self.Ur[k], self.Uinit[k])
            self.optimizer.set_initial(self.Ar[k], self.Ainit[k])

    def solve(self, x0r, x0l, xRefr, xRefl, uRefr, uRefl):
        # Initialize parameters
        self.optimizer.set_value(self.x0r, x0r)
        self.optimizer.set_value(self.xrfr, xRefr)
        self.optimizer.set_value(self.urfr, uRefr)

        self.optimizer.set_value(self.x0l, x0l)
        self.optimizer.set_value(self.xrfl, xRefl)
        self.optimizer.set_value(self.urfl, uRefl)

        # Warm Start
        # self.set_initial()

        # Solve
        solution = self.optimizer.solve()
        # self.updateSolution(solution)
        u_r_star = np.squeeze(solution.value(self.Ur[0]))
        u_l_star = np.squeeze(solution.value(self.Ul[0]))
        return u_r_star, u_l_star

    def updateSolution(self, solution) -> None:
        for k in range(self.H):
            self.Xinit[k + 1] = np.squeeze(solution.value(self.X[k]))
            self.Uinit[k] = np.squeeze(solution.value(self.U[k]))
            self.Ainit[k] = np.squeeze(solution.value(self.A[k]))

    def inverseKinematics(self, model, q0, Href, arm, target='full'):
        data = model.createData()
        eps = 1e-6  
        IT_MAX = 4000
        DT = 1e-1
        damp = 1e-12  
        frame = arm + 'hand'
        frame_id = model.getFrameId(frame)

        q = q0.copy()  
        i = 0  
        while True:  
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacement(model, data, frame_id)  # Update frame placement  
            if target is 'full':
                iMd = data.oMf[frame_id].actInv(Href)  # Use oMf instead of oMi  
                err = pin.log(iMd).vector  # in frame frame  
                J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)  # Use frame Jacobian  
                J = -np.dot(pin.Jlog6(iMd.inverse()), J)  
                v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))  
                q = pin.integrate(model, q, v * DT)  
                # if not i % 10:  
                #     print(f"{i}: error = {err.T}")  
            else:
                err = data.oMf[frame_id].translation - Href
                J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
                v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(3), err))  
                q = pin.integrate(model, q, v * DT)  
                # if not i % 10:  
                #     print(f"{i}: error = {err.T}")  
            if norm(err) < eps:  
                success = True  
                break  
            if i >= IT_MAX:  
                success = False  
                break  
            i += 1  
      
        if success:  
            print(f"Convergence achieved for {frame}!")  
        else:  
            print(f"\nWarning: the iterative algorithm has not reached convergence to the desired precision {frame}")  
      
        # print(f"\nresult: {q.flatten().tolist()}")  
        # print(f"\nfinal error: {err.T}")  
        return q
