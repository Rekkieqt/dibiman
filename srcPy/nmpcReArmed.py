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
    def __init__(self, armParameters, Ts, ocpParameters, objectParameters):
        # Loading the model of the arm from full body URDF
        self.models = []
        for arm in armParameters:
            self.models.append(self.getArmModel(arm))

        self.createDatas(armParameters)
        self.nq = self.models[0].nq
        self.nv = self.models[0].nv
        self.na = self.nq
        self.nu = self.na
        self.nx = self.nq + self.nv

        # Restrained velocity
        self.nJ = 6

        # Jacobian Frame IDs
        self.rightFrameId = self.models[0].getFrameId('r_hand')
        self.leftFrameId = self.models[1].getFrameId('l_hand')

        # Model Dynamics
        self.dt = Ts
        self.H = ocpParameters['H']
        self.Fk_Forward_r = self.forwardModel(self.models[0])
        self.Fk_Forward_l = self.forwardModel(self.models[1])

        # Solver Initialization
        self.optimizer = ca.Opti()
        self.solverOptions = {
                'print_time': 0,
                'expand': True,
                # 'debug': True,
                # 'jit': True,
                # 'jit_options': {'flags': '-O2', 'verbose': False},
                # 'fatrop.print_level': 0,
                # 'fatrop.tolerance': 1e-3,
                # 'fatrop.max_iter': 200
                # 'structure_detection': 'auto'
                # 'ipopt.hessian_approximation': 'limited-memory',
                'ipopt.print_level': 0,
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

    def createDatas(self, armParams) -> None:
        self.datas = []
        self.handIds = []
        for model, arm in zip(self.models, armParams):
            self.datas.append(model.createData())
            prefix = arm['prefix']
            self.handIds.append(model.getFrameId(prefix + 'hand'))

    def createObjIds(self, armParams) -> None:
        self.objIds = []
        for model, arm in zip(self.models, armParams):
            prefix = arm['prefix']
            self.objIds.append(model.getFrameId(prefix + 'object'))

    def objectDynamics(self, params) -> ca.Function:
        """
        Object in Body Newton-Euler Equations
        """
        model = pin.buildModelFromUrdf(params['path'], pin.JointModelFreeFlyer())
        data = model.createData()
        inertias = model.inertias[1]
        R = ca.DM.eye(3) # Rotation matrix of the body to the inertial/world frame
        I = inertias.inertia # Inertia tensor 3x3
        m = inertias.mass # Object Mass
        massMat = ca.horzcat(m * ca.DM.eye(3), ca.DM.zeros((3, 3)))
        inertMat = ca.horzcat(ca.DM.zeros((3, 3)), I)
        M = ca.vertcat(massMat, inertMat)

        # Hand Wrenches
        f1 = ca.SX.sym('wrench1', 6)
        f2 = ca.SX.sym('wrench2', 6)
        # Rotate the hand frames to have 'z' inward
        # H_r_con_world = self.handToContactSE3(self.models[0], self.datas[0], self.handIds[0], np.zeros((3, )))
        # H_l_con_world = self.handToContactSE3(self.models[1], self.datas[1], self.handIds[1], np.array([-np.pi/2, 0, 0]))

        # Rotate the object frames such that they have a 'z' inward p. 239 "Intro to Robotic Manipulation 1994"
        p_r = self.datas[0].oMf[self.handIds[0]].translation
        p_l = self.datas[1].oMf[self.handIds[1]].translation
        H_r_con_world, H_l_con_world = self.contactWorldSE3(p_r, p_l)

        H_r_con_to_obj = self.contactToObjectSE3(self.models[0], self.datas[0], self.objIds[0], H_r_con_world)
        H_l_con_to_obj = self.contactToObjectSE3(self.models[1], self.datas[1], self.objIds[1], H_l_con_world)

        print(np.linalg.matrix_rank(H_r_con_to_obj.dualAction))
        print(np.linalg.matrix_rank(H_l_con_to_obj.dualAction))
        self.G = ca.horzcat(H_r_con_to_obj.dualAction, H_l_con_to_obj.dualAction)
        fc = ca.vertcat(f1, f2)
        hnet = self.G @ fc
        # hnet = H_r_con_to_obj.dualAction @ f1 + H_l_con_to_obj.dualAction @ f2
        Fg = R.T @ np.array([0, 0, -9.8])

        # Body frame instantenous velocity
        w = ca.SX.sym('omega', 3)
        v = ca.SX.sym('v', 3)

        # Cross products
        dv = - m * ca.cross(w, v) + Fg
        dw = - ca.cross(w, I @ w)
        
        # Combined body velocity vector
        dV = ca.vertcat(dv, dw)
        dV = ca.inv(M) @ (dV + hnet)
        twist = ca.vertcat(v, w)

        return ca.Function('ObjDynEq', [twist, f1, f2], [dV], ['vel', 'f1', 'f2'], ['acc'])

    def softFingerGrasp(self, miu=.9, gamma=.5) -> ca.Function:
        """
        Coulomb Friction Cone
        """
        f = ca.SX.sym('force', 6)
        eps1 = miu * f[2] - ca.sqrt(f[0]**2 + f[1]**2 + 1e-6)
        eps2 = f[2]
        eps3 = ca.sqrt(f[5]**2 + 1e-6) - gamma * f[2] 
        c_coeff = ca.vertcat(eps1, eps2, eps3)
        return ca.Function('Coulomb_Coeff', [f], [c_coeff], ['f'], ['c_coeff'])

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
        return ca.Function('Fk', [x, a], [xk], ['x', 'a'], ['xk']).expand()

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

    def armJacobian(self, model, frameID) -> ca.Function:
        # B.T @ Ri @ J(q) B = Identity so its omitted
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()

        q = ca.SX.sym('q', cmodel.nq)
        v = ca.SX.sym('v', cmodel.nv)

        J = cpin.computeFrameJacobian(cmodel, cdata, q, frameID, pin.WORLD)

        dx = J @ v
        # lin_vel = dx[0:3]
        x = ca.vertcat(q, v)
        # return ca.Function('J_obj', [q], [J_custom], ['q'], ['Jac'])
        return ca.Function('J_obj', [x], [dx], ['x'], ['spatial_vel'])

    def objForceSolver(self, objectParams, T=1) -> None:
        opti = ca.Opti()
        objDynamics = self.objectDynamics(objectParams)
        frictCone = self.softFingerGrasp()
        twist0 = np.ones((6, ))

        H = int(T/self.dt)

        F1 = []
        F2 = []
        twistV = opti.parameter(6)
        for k in range(H):
            F1.append(opti.variable(6))
            F2.append(opti.variable(6))

        obj = 0
        for i in range(H):
            obj += ca.sumsqr(F1[i])
            obj += ca.sumsqr(F2[i])

        opti.minimize(obj)

        # Subject to the model/ descrete function
        for k in range(H):
            opti.subject_to(objDynamics(twistV, F1[k], F2[k]) == 0)
            opti.subject_to(frictCone(F1[k]) >= 0)
            opti.subject_to(frictCone(F2[k]) >= 0)

        opti.solver('ipopt', self.solverOptions)
        opti.set_value(twistV, np.zeros((6, )))
        
        solution = opti.solve()
        F1_star = np.array(solution.value(F1[-1]))
        F2_star = np.array(solution.value(F2[-1]))
        
        return F1_star, F2_star

    def rneaSolver(self, params) -> None:
        self.solverVariablesInit()
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
        self.Xl.append(self.optimizer.variable(self.nx))
        
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
            self.optimizer.subject_to(self.Xr[k + 1] == self.inverseModel(self.nq)(self.Xr[k], self.Ar[k]))
            self.optimizer.subject_to(self.Ur[k] == self.rnea(self.models[0])(self.Xr[k], self.Ar[k]))

            self.optimizer.subject_to(self.Xl[k + 1] == self.inverseModel(self.nq)(self.Xl[k], self.Al[k]))
            self.optimizer.subject_to(self.Ul[k] == self.rnea(self.models[1])(self.Xl[k], self.Al[k]))

            err = self.armJacobian(self.models[0], self.handIds[0])(self.Xr[k + 1]) - self.armJacobian(self.models[1], self.handIds[1])(self.Xl[k + 1])
            self.optimizer.subject_to(err == 0)

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

    def inverseKinematics(self, model, q0, Href, frame_id, target='full'):
        data = model.createData()
        eps = 1e-6  
        IT_MAX = 4000
        DT = 1e-1
        damp = 1e-12  

        q = q0.copy()  
        i = 0  
        while True:  
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacement(model, data, frame_id)  # Update frame placement  
            if target == 'full':
                iMd = data.oMf[frame_id].actInv(Href)  # Use oMf instead of oMi  
                err = pin.log(iMd).vector  # in frame frame  
                J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)  # Use frame Jacobian  
                J = -np.dot(pin.Jlog6(iMd.inverse()), J)  
                v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))  
                q = pin.integrate(model, q, v * DT)  
                # if not i % 10:  
                #     print(f"{i}: error = {err.T}")  
            else:
                err = data.oMf[frame_id].translation - Href.translation
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
            print(f"Convergence achieved for {model.frames[frame_id].name}!")  
        else:  
            print(f"\nWarning: the iterative algorithm has not reached convergence to the desired precision {model.frames[frame_id].name}")  
        return q

    def contactWorldSE3(self, p_right, p_left):
        ''' Given icub world/root reference frame, object being in front '''
        # Rc_right = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        rot_c1 = pin.rpy.rpyToMatrix(np.array([0, -np.pi/2, -np.pi/2]))
        # Rc_left = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rot_c2 = pin.rpy.rpyToMatrix(np.array([np.pi/2, 0, 0]))

        H_r_con_world = pin.SE3(rot_c2, p_right)
        H_l_con_world = pin.SE3(rot_c1, p_left)
        return H_r_con_world, H_l_con_world

    def contactToObjectSE3(self, model, data, objectFrameId, H_con_world):
        ''' Relative between hand and object '''
        H_con_to_obj = data.oMf[objectFrameId].inverse() * H_con_world
        return H_con_to_obj

    def handToContactSE3(self, model, data, handId, rpyVec=np.zeros((3, ))):
        ''' Rotating the hand such that z direction points inwards the object '''
        R_ = pin.rpy.rpyToMatrix(rpyVec)

        ''' Obtain world SE3 of the hand '''
        H_hand = data.oMf[handId]

        ''' Relative between hand and contact '''
        H_con = pin.SE3(R_, 
                         np.zeros((3, )))

        H_con_world = H_hand * H_con

        return H_con_world

    def updateSolution(self, solution) -> None:
        for k in range(self.H):
            self.Xinit[k + 1] = np.squeeze(solution.value(self.X[k]))
            self.Uinit[k] = np.squeeze(solution.value(self.U[k]))
            self.Ainit[k] = np.squeeze(solution.value(self.A[k]))

