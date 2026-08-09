import sys
import os
import casadi as ca
from utils import *
import numpy as np
from numpy.linalg import norm, solve
import pinocchio as pin
import pinocchio.casadi as cpin
# print(dir(cpin)) -> prints callable functions

class NMPC:
    def __init__(self, armParameters, ocpParameters):
        # Loading the model of the arm from full body URDF
        self.modelR = self.getArmModel(armParameters, 'r_')
        self.modelL = self.getArmModel(armParameters, 'l_')
        self.dataR = self.modelR.createData()
        self.dataL = self.modelL.createData()

        self.dt = ocpParameters['Ts']
        self.H = ocpParameters['H']

        self.B = ca.DM.eye(6) # Friction Coefficient matrix
        self.B[3, 3] = 0
        self.B[4, 4] = 0

        # Contact rotation matrices
        # self.R1 = ca.DM([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) # Roc1
        # self.R2 = ca.DM([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) # Roc2
        rpy_cr = np.array([np.pi/2, 0, 0])
        rpy_cl = np.array([-np.pi/2, 0, np.pi])
        self.R1 = pin.rpy.rpyToMatrix(rpy_cr)
        self.R2 = pin.rpy.rpyToMatrix(rpy_cl)

        # Solver Initialization
        self.QL = []
        self.QR = []
        self.VL = []
        self.VR = []
        self.AL = []
        self.AR = []
        self.UL = []
        self.UR = []
        self.FL = []
        self.FR = []
        self.Vo = []
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
    
        # self.warmStart(model, q0)

    def initSolver(self, objParameters, Rl, Rr):
        self.objVbk = self.objectDynamics(objParameters)
        self.G = ca.horzcat(self.GR, self.GL)

        # Individual RNEAs
        self.hk_rneaR = self.rnea(self.modelR, Rr)
        self.hk_rneaL = self.rnea(self.modelL, Rl)

        # Arm Model Dynamics
        self.Fk_Inverse = self.inverseDynamics(self.modelR.nq)
        
        # Jacobians 
        self.hk_JacR = self.armJacobian(self.modelR, 'r_', Rr)
        self.hk_JacL = self.armJacobian(self.modelL, 'l_', Rl)

        # Hand/Arms Jacobian
        self.hk_Jh = self.handJacobian(self.modelR.nq)

        # Friction Cone constraint
        self.gForcek = self.softFingerGrasp()

        self.opti = self.rneaSolver(self.modelR.nq)

    def getArmModel(self, params, arm):
        """
        Get Model from urdf
        """
        modelpath = params['path']
        arm_prefix = arm
        jointsToFree = params['joints']
        jointsToFree = [arm + jnts for jnts in jointsToFree]
        jointsToFree.append('universe')

        fullModel = pin.buildModelFromUrdf(modelpath)
        allJoints = list(fullModel.names)
        jointsToLock = [item for item in allJoints if item not in jointsToFree]
        jointsToLockIDs = []

        for jn in jointsToLock:
            jointsToLockIDs.append(fullModel.getJointId(jn))

        initConf = pin.neutral(fullModel)
        model = pin.buildReducedModel(fullModel, jointsToLockIDs, initConf)
        return model

    def inverseDynamics(self, nq):
        # State variable2
        q = ca.SX.sym("q", nq)
        v = ca.SX.sym("v", nq)
        a = ca.SX.sym("a", nq)

        # Euler Integration
        qk = q + self.dt * v
        vk = v + self.dt * a
        return ca.Function('Fk', [q, v, a], [qk, vk], ['q', 'v', 'a'], ['qk', 'vk']).expand()

    def forwardDynamics(self, model, R=np.eye(3)):
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()

        # State variables
        q = ca.SX.sym("q", cmodel.nq)
        v = ca.SX.sym("v", cmodel.nv)

        # Control variables
        # tau = ca.SX.sym("tau", cmodel.nv)
        u = ca.SX.sym("tau", cmodel.nv)
        f = ca.SX.sym("f_ext", 6)

        contactFrame = cpin.SE3(
                ca.SX(R),
                ca.SX.zeros(3)
                )

        # Mapping body frame forces to object contact forces
        f_contact = cpin.Force(f)
        # f_joint6 = f_contact.se3ActionInverse(contactFrame)

        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(model.njoints)]
        f_ext[6] = f_contact.se3ActionInverse(contactFrame)
        # f_ext[6] = cpin.Force(f)  # End-effector is joint index 6

        # ABA
        # a = cpin.aba(cmodel, cdata, q, v, tau, f_ext)
        a = cpin.aba(cmodel, cdata, q, v, u)
        # dx = ca.vertcat(v, a)
        # x = ca.vertcat(q, v)
        # u = ca.vertcat(f, tau)

        # State concatenation
        # aba = ca.Function('aba', [q, v, u], [a], ['x', 'u'], ['ode'])

        # Integrator
        qk = q + v * self.dt
        vk = v + a * self.dt
        # return ca.Function('ddqk', [q, v, f, tau], [qk, vk], ['qk', 'vk', 'conForce', 'tau'], ['qk_next', 'vk_next'])
        return ca.Function('ddqk', [q, v, u], [qk, vk], ['qk', 'vk', 'tau'], ['qk_next', 'vk_next'])

    def rnea(self, model, R=np.eye(3)) -> ca.Function:
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        """
        Acceleration and Torque as inputs to the system

        """
        # State Variables
        q = ca.SX.sym("q", cmodel.nq)
        v = ca.SX.sym("v", cmodel.nv)
        a = ca.SX.sym("a", cmodel.nv)
        f = ca.SX.sym("f", 6)

        contactFrame = cpin.SE3(
                ca.SX(R),
                ca.SX.zeros(3)
                )

        # Mapping body frame forces to object contact forces
        f_contact = cpin.Force(f)
        # f_joint6 = f_contact.se3ActionInverse(contactFrame)

        f_ext = [cpin.Force(ca.SX.zeros(6)) for _ in range(model.njoints)]
        f_ext[6] = f_contact.se3ActionInverse(contactFrame)
        # f_ext[6] = cpin.Force(f)  # End-effector is joint index 6

        # RNEA Function
        # tau = cpin.rnea(cmodel, cdata, q, v, a, f_ext)
        tau = cpin.rnea(cmodel, cdata, q, v, a)
        # cpin.computeRNEADerivatives(cmodel, cdata, q, v, a, f_ext)
        cpin.computeRNEADerivatives(cmodel, cdata, q, v, a)
        cpin.framesForwardKinematics(cmodel, cdata, q)

        # RNEA Derivatives
        du_dq = cdata.dtau_dq
        du_dv = cdata.dtau_dv
        du_da = cdata.M
        rneaJacobian = ca.horzcat(du_dq, du_dv, du_da)

        # Define f(x) model
        # rneaJac = ca.Function('jac_rnea', [q, v, a, f], [rneaJacobian])
        rneaJac = ca.Function('jac_rnea', [q, v, a], [rneaJacobian])
        # return ca.Function('rnea', [q, v, a, f], [tau], ['q', 'v', 'a', 'f'], ['tau'], {'custom_jacobian': rneaJac, 'jac_penalty': 0}).expand()
        return ca.Function('rnea', [q, v, a], [tau], ['q', 'v', 'a'], ['tau'], {'custom_jacobian': rneaJac, 'jac_penalty': 0}).expand()

    def armJacobian(self, model, arm, R=np.eye(3)) -> ca.Function:
        # B.T @ Ri @ J(q) B = Identity so its omitted
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()

        q = ca.SX.sym('q', cmodel.nq)

        frame = arm + 'hand'
        frameID = cmodel.getFrameId(frame)

        J = cpin.computeFrameJacobian(cmodel, cdata, q, frameID, pin.LOCAL)

        # Apply rotation to both linear and angular parts of the Jacobian
        # J_custom_linear = R @ J[0:3, :]
        # J_custom_angular = R @ J[3:6, :]
        # J_custom = ca.vertcat(J_custom_linear, J_custom_angular)
        # J = cpin.getJointJacobian(cmodel, cdata, frameID, pin.LOCAL)
        M = pin.SE3(R, np.zeros((3, )))
        J_custom = M.action * J

        # v = ca.SX.sym('v', cmodel.nv)
        # Jh = J @ v
        # return ca.Function('Jh', [q, v], [Jh], ['q', 'v'], ['Jac'])
        # return ca.Function('J', [q], [J], ['q'], ['Jac'])
        return ca.Function('J_obj', [q], [J_custom], ['q'], ['Jac'])
    
    def objectDynamics(self, params) -> ca.Function:
        model = pin.buildModelFromUrdf(params['path'], pin.JointModelFreeFlyer())
        data = model.createData()
        inertias = model.inertias[0]
        R = ca.DM.eye(3) # Rotation matrix of the body to the inertial/world frame
        I = inertias.inertia # Inertia tensor 3x3
        m = inertias.mass # Object Mass
        massMat = ca.horzcat(m * ca.DM.eye(3), ca.DM.zeros((3, 3)))
        inertMat = ca.horzcat(ca.DM.zeros((3, 3)), I)
        M = ca.vertcat(massMat, inertMat)

        # Contact locations
        p1 = params['pr']
        p2 = params['pl']

        # Grasp Matrices 1 and 2
        M1 = pin.SE3(self.R1, np.zeros((3, )))
        self.GR = M1.dualAction @ self.B
        M2 = pin.SE3(self.R2, np.zeros((3, )))
        self.GL = M2.dualAction @ self.B

        # Body Wrenches
        f1 = ca.SX.sym('wrench1', 6)
        f2 = ca.SX.sym('wrench2', 6)
        
        # Forces on the body
        Fo = self.GR * f1 + self.GL * f2
        Fg = R.T @ np.array([0, 0, -9.8])

        # Body frame instantenous velocity
        w = ca.SX.sym('omega', 3)
        v = ca.SX.sym('v', 3)

        # Cross products
        dv = - m * ca.cross(w, v) + Fg
        dw = - ca.cross(w, I @ w)
        
        # Combined body velocity vector
        dV = ca.vertcat(dv, dw)
        dV = ca.inv(M) @ (dV + Fo)
        
        # Explicit Euler integration
        V0 = ca.SX.sym('Vk', 6)
        Vk = V0 + dV * self.dt
        
        return ca.Function('Jh', [V0, f1, f2], [Vk], ['v0', 'f1', 'f2'], ['Vk'])

    def handJacobian(self, nq) -> ca.Function:
        qR = ca.SX.sym('qR', nq)
        qL = ca.SX.sym('qL', nq)

        vR = ca.SX.sym('vR', nq)
        vL = ca.SX.sym('vL', nq)

        modVelR = self.B @ self.hk_JacR(qR) @ vR
        modVelL = self.B @ self.hk_JacL(qL) @ vL

        conVelArms = ca.vertcat(modVelR, modVelL)

        return ca.Function('hand_jac_constraint', [qR, qL, vR, vL], [conVelArms], ['qR', 'qL', 'vR', 'vL'], ['velConstraintArms'])

    def inverseKinematics(self, model, q0, Href, arm):
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
            iMd = data.oMf[frame_id].actInv(Href)  # Use oMf instead of oMi  
            err = pin.log(iMd).vector  # in frame frame  
            if norm(err) < eps:  
                success = True  
                break  
            if i >= IT_MAX:  
                success = False  
                break  
            J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)  # Use frame Jacobian  
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)  
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))  
            q = pin.integrate(model, q, v * DT)  
            # if not i % 10:  
            #     print(f"{i}: error = {err.T}")  
            i += 1  
      
        if success:  
            print("Convergence achieved!")  
        else:  
            print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")  
      
        # print(f"\nresult: {q.flatten().tolist()}")  
        # print(f"\nfinal error: {err.T}")  
        return q

    def softFingerGrasp(self, miu=2, gamma=2) -> ca.Function:
        f = ca.SX.sym('force', 6)
        eps1 = miu * f[2] - ca.sqrt(f[0]**2 + f[1]**2 + 1e-6)
        eps2 = f[2]
        eps3 = gamma * f[2] - ca.sqrt(f[3]**2 + 1e-6)
        c_coeff = ca.vertcat(eps1, eps2, eps3)
        return ca.Function('Coulomb_Coeff', [f], [c_coeff], ['f'], ['c_coeff'])

    def rneaSolver(self, nq):
        optimizer = ca.Opti()
        # Initialize variables
        for k in range(self.H):
            self.runningVars(optimizer, nq)

        # Terminal variables
        self.terminalVars(optimizer, nq)

        # Reference parameters
        self.qrefR = optimizer.parameter(nq)
        self.qrefL = optimizer.parameter(nq)
        self.urefR = optimizer.parameter(nq)
        self.urefL = optimizer.parameter(nq)

        # Cost function
        obj = 0

        # Running Cost
        for i in range(self.H):
            obj += self.runningCost(i)

        # Terminal Cost
        # obj += self.terminalCost(self.qrefR, self.qrefL)

        optimizer.minimize(obj)

        # Initial Parameters
        self.q0R = optimizer.parameter(nq)
        self.v0R = optimizer.parameter(nq)
        self.q0L = optimizer.parameter(nq)
        self.v0L = optimizer.parameter(nq)
        # self.vObj0 = optimizer.parameter(6)

        # Subject to the model/ descrete function
        optimizer.subject_to(self.QR[0] == self.q0R)
        optimizer.subject_to(self.VR[0] == self.v0R)
        optimizer.subject_to(self.QL[0] == self.q0L)
        optimizer.subject_to(self.VL[0] == self.v0L)
        # optimizer.subject_to(self.Vo[0] == self.vObj0)

        for k in range(self.H):
            # h(xk, uk) = 0
            # xk+1 = f(xk, uk)
            self.h(optimizer, k)
            # g(xk, uk) >= 0
            # self.g(optimizer, k)

        optimizer.solver('ipopt', self.solverOptions)
        return optimizer

    def h(self, optimizer, k) -> None:
        # Inverse dynamics constraint
        # qk, vk = self.Fk_Inverse(self.QR[k], self.VR[k], self.AR[k])
        optimizer.subject_to(self.QR[k + 1] == self.QR[k] + self.VR[k] * self.dt)
        optimizer.subject_to(self.VR[k + 1] == self.VR[k] + self.AR[k] * self.dt)
        optimizer.subject_to(self.UR[k] == self.hk_rneaR(self.QR[k], self.VR[k], self.AR[k]))

        # qk, vk = self.Fk_Inverse(self.QL[k], self.VL[k], self.AL[k])
        optimizer.subject_to(self.QL[k + 1] == self.QL[k] + self.VL[k] * self.dt)
        optimizer.subject_to(self.VL[k + 1] == self.VL[k] + self.AL[k] * self.dt)
        optimizer.subject_to(self.UL[k] == self.hk_rneaL(self.QL[k], self.VL[k], self.AL[k]))

        # Fundamental grasp constraint
        # optimizer.subject_to(self.G.T @ self.Vo[k] == self.hk_Jh(self.QR[k], self.QL[k], self.VR[k], self.VL[k]))
        # Same velocity constraint
        # optimizer.subject_to(self.hk_JacR(self.QR[k]) @ self.VR[k] == self.hk_JacL(self.QL[k]) @ self.VL[k])

    def g(self, optimizer, k) -> None:
        optimizer.subject_to(self.gForcek(self.FR[k]) >= 0)
        optimizer.subject_to(self.gForcek(self.FL[k]) >= 0)

    def runningCost(self, k):
        loss = 0
        loss += 5 * (self.QR[k + 1] - self.qrefR).T @ (self.QR[k + 1] - self.qrefR)
        loss += 5 * (self.QL[k + 1] - self.qrefL).T @ (self.QL[k + 1] - self.qrefL)
        loss += self.VR[k + 1].T @ self.VR[k + 1]
        loss += self.VL[k + 1].T @ self.VL[k + 1]
        loss += (self.UR[k] - self.urefR).T @ (self.UR[k] - self.urefR)
        loss += (self.UL[k] - self.urefL).T @ (self.UL[k] - self.urefL)
        # loss += self.FR[k].T @ self.FR[k]
        # loss += self.FL[k].T @ self.FL[k]
        return loss

    def terminalCost(self, qrefR, qrefL):
        loss = 0
        loss += 10 * (self.QL[-1] - qrefL).T @ (self.QL[-1] - qrefL)
        loss += 10 * (self.QR[-1] - qrefR).T @ (self.QR[-1] - qrefR)
        loss += 1 * self.VR[-1].T @ self.VR[-1]
        loss += 1 * self.VL[-1].T @ self.VL[-1]
        return loss

    def runningVars(self, optimizer, nq):
        self.QR.append(optimizer.variable(nq))
        self.VR.append(optimizer.variable(nq))
        self.AR.append(optimizer.variable(nq))
        self.UR.append(optimizer.variable(nq))
        # self.FR.append(optimizer.variable(6))
        self.QL.append(optimizer.variable(nq))
        self.VL.append(optimizer.variable(nq))
        self.AL.append(optimizer.variable(nq))
        self.UL.append(optimizer.variable(nq))
        # self.FL.append(optimizer.variable(6))
        # self.Vo.append(optimizer.variable(6))

    def terminalVars(self, optimizer, nq):
        self.QR.append(optimizer.variable(nq))
        self.QL.append(optimizer.variable(nq))
        self.VR.append(optimizer.variable(nq))
        self.VL.append(optimizer.variable(nq))
        # self.Vo.append(optimizer.variable(6))


    def solve(self, q0R, q0L, v0R, v0L, qRefR, qRefL, uRefR, uRefL):
        # Initialize parameters
        self.opti.set_value(self.q0R, q0R)
        self.opti.set_value(self.q0L, q0L)
        self.opti.set_value(self.v0R, v0R)
        self.opti.set_value(self.v0L, v0L)
        self.opti.set_value(self.qrefR, qRefR)
        self.opti.set_value(self.qrefL, qRefL)
        self.opti.set_value(self.urefR, uRefR)
        self.opti.set_value(self.urefL, uRefL)

        # self.opti.set_value(self.vObj0, vObject)

        # Warm Start
        # self.set_initial(self.opti)

        # Solve
        solution = self.opti.solve()
        # self.updateSolution(solution)
        u_r_star = np.squeeze(solution.value(self.UR[0]))
        # f_r_star = np.squeeze(solution.value(self.FR[0]))
        u_l_star = np.squeeze(solution.value(self.UL[0])) 
        # f_l_star = np.squeeze(solution.value(self.FL[0]))
        # return u_r_star, f_r_star, u_l_star, f_l_star
        return u_r_star, u_l_star

""" Warm Start
    def set_initial(self, optimizer) -> None:
        for k in range(self.H):
            optimizer.set_initial(self.Q[k + 1], self.Qw[k + 1])
            optimizer.set_initial(self.V[k + 1], self.Vw[k + 1])
            optimizer.set_initial(self.U[k], self.Uw[k])
            optimizer.set_initial(self.A[k], self.Aw[k])

    def updateSolution(self, solution) -> None:
        for k in range(self.H):
            self.Qw[k + 1] = np.squeeze(solution.value(self.Q[k + 1])).copy()
            self.Vw[k + 1] = np.squeeze(solution.value(self.V[k + 1])).copy()
            self.Uw[k] = np.squeeze(solution.value(self.U[k])).copy()
            self.Aw[k] = np.squeeze(solution.value(self.A[k])).copy()

    def warmStart(self, model, q0) -> None:
        data = model.createData()
        v0 = np.zeros((model.nv, ))
        a0 = v0
        u0 = pin.rnea(model, data, q0, v0, a0)
        # Make this a return
        self.Qw = [q0 for _ in range(self.H + 1)]
        self.Vw = [v0 for _ in range(self.H + 1)]
        self.Aw = [a0 for _ in range(self.H)]
        self.Uw = [u0 for _ in range(self.H)]
"""
