import sys
import os
import casadi as ca
from plot_utils import *
import numpy as np
from numpy.linalg import norm, solve
import pinocchio as pin
import pinocchio.casadi as cpin

class objectFOP:
    def __init__(self, objectModelPath, robotModelPath):

        self.objectModel = pin.buildModelFromUrdf(objectModelPath, pin.JointModelFreeFlyer())
        self.objectData = self.objectModel.createData()

        self.robotModel = pin.buildModelFromUrdf(robotModelPath)
        self.robotData = self.robotModel.createData()

        self.addObjectFrameToModel()

        ''' Given icub world/root reference frame, object being in front '''
        # Rc_right = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        self.left_c1_R = pin.rpy.rpyToMatrix(np.array([0, -np.pi/2, -np.pi/2]))
        # Rc_left = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.right_c2_R = pin.rpy.rpyToMatrix(np.array([np.pi/2, 0, 0]))

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
                'ipopt.print_level': 1,
                # 'ipopt.tol': 1e-3
                }

    def objectNetForce(self, f_in) -> pin.Force:
        model = self.objectModel
        data = self.objectData

        f_ext = [pin.Force.Zero() for _ in range(model.njoints)]
        f_ext[1] = pin.Force(f_in)

        q = pin.neutral(model)
        v = np.zeros((model.nv, ))
        a = np.zeros((model.nv, ))
        tau = np.zeros((model.nv, ))

        pin.rnea(model, data, q, v, a, f_ext)

        return data.f[1]

    def objectDynamics(self) -> ca.Function:
        """
        Object in Body Newton-Euler Equations
        """
        model = self.objectModel
        data = self.objectData

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
        robotM = self.robotModel
        robotD = self.robotData

        r_hand_id = robotM.getFrameId('r_hand_dh_frame')
        l_hand_id = robotM.getFrameId('l_hand_dh_frame')
        l_obj_id = robotM.getFrameId('l_object')

        p_r = robotD.oMf[r_hand_id].translation
        p_l = robotD.oMf[l_hand_id].translation

        H_r_con_world = pin.SE3(self.right_c2_R, p_r)
        H_l_con_world = pin.SE3(self.left_c1_R, p_l)

        H_r_con_to_obj = robotD.oMf[l_obj_id].inverse() * H_r_con_world
        H_l_con_to_obj = robotD.oMf[l_obj_id].inverse() * H_l_con_world

        G = ca.horzcat(H_r_con_to_obj.dualAction, H_l_con_to_obj.dualAction)

        fc = ca.vertcat(f1, f2)
        hnet = G @ fc
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

    def objForceSolver(self, T=1):
        opti = ca.Opti()
        objDynamics = self.objectDynamics()
        frictCone = self.softFingerGrasp()
        twist0 = np.ones((6, ))

        H = 1 # int(T/self.dt)

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

    def addObjectFrameToModel(self) -> None:
        model = self.robotModel
        data = self.robotData

        q = pin.neutral(model)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        left_hand_id = model.getFrameId('l_hand_dh_frame')
        right_hand_id = model.getFrameId('r_hand_dh_frame')

        obj_p = .5 * (data.oMf[left_hand_id].translation + data.oMf[right_hand_id].translation)
        obj_R = np.eye(3)
        obj_H = pin.SE3(obj_R, obj_p)
        # # 'wrist_yaw'

        left_hand_joint_id = model.getJointId('l_wrist_yaw')
        left_hand_joint_to_obj_H = data.oMi[left_hand_joint_id].inverse() * obj_H
        left_hand_joint_to_obj_frame = pin.Frame('l_object',
                                        left_hand_joint_id,
                                        left_hand_id,
                                        left_hand_joint_to_obj_H,
                                        pin.FrameType.OP_FRAME
                                        )

        model.addFrame(left_hand_joint_to_obj_frame)
        data = model.createData()

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        self.robotData = data


def main():
    objectPath = os.getcwd() + '/../conf/testBox.urdf'
    icubPath = os.getcwd() + '/../conf/model.urdf'

    fop_test = objectFOP(objectPath, icubPath)
    fnet = fop_test.objectNetForce(np.array([5, 2, 0, 0, 0, 0]))

    print(f"Net Obj Force Spatial {fnet}")

    try:
        wrench_r, wrench_l = fop_test.objForceSolver() 
    except Exception as e:
        raise

    print(f'right wrench :{wrench_r}')
    print(f'left wrench :{wrench_l}')
    fc = np.hstack((wrench_r, wrench_l))
    print(np.hstack((wrench_r, wrench_l)).shape)



if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
