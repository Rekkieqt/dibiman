import sys
import os

import casadi as ca
from utils import *
from pathlib import Path
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

class objectOCP:
    def __init__(self, path, optionsOCP):
        self.model, self.data, self.cmodel, self.cdata = self.getBoxModel(path)
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nxDiff = self.nx - 1
        self.H = optionsOCP['H']

        # Lie group algebra equations
        self.hk_difference = self.state_difference(self.cmodel)
        self.hk_integrate = self.state_integrate(self.cmodel)
        self.fk_forward = self.euler_integration(self.cmodel, self.cdata, optionsOCP['dt'])

        # Solver Options
        self.opti = ca.Opti()
        self.solver = 'ipopt'
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
        # Initialize warm start data
        self.solverVariablesInit()
        self.objectOCPSolverNaive(optionsOCP)

    def getBoxModel(self, path):
        """
        Get Model from urdf
        """
        # model_path = Path(os.environ.get("pinRobotDir"))
        # urdf_filename = model_path / "hector_description/robots/quadrotor_base.urdf"
        # model = pin.buildModelFromUrdf(urdf_filename, pin.JointModelFreeFlyer())
        model = pin.buildModelFromUrdf(path, pin.JointModelFreeFlyer())
        data = model.createData()
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        return model, data, cmodel, cdata

    def actuation_model(self, nu):
        u1 = ca.SX.sym("u1", nu)
        u2 = ca.SX.sym("u2", nu)
        u = ca.vertcat(u1, u2)
        self.nu = u.shape[0]
        tau = u1 + u2

        return ca.Function("act_model", [u], [tau], ["u"], ["tau"])

    def state_integrate(self, model):
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

    def state_difference(self, model):
        q0 = ca.SX.sym("q0", model.nq)
        q1 = ca.SX.sym("q1", model.nq)
        v0 = ca.SX.sym("v0", model.nv)
        v1 = ca.SX.sym("v1", model.nv)

        q_diff = cpin.difference(model, q0, q1) # of dimension 6, because quaternions only have 3 dof
        v_diff = v1 - v0

        x0 = ca.vertcat(q0, v0)
        x1 = ca.vertcat(q1, v1)
        x_diff = ca.vertcat(q_diff, v_diff) # dimension 6

        return ca.Function("difference", [x0, x1], [x_diff], ["x0", "x1"], ["x_diff"])

    def euler_integration(self, model, data, dt=0.05):
        self.nu = model.nv
        u1 = ca.SX.sym('u1', self.nu)
        u2 = ca.SX.sym('u2', self.nu)
        u = ca.vertcat(u1, u2)

        # tau = self.actuation_model(nu)(u)
        tau = ca.SX.sym('u', self.nv)

        q = ca.SX.sym("q", self.nq)
        v = ca.SX.sym("v", self.nv)

        a = cpin.aba(model, data, q, v, tau)

        dq = v * dt + a * dt**2
        dv = a * dt

        x = ca.vertcat(q, v)
        dx = ca.vertcat(dq, dv)
        x_next = self.state_integrate(model)(x, dx)

        return ca.Function("int_dyn", [x, tau], [x_next], ["x", "u"], ["x_next"])

    def objectOCPSolverNaive(self, params):
        r = params['r'] # input cost
        q = params['q'] # state cost
        H = params['H'] # horizon length

        R = r * ca.DM.eye(self.nu)
        Q = q * ca.DM.eye(self.nxDiff)

        for k in range(H):
            self.X.append(self.opti.variable(self.nxDiff))
            self.U.append(self.opti.variable(self.nu))
        self.X.append(self.opti.variable(self.nxDiff))

        # Objective function
        obj = 0

        # Initial state
        x0 = self.hk_integrate(self.xrf, self.X[0])
        self.opti.subject_to(self.hk_difference(x0, self.x0) == [0] * self.nxDiff)

        # State & Control regularization
        for k in range(H):
            xk = self.hk_integrate(self.xrf, self.X[k])
            err = self.hk_difference(self.xrf, xk)
            obj += ca.mtimes([err.T, Q, err])
            obj += ca.mtimes([self.U[k].T, R, self.U[k]])

        self.opti.minimize(obj)

        # Dynamical constraints
        for k in range(H):
            xk = self.hk_integrate(self.xrf, self.X[k])
            xk1 = self.hk_integrate(self.xrf, self.X[k + 1])
            fxu = self.fk_forward(xk, self.U[k])
            self.opti.subject_to(self.hk_difference(xk1, fxu) == [0] * self.nxDiff)

        self.opti.solver(self.solver, self.solverOptions)

    def solverVariablesInit(self) -> None:
        self.x0 = self.opti.parameter(self.nx, )
        self.xrf = self.opti.parameter(self.nx, )
        self.X = []
        self.U = []
        self.Xinit = [np.zeros((self.nxDiff, )) for _ in range(self.H + 1)]
        self.Uinit = [np.zeros((self.nu, )) for _ in range(self.H)]

    def set_initial(self) -> None:
        for k in range(self.H):
            self.opti.set_initial(self.X[k + 1], self.Xinit[k + 1])
            self.opti.set_initial(self.U[k], self.Uinit[k])

    def updateSolution(self, solution) -> None:
        for k in range(self.H):
            self.Xinit[k + 1] = np.squeeze(solution.value(self.X[k]))
            self.Uinit[k] = np.squeeze(solution.value(self.U[k]))

    def solve(self, x0, xRef):
        # Initialize parameters
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.xrf, xRef)

        # Warm Start
        self.set_initial()

        # Solve
        solution = self.opti.solve()
        self.updateSolution(solution)
        return self.Uinit[0]

    def plantModel(self):
        return self.fk_forward

    def pinModelandData(self):
        return self.model, self.data


def main():
    H = 2
    Ts = 0.05
    objPath = os.getcwd() + '/../conf/testBox.urdf'
    objectNMPC = objectOCP(objPath, optionsOCP={
        'dt': Ts,
        'H' : H,
        'r' : 1,
        'q' : 7
        })
    model, data = objectNMPC.pinModelandData()
    Fk = objectNMPC.plantModel()

    nq = model.nq
    nv = model.nv

    """
    Control Loop
    """
    # initial state
    Hi = pin.SE3.Random()
    Hivec = pin.SE3ToXYZQUAT(Hi)
    v0 = np.zeros([nv, ])
    x0 = np.concatenate((Hivec, v0), axis=0)

    # reference state
    Hf = pin.SE3.Random()
    Hfvec = pin.SE3ToXYZQUAT(Hf)
    vf = v0
    xf = np.concatenate((Hfvec, vf), axis=0)

    T = 8
    N = int(T/Ts)
    t = np.linspace(0, T, N+1)
    x = [None] * (N + 1)
    p = np.zeros((3, N+1))
    u = [None] * N
    x[0] = x0
    p[:, 0] = Hi.translation

    # Homogenous matrices
    frames = []
    frames.append(np.array(Hi))

    for i in range(N):
        u[i] = np.squeeze(objectNMPC.solve(x[i], xf))
        x[i + 1] = np.squeeze(Fk(x[i], u[i]))
        p[:, i + 1] = pin.XYZQUATToSE3(x[i + 1][0:nq]).translation
        # Xpast, Upast = Xsolved.copy(), Usolved.copy()

    # Printing last result and references
    print('States, last measured vs ref')
    print(x[-1])
    print(xf)
    qf = x[-1][0:nq]
    print(pin.difference(model, qf, Hfvec))
    print('Controls, force and torque')
    print(u[-1])
    frames.append(np.array(pin.XYZQUATToSE3(x[-1][0:nq])))

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
        plot3D({
            'title': 'Box Trajectory',
            'xlabel': 'Pos',
            'x': p,
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
