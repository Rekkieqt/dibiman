import matplotlib.pyplot as plt
import numpy as np


def plotTraj(data):
    x = data['x']
    u = data['u']
    t = data['t']
    xref = data['xref']
    uref = data['uref']
    N = data['N']

    X = np.array(x)
    Xref = np.tile(xref[:, np.newaxis], (1, N + 1))

    U = np.array(u)
    Uref = np.tile(uref[:, np.newaxis], (1, N))

    plt.figure(1)
    plt.plot(t, X)
    plt.plot(t, Xref.T, '--')
    plt.title("Joint Positions and Velocities")
    plt.ylabel("q [rad], v [rad/s]")
    plt.xlabel("time [s]")
    plt.show()

    plt.figure(2)
    plt.plot(t[:N], U)
    plt.plot(t[:N], Uref.T, '--')
    plt.title("Joint Input Torques")
    plt.ylabel("tau [Nm]")
    plt.xlabel("time [s]")
    plt.show()
