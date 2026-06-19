import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
"""
TO-DO:
    - Add 3D Frame Rotation
"""


def plotTraj(data):
    x = data['x']
    t = data['t']
    title = data['title']
    xl = data['xlabel']
    yl = data['ylabel']
    xref = data['xref']
    # uref = data['uref']
    N = len(x)
    t = t[:N]

    X = np.array(x)
    Xref = np.tile(xref[:, np.newaxis], (1, N))

    plt.figure()
    plt.plot(t, X)
    plt.plot(t, Xref.T, '--')
    plt.title(title)
    plt.ylabel(yl)
    plt.xlabel(xl)
    plt.show()


def plot3D(data):
    # Assuming X is your [12, N] state array
    title = data['title']
    xl = data['xlabel']
    x = data['x']

    X = np.array(x)
    x = X[0, :]
    y = X[1, :]
    z = X[2, :]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Trajectory
    ax.plot(x, y, z, label=xl, color='blue', linewidth=2)
    ax.scatter(x[0], y[0], z[0], color='green', label='Start', s=50)
    ax.scatter(x[-1], y[-1], z[-1], color='red', label='End', s=50)

    # Equal axis scaling workaround
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Labels and formatting
    ax.set_title(title)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.show()


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

        Fk = ca.Function('Fk', [x0, u], [xk], ['x0', 'u'], ['xf']).expand()

    elif method == 'euler':

        xk = xk + tf * F(x0, u)
        Fk = ca.Function('Fk', [x0, u], [xk], ['x0', 'u'], ['xf']).expand()
        
    elif method == 'objRK4':
        p1 = ca.SX.sym('p1', F.size_in(2))
        p2 = ca.SX.sym('p2', F.size_in(3))
        M = 4 # RK4 steps per interval
        DT = tf/M
        for j in range(M):
            k1 = F(xk, u)
            k2 = F(xk + DT/2 * k1, u, p1, p2)
            k3 = F(xk + DT/2 * k2, u, p1, p2)
            k4 = F(xk + DT * k3, u, p1, p2)
            xk = xk + DT/6 * (k1 + 2 * k2 + 2 * k3 + k4)

        Fk = ca.Function('Fk', [x0, u, p1, p2], [xk], ['x0', 'u'], ['xf']).expand()

    return Fk

