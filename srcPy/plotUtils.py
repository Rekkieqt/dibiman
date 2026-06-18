import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
"""
TO-DO:
    - Parametrize plotTraj function
    - adjust plot3D function
"""


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


def plot3D(data):
    # Assuming X is your [12, N] state array
    x = X_sol[0, :]
    y = X_sol[1, :]
    z = X_sol[2, :]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Trajectory
    ax.plot(x, y, z, label='Drone Trajectory', color='blue', linewidth=2)
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
    ax.set_title('3D Drone Flight Trajectory')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.show()
