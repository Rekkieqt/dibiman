import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
from mpl_toolkits.mplot3d import Axes3D
"""
TO-DO:
    - Add 3D Frame Rotation
"""


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
    frames = data['frames']

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

    # Plot Frames
    plotCoordinateFrame(ax, frames[0], size=0.05)
    plotCoordinateFrame(ax, frames[-1], size=0.05)

    plt.tight_layout()
    plt.show()



""" from https://github.com/ethz-asl/kalibr/tree/master """
def plotCoordinateFrame(axis, T_0f, size=1, linewidth=3, name=None):
    """Plot a coordinate frame on a 3d axis. In the resulting plot,
    x = red, y = green, z = blue.
    
    plotCoordinateFrame(axis, T_0f, size=1, linewidth=3)

    Arguments:
    axis: an axis of type matplotlib.axes.Axes3D
    T_0f: The 4x4 transformation matrix that takes points from the frame of interest, to the plotting frame
    size: the length of each line in the coordinate frame
    linewidth: the width of each line in the coordinate frame

    Usage is a bit irritating:
    import mpl_toolkits.mplot3d.axes3d as p3
    import pylab as pl

    f1 = pl.figure(1)
    # old syntax
    # a3d = p3.Axes3D(f1)
    # new syntax
    a3d = f1.add_subplot(111, projection='3d')
    # ... Fill in T_0f, the 4x4 transformation matrix
    plotCoordinateFrame(a3d, T_0f)

    see http://matplotlib.sourceforge.net/mpl_toolkits/mplot3d/tutorial.html for more details
    """
    # \todo fix this check.
    #if type(axis) != axes.Axes3D:
    #    raise TypeError("axis argument is the wrong type. Expected matplotlib.axes.Axes3D, got %s" % (type(axis)))

    p_f = np.array([ [ 0,0,0,1], [size,0,0,1], [0,size,0,1], [0,0,size,1]]).T;
    p_0 = np.dot(T_0f,p_f)

    X = np.append([p_0[:,0].T], [p_0[:,1].T], axis=0 )
    Y = np.append([p_0[:,0].T], [p_0[:,2].T], axis=0 )
    Z = np.append([p_0[:,0].T], [p_0[:,3].T], axis=0 )
    axis.plot3D(X[:,0],X[:,1],X[:,2], 'r-', linewidth=linewidth)
    axis.plot3D(Y[:,0],Y[:,1],Y[:,2], 'g-', linewidth=linewidth)
    axis.plot3D(Z[:,0],Z[:,1],Z[:,2], 'b-', linewidth=linewidth)

    if name is not None:
        axis.text(X[0,0],X[0,1],X[0,2], name, zdir='x')
