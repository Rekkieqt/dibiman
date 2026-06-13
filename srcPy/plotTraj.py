import matplotlib.pyplot as plt
import numpy as np


def plotTraj(data):
    x = data['x']
    t = data['t']
    xref = data['xref']
    N = data['N']

    X = np.array(x)
    Xref = np.tile(xref[:, np.newaxis], (1, N + 1))
    
    plt.plot(t, X)
    plt.plot(t, Xref.T, '--')
    plt.show()

