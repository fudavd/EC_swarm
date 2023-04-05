import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.ndimage import rotate

matplotlib.use('Qt5Agg')

class swarm_plotter:

    def __init__(self, arena: str = "circle_30x30"):
        self.xleftlimit = 0
        self.yleftlimit = 0
        self.xrightlimit = int(re.findall('\d+', arena)[-1])
        self.yrightlimit = int(re.findall('\d+', arena)[-1])

        self.mapp = sio.loadmat(f'./utils/Gradient Maps/{arena}.mat')
        self.mapp = self.mapp['I']
        self.mapp = rotate(self.mapp, angle=90)
        self.size_x = int(re.findall('\d+', arena)[-1])
        self.size_y = int(re.findall('\d+', arena)[-1])

        plt.ion()
        plt.show()

    def plot_swarm_quiver(self,positions,headings):
        plt.imshow(self.mapp, extent=[0, self.size_x, 0, self.size_y])
        plt.axis([self.xleftlimit, self.xrightlimit, self.yleftlimit, self.yrightlimit])
        plt.scatter(positions[0], positions[1], c='#ff7f0e')
        plt.quiver(positions[0], positions[1], np.cos(headings), np.sin(headings))

        # plt.draw()
        plt.pause(0.001)
        plt.clf()
