import copy
import re
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.ndimage import rotate

# matplotlib.use('Qt5Agg')

class swarm_plotter:

    def __init__(self, arena: str = "circle_30x30", colors: List = ['#ff7f0e']):
        self.xleftlimit = 0
        self.yleftlimit = 0
        self.xrightlimit = int(re.findall('\d+', arena)[-1])
        self.yrightlimit = int(re.findall('\d+', arena)[-1])
        self.colors = colors

        self.mapp = sio.loadmat(f'./utils/Gradient Maps/{arena}.mat')
        self.mapp = self.mapp['I']
        self.mapp = rotate(self.mapp, angle=90)
        self.size_x = int(re.findall('\d+', arena)[-1])
        self.size_y = int(re.findall('\d+', arena)[-1])
        self.x_trace = []
        self.y_trace = []

        plt.ion()
        plt.show()

    def plot_swarm_quiver(self, positions, headings, frame_number=None):
        plt.imshow(self.mapp, extent=[0, self.size_x, 0, self.size_y])
        plt.axis([self.xleftlimit, self.xrightlimit, self.yleftlimit, self.yrightlimit])
        self.x_trace.append(copy.deepcopy(positions[0]))
        self.y_trace.append(copy.deepcopy(positions[1]))
        plt.plot(self.x_trace, self.y_trace, zorder=1)
        plt.scatter(positions[0], positions[1], c=self.colors, zorder=2)
        plt.quiver(positions[0], positions[1], np.cos(headings), np.sin(headings), zorder=3)


        plt.draw()
        plt.pause(0.00001)
        if frame_number is not None:
            plt.tight_layout()
            plt.savefig(f"./results/images/plot/{frame_number}.png")
        plt.clf()
