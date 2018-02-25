import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class LiveGraph(object):
    def __init__(self, maxlen=100):
        self.values = deque(maxlen=maxlen)
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.maxlen = maxlen

    def show(self):
        self.figure.show()
        self.figure.canvas.draw()

    def add_value(self, value):
        self.values.append(value)
        self.ax.clear()
        self.ax.plot(self.values)
        self.figure.canvas.draw()