import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class LiveGraph(object):
    def __init__(self, maxlen=100, lines=1, labels=None):
        if labels and len(labels) != lines:
            raise ValueError("len(labels) != lines in live_graph")
        self.values = []
        for i in range(lines):
            self.values.append(deque(maxlen=maxlen))
        self.lines = lines
        self.labels = labels
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.maxlen = maxlen

    def show(self):
        self.figure.show()
        self.figure.canvas.draw()

    def add_value(self, new_values):
        self.ax.clear()
        plot_handles = []
        for i in range(self.lines):
            self.values[i].append(new_values[i])
            h, = self.ax.plot(self.values[i])
            plot_handles.append(h)
        if self.labels:
            self.ax.legend(plot_handles, self.labels)

        self.figure.canvas.draw()