import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

class ROI_Selector(object):
    def __init__(self, img):
        self.ax = plt.gca()
        self.ax.imshow(img)
        self.rect = Rectangle((0,0), 1, 1, fill=False)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        print('pressed at:',round(self.x0), round(self.y0))

    def on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()
        print('released at:',round(self.x1), round(self.y1))
    
    def get_roi(self):
        if np.array([self.x0, self.x1, self.y0, self.y1]).any() == None:
            return self.x0, self.x1, self.y0, self.y1
        else:
            x_start, x_end, y_start, y_end = round(self.x0), round(self.x1), round(self.y0), round(self.y1)
            return x_start, x_end, y_start, y_end