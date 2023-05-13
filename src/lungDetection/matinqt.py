from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, PatchCollection
import numpy as np

class CFigureCanvas(FigureCanvas):

    def __init__(self, parent=None):
        # Just don’t give figsize and it will fit itself
        fig = Figure(figsize=(12, 9))
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111, projection='3d')
        self.max_values = None
        
        # Enable interactive navigation for 3D plots
        self.navigation_toolbar = True

    def plt_3d(self, verts, faces, alpha=0.7):
        print("Drawing")
        if self.max_values is None:
            self.max_values = np.max(verts, axis=0)
        x, y, z = verts.T
        mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=alpha)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        
        self.axes.add_collection3d(mesh)
        self.axes.set_xlim(0, self.max_values[0])
        self.axes.set_ylim(0, self.max_values[1])
        self.axes.set_zlim(0, self.max_values[2])
        self.axes.set_facecolor((0.7, 0.7, 0.7))

