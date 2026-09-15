try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (QVBoxLayout, QDialog, QLabel, QPlainTextEdit)
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (QVBoxLayout, QDialog, QLabel, QPlainTextEdit)
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use("QtAgg")
import numpy as np

class TestOrbits(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test Orbits")
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setMinimumSize(700, 520)
        self.resize(900, 650)
        self.setSizeGripEnabled(True)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Test Orbits"))
        self.fig_x=Figure(figsize=(5,5),tight_layout=True)
        self.fig_y=Figure(figsize=(5,5),tight_layout=True)
        self.canvas_x=FigureCanvas(self.fig_x)
        self.canvas_y=FigureCanvas(self.fig_y)
        self.axes_x=self.fig_x.add_subplot(111)
        self.axes_y=self.fig_y.add_subplot(111)
        layout.addWidget(self.canvas_x)
        layout.addWidget(self.canvas_y)
    def _plot_test_orbits(self,selected_bpms,O0x,O0y,O1x=None,O1y=None,O2x=None,O2y=None,dfs_err_x=None,dfs_err_y=None):
        #DFS test orbit x=O1x-O0x
        #WFS test orbit x=O2x-O0x
        l_bpms = len(selected_bpms)
        scale = np.arange(l_bpms)

        def flatten(values):
            if values is None:
                return None
            return np.asarray(values, dtype=float).reshape(-1)

        O0x=flatten(O0x)
        O0y=flatten(O0y)
        O1x=flatten(O1x)
        O1y=flatten(O1y)
        O2x=flatten(O2x)
        O2y=flatten(O2y)
        dfs_err_x=flatten(dfs_err_x)
        dfs_err_y=flatten(dfs_err_y)

        if O0x is None or O0y is None:
            return
        xpos=np.arange(len(selected_bpms))

        self.axes_x.clear()
        self.axes_y.clear()

        if O1x is not None:
            self.axes_x.errorbar(xpos, O1x-O0x, yerr=dfs_err_x, fmt="o-", capsize=3, label="DFS (O1-O0)")
        if O2x is not None:
            self.axes_x.plot(xpos, (O2x-O0x),label="WFS (O2-O0)")

        if O1y is not None:
            self.axes_y.errorbar(xpos, O1y-O0y, yerr=dfs_err_y, fmt="o-", capsize=3, label="DFS (O1-O0)")
        if O2y is not None:
            self.axes_y.plot(xpos, (O2y-O0y),label="WFS (O2-O0)")

        self.axes_x.set_xlabel("BPM index") #change to names
        self.axes_x.set_ylabel("Horizontal orbit difference [mm]")
        self.axes_y.set_ylabel("Vertical orbit difference [mm]")
        self.axes_y.set_xlabel("BPM index") #change to names
        self.axes_x.grid(True,alpha=0.3)
        self.axes_y.grid(True,alpha=0.3)
        self.axes_x.set_xticks(xpos)
        self.axes_y.set_xticks(xpos)
        self.axes_x.set_xticklabels(selected_bpms,rotation=90,fontsize=8)
        self.axes_y.set_xticklabels(selected_bpms,rotation=90,fontsize=8)


        if (O1x is not None) or (O2x is not None):
            self.axes_x.legend(fontsize=8,loc="upper right")

        if (O1y is not None) or (O2y is not None):
            self.axes_y.legend(fontsize=8,loc="upper right")

        self.canvas_x.draw_idle()
        self.canvas_y.draw_idle() #not draw immediately, but wait
