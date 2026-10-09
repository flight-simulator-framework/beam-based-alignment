import numpy as np

try:
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
except ImportError:
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)


class MeasureDispersionWakefields(QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)

        titles = {
            "dispersion_energy": "Energy-change test orbit",
            "dispersion_quadrupole_scaling": "Quadrupole-scaling test orbit",
            "wakefields": "Intensity-change test orbit",
        }

        title = titles[result["measurement_type"]]
        self.setWindowTitle(title)
        self.resize(1100, 750)

        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(10, 6), layout="constrained")
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)

        axes = self.figure.subplots(2, 1, sharex=True)
        positions = np.arange(len(result["bpm_names"]))

        for ax, key, plane, color in zip(axes, ("x_samples", "y_samples"), ("X", "Y"), ("tab:red", "tab:blue")):
            samples = np.ma.masked_invalid(
                np.asarray(result[key], dtype=float)
            )
            mean = samples.mean(axis=0).filled(np.nan)
            spread = samples.std(axis=0, ddof=1).filled(np.nan)
            errors = spread if result["n_measurements"] > 1 else None
            ax.errorbar(positions, mean, yerr=errors, fmt="o-", color=color, capsize=3)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_ylabel(f"Δ{plane} [{result['bpm_unit']}]")
            ax.grid(True, alpha=0.3)

        axes[0].set_title(title)
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(
            result["bpm_names"],
            rotation=60,
            ha="right",
        )
        axes[1].set_xlabel("BPM")

        self.canvas.draw_idle()