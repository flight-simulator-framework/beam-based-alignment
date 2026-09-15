import os
import sys
from datetime import datetime

try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
except ImportError:
    from PyQt5.QtWidgets import QApplication, QMessageBox

from BBA_GUI import MainWindow as CorrectorBBAWindow
from Backend.ActuatorMode import ActuatorMode
from Backend.BBA_helpers.QM_mode_helpers import QM_mode_helpers


class MainWindow(QM_mode_helpers, CorrectorBBAWindow):
    def __init__(self, interface, dir_name, nominal_state=None, start_state=None):
        super().__init__(interface, dir_name, nominal_state, start_state)
        self.actuator_mode = ActuatorMode.QM
        self.qm_corrs = self._quadrupole_movers()
        self._configure_qm_window()

    def _quadrupole_movers(self):
        if not hasattr(self.interface, "get_quadrupole_movers_names"):
            return []
        return [str(name) for name in self.interface.get_quadrupole_movers_names()]

    def _configure_qm_window(self):
        self.setWindowTitle("BBA QM GUI")
        self.actuator_mode_label.setVisible(False)
        self.actuator_mode_combo.setVisible(False)
        self.sextupole_restoration_button.setVisible(False)

        self.correctors_list.clear()
        self.correctors_list.addItems(self.qm_corrs)
        self.groupBox_5.setTitle("Quadrupole movers")

        self._setup_qm_controls()
        self._update_qm_widgets_visibility()
        self.radio_buttons[0].setChecked(True)
        for widget in (
            self.dfs_response_3,
            self.pushButton_9,
            self.mode_dispersion,
            self.wfs_response_3,
            self.pushButton_10,
            self.mode_wakefield,
        ):
            widget.setEnabled(False)
        self._refresh_specific_bpm_candidates()
        self._refresh_metric_plots_for_mode()

    def _get_selection(self):
        selected_corrs = [
            self.correctors_list.item(i).text()
            for i in range(self.correctors_list.count())
            if self.correctors_list.item(i).isSelected()
        ]
        selected_bpms = [
            self.bpms_list.item(i).text()
            for i in range(self.bpms_list.count())
            if self.bpms_list.item(i).isSelected()
        ]
        return selected_corrs or self.qm_corrs, selected_bpms or self.initial_state.get_bpms()["names"]

    def _start_correction(self, silent=False, preserve_plots=False):
        self._start_qm_correction(silent=silent, preserve_plots=preserve_plots)

    def _clear_graphs(self):
        self._cancel = True
        for series in (
            self._hist_orbit_x,
            self._hist_orbit_y,
            self._hist_disp_x,
            self._hist_disp_y,
            self._hist_wake_x,
            self._hist_wake_y,
            self._hist_orbit,
            self._hist_disp,
            self._hist_wake,
            self._hist_orbit_x_err,
            self._hist_orbit_y_err,
            self._hist_orbit_err,
            self._hist_disp_x_err,
            self._hist_disp_y_err,
            self._hist_disp_err,
            self._hist_wake_x_err,
            self._hist_wake_y_err,
            self._hist_wake_err,
        ):
            series.clear()
        self._refresh_metric_plots_for_mode()
        self._refresh_all_plot_popups()

    def _refresh_metric_plots_for_mode(self):
        self.plot_widget_4.setEnabled(False)
        self.plot_widget_5.setEnabled(False)
        self._plot_series(
            self.traj_ax,
            self.traj_canvas,
            [],
            [],
            [],
            title="QM - distance from initial trajectory",
        )
        self._plot_disabled_panel(self.disp_ax, self.disp_canvas, "DFS not used in QM mode")
        self._plot_disabled_panel(self.wake_ax, self.wake_canvas, "WFS not used in QM mode")

    @staticmethod
    def _plot_disabled_panel(ax, canvas, title):
        if canvas is None or ax is None:
            return
        ax.clear()
        ax.set_facecolor("#F0F0F0")
        ax.text(0.5, 0.5, title, transform=ax.transAxes, ha="center", va="center", fontsize=12, color="#777777")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")
        canvas.draw_idle()

    def _display_response_matrix(self):
        orbit_dir = self.trajectory_response_3.text()
        if not orbit_dir:
            QMessageBox.warning(self, "Warning", "No trajectory directory selected")
            return
        self.handling(
            "ComputeResponseMatrix_QM_GUI.py",
            cwd=self.cwd,
            args=["--dir1", orbit_dir, "--compute"],
        )


def main():
    app = QApplication(sys.argv)
    from Backend import SelectInterface

    interface = SelectInterface.choose_acc_and_interface()
    if interface is None:
        return 1

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.expanduser(
        f"~/CERN-Flight_Simulator-Data/BBA_QM_{interface.get_name()}_{time_str}_session_settings"
    )
    window = MainWindow(interface, dir_name, start_state=interface.get_state())
    if hasattr(interface, "log_messages"):
        interface.log_messages(window.log)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
