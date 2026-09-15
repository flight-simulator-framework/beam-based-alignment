import sys, os, re, matplotlib
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

try:
    from PyQt6 import uic
    from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
    from PyQt6.QtWidgets import (QGroupBox, QApplication, QRadioButton, QSizePolicy, QMainWindow, QFileDialog,
                                 QListWidget, QListWidgetItem, QMessageBox, QProgressDialog, QVBoxLayout, QPushButton,
                                 QDialog, QLabel, QStyledItemDelegate, QWidget, QHBoxLayout)
    from PyQt6.QtGui import QPainter, QPixmap, QPalette

    pyqt_version = 6
except ImportError:
    from PyQt5 import uic
    from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
    from PyQt5.QtWidgets import (QGroupBox, QApplication, QRadioButton, QSizePolicy, QMainWindow, QFileDialog,
                                 QListWidget, QListWidgetItem, QMessageBox, QProgressDialog, QVBoxLayout, QPushButton,
                                 QDialog, QLabel, QStyledItemDelegate, QWidget, QHBoxLayout)
    from PyQt5.QtGui import QPainter, QPixmap, QPalette

    pyqt_version = 5
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from Backend.LogConsole import LogConsole
from Backend.BBA_helpers.TestOrbits_BBA import TestOrbits
from Backend.SaveOrLoad import SaveOrLoad
from Backend.ResponseMatrix_DFS_WFS import ResponseMatrix_DFS_WFS, AdaptiveResponseMatrix
import matplotlib.pyplot as plt
from Backend.BBA_helpers.BPM_weights import BPM_weights
from Backend.ActuatorMode import ActuatorMode
from traceback import print_exception
from Interfaces.interface_setup import INTERFACE_SETUP
from Knobs.jitter_subtraction import (apply_jitter_subtraction, explain_reference_selection, fit_jitter_model)


class PlotPopup(QMainWindow):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 700)
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        self.fig = Figure(figsize=(9, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)


class BpmWeightsDelegate(QStyledItemDelegate):
    WEIGHTS_ROLE = int(Qt.ItemDataRole.UserRole) + 1

    def paint(self, painter: QPainter, option, index):
        painter.save()
        try:
            opt = option
            self.initStyleOption(opt, index)
            style = opt.widget.style() if opt.widget is not None else None
            if style is not None:
                opt_no_text = opt
                opt_no_text.text = ""
                style.drawControl(style.ControlElement.CE_ItemViewItem, opt_no_text, painter, opt.widget)
            bpm_name = str(index.data(Qt.ItemDataRole.UserRole) or index.data(Qt.ItemDataRole.DisplayRole) or "")
            weights = str(index.data(self.WEIGHTS_ROLE) or "")
            r = opt.rect
            margin = 6
            painter.setFont(opt.font)
            fm = painter.fontMetrics()
            w_w = fm.horizontalAdvance(weights) if weights else 0
            left_rect = r.adjusted(margin, 0, -(w_w + 2 * margin), 0)
            right_rect = r.adjusted(margin, 0, (-margin - 15), 0)
            painter.setPen(opt.palette.color(opt.palette.ColorRole.Text))
            painter.drawText(left_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), bpm_name)
            if weights:
                painter.drawText(right_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight), weights)
        finally:
            painter.restore()


class MainWindow(QMainWindow, SaveOrLoad, ResponseMatrix_DFS_WFS):
    def __init__(self, interface, dir_name, nominal_state=None, start_state=None):
        super().__init__()
        self.cwd = os.getcwd()
        self.interface = interface
        self.dir_name = dir_name
        self.nominal_state = nominal_state
        self.start_state = start_state if start_state is not None else interface.get_state()
        self.restore_state = self.start_state
        self.initial_state = self.start_state
        self.measurement_start_state = None
        self._cancel = False
        self._number_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        self.reset_reference_orbit = False
        ui_path = os.path.join(os.path.dirname(__file__), "UI files/BBA_GUI.ui")
        uic.loadUi(ui_path, self)
        self._clock_zone_name = self._get_clock_zone()
        self._setup_machine_clock()
        self._load_logo()
        self.bpms_list.setItemDelegate(BpmWeightsDelegate(self.bpms_list))
        self._data_dirs = {"traj": None, "dfs": None, "wfs": None}
        self._hist_orbit, self._hist_disp, self._hist_wake = [], [], []
        self._hist_orbit_x, self._hist_orbit_y = [], []
        self._hist_transmission = []
        self._hist_disp_x, self._hist_disp_y = [], []
        self._hist_wake_x, self._hist_wake_y = [], []
        self._hist_orbit_x_err, self._hist_orbit_y_err, self._hist_orbit_err = [], [], []
        self._hist_disp_x_err, self._hist_disp_y_err, self._hist_disp_err = [], [], []
        self._hist_wake_x_err, self._hist_wake_y_err, self._hist_wake_err = [], [], []
        self._hist_abs_rms_x, self._hist_abs_rms_y, self._hist_abs_rms_xy = [], [], []
        self.log_console = None
        self.show_response_matrix = None
        self.test_orbits = None
        self.traj_popup, self.disp_popup, self.wake_popup = None, None, None
        self._setup_canvases()
        self._plot_double_clicks()
        self.bpm_weights = {}
        self._populate_lists()
        self._procs = []
        self.load_correctors_button.clicked.connect(self._load_correctors)
        self.load_bpms_button.clicked.connect(self._load_bpms)
        self.radio_buttons = [self.mode_orbit, self.mode_dispersion, self.mode_wakefield]
        self.pushButton_log.clicked.connect(self._show_console_log)
        self.pushButton_testorb.clicked.connect(self._show_test_orbits)
        self.session_database_3.setText(dir_name)
        self.pushButton_8.clicked.connect(self._pick_and_load_traj_data)
        self.pushButton_9.clicked.connect(self._pick_and_load_disp_data)
        self.pushButton_10.clicked.connect(self._pick_and_load_wake_data)
        self.clear_graphs_button.clicked.connect(self._clear_graphs)
        self.restore_initial_settings.clicked.connect(self._restore_initial_settings)
        self.modes = ['Orbit', 'Dispersion', 'Wakefield']
        self._running = False
        self.appropriate_settings_energy = None
        self.appropriate_settings_intensity = None
        self.appropriate_settings_reset_e = None
        self.appropriate_settings_reset_ch = None
        self.start_button.clicked.connect(self._on_start_click)
        self.stop_button.clicked.connect(self._stop_correction)
        self.corrs = self.initial_state.get_correctors()["names"]
        self.jitter_model = None
        self.subtract_jitter_checkbox.setChecked(False)
        self.actuator_mode = ActuatorMode.Kicker
        self._setup_corrector_controls()
        self._setup_beam_change_controls()
        self.restore_machine_status_button.clicked.connect(self._pick_and_load_machine_status_file)
        self.initial_charge_value = None
        self.orbit_at_first_start_click_x = None
        self.orbit_at_first_start_click_y = None

    def _save_machine_status(self):
        saved_at = self._clock_now()
        time_str = saved_at.strftime("%y%m%d%H%M%S")
        default_dir = os.path.expanduser(os.path.expandvars("~/CERN-Flight_Simulator-Data/"))
        self._session_dir = os.path.join(default_dir, f"BBA_{self.interface.get_name()}{time_str}_session_settings")
        os.makedirs(self._session_dir, exist_ok=True)
        machine_state = self.interface.get_state()
        machine_state.timestamp = saved_at
        machine_state.save(filename=os.path.join(self._session_dir, "machine_status.pkl"))
        self.session_database_3.setText(self._session_dir)
        return machine_state

    def _get_clock_zone(self):
        interface_defaults = self._get_interface_initial_settings() or {}
        timezone = interface_defaults.get("clock_timezone", "Europe/Zurich")
        return timezone

    def _clock_now(self):
        return datetime.now(ZoneInfo(self._clock_zone_name))

    def _setup_machine_clock(self):
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(250)
        self._clock_timer.timeout.connect(self._update_machine_clock)
        self._update_machine_clock()
        self._clock_timer.start()

    def _update_machine_clock(self):
        now = self._clock_now()
        self.machine_clock_label.setText(f"{now:%Y-%m-%d %H:%M:%S} {now.tzname()}")

    def _setup_corrector_controls(self):
        self.groupBox_9.setVisible(False)
        self.actuator_mode_label.setVisible(False)
        self.actuator_mode_combo.setVisible(False)
        self.pushButton_11.clicked.connect(self.load_session_settings)
        self.setWindowTitle("BBA GUI")
        self.lineEdit.setText("1")
        self.lineEdit_2.setText("10")
        self.lineEdit_3.setText("10")
        self.lineEdit_4.setText("0.001")
        self.lineEdit_5.setText("10")
        self.lineEdit_6.setText("0.4")
        self.lineEdit_beta.setText("0")
        self.transmission_value.setText("0.65")
        self._setup_nsamples_control()
        self.compute_response_matrix_button.clicked.connect(self._display_response_matrix)
        self.pushButton_reset_ref_orbit.clicked.connect(self._reset_reference_orbit)
        self.reset_ref_orb = False
        self.bpms_list.itemDoubleClicked.connect(self._edit_bpm_weights)
        correctors = self.interface.get_correctors()
        correctors_list = correctors['names']
        self.hcorrector_names = set(map(str, self.interface.get_hcorrectors_names() or []))  # takes correctors names, if None, then use an empty list, makes everything a string and saves as a set without the duplicates
        self.vcorrector_names = set(map(str, self.interface.get_vcorrectors_names() or []))
        units_settings, sysid_kick, bpm_unit, corrs_unit = self._get_interface_units()
        self.sysid_kick = sysid_kick
        self.bpm_unit = bpm_unit
        self.corrs_unit = corrs_unit
        max_curr_h = 0.0
        max_curr_v = 0.0
        if correctors_list is not None:
            hcorrs = self.interface.get_hcorrectors_names()
            vcorrs = self.interface.get_vcorrectors_names()
            hcorr_indexes = np.array([index for index, string in enumerate(correctors_list) if string in hcorrs])
            vcorr_indexes = np.array([index for index, string in enumerate(correctors_list) if string in vcorrs])

            def clean_array(a):
                a = np.array([0 if x is None else x for x in a], dtype=float)
                a[np.isnan(a)] = 0
                return a

            max_curr_h = 1.15 * np.max(np.abs(clean_array(np.array(correctors['bdes'])[hcorr_indexes])))
            max_curr_v = 1.15 * np.max(np.abs(clean_array(np.array(correctors['bdes'])[vcorr_indexes])))
            if "bba_max_h_strength" in units_settings: max_curr_h = units_settings["bba_max_h_strength"]
            if "bba_max_v_strength" in units_settings: max_curr_v = units_settings["bba_max_v_strength"]
        self.max_horizontal_current_spinbox.setValue(max_curr_h)
        self.max_horizontal_current_spinbox.setSingleStep(0.01)
        self.max_vertical_current_spinbox.setValue(max_curr_v)
        self.max_vertical_current_spinbox.setSingleStep(0.01)

        self._refresh_metric_plots_for_mode()

    def _setup_nsamples_control(self):
        self.nsamples_input.setText(str(max(1, int(self.interface.nsamples))))
        self.nsamples_input.textChanged.connect(self._set_interface_nsamples)

    def _set_interface_nsamples(self, value):
        self.interface.nsamples = max(1, int(value))
        self.nsamples_input.setStyleSheet("")

    def _load_logo(self):
        self.logo_label.setText("")
        self.logo_label.setScaledContents(False)
        transform_mode = (
            Qt.TransformationMode.SmoothTransformation
            if pyqt_version == 6
            else Qt.SmoothTransformation
        )
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(base_dir, "UI files", "Assets", "CERN_logo.png")
        if not os.path.isfile(logo_path):
            return
        pixmap = QPixmap(logo_path)
        if pixmap.isNull():
            return
        scaled = pixmap.scaledToHeight(80, transform_mode)
        self.logo_label.setPixmap(scaled)
        self.logo_label.setToolTip(logo_path)

    def _plot_double_clicks(self):
        if getattr(self, "traj_canvas", None) is not None:  # mpl connect watches/listens to events in matplotlib popup
            self.traj_canvas.mpl_connect("button_press_event", lambda event: self._handle_plot_click(event, "traj"))
        if getattr(self, "disp_canvas", None) is not None:
            self.disp_canvas.mpl_connect("button_press_event", lambda event: self._handle_plot_click(event, "disp"))
        if getattr(self, "wake_canvas", None) is not None:
            self.wake_canvas.mpl_connect("button_press_event", lambda event: self._handle_plot_click(event, "wake"))

    def _handle_plot_click(self, event, plot_type):
        if event is None: return
        if getattr(event, "dblclick", False) and getattr(event, "button", None) == 1:
            self._open_plot_popup(plot_type)

    def _open_plot_popup(self, plot_type):
        popup_attr = f"{plot_type}_popup"
        popup = getattr(self, popup_attr, None)
        if popup is None:
            titles = {
                "traj": "Trajectory (Residual to response matrix reference)",
                "disp": "Dispersion (Residual to response matrix reference)",
                "wake": "Wakefield (Residual to response matrix reference)",
            }
            popup = PlotPopup(titles.get(plot_type, "Correction plot"), parent=self)
            setattr(self, popup_attr, popup)
        self._refresh_plot_popup(plot_type)
        popup.show()
        popup.raise_()
        popup.activateWindow()

    def _refresh_plot_popup(self, plot_kind):
        popup = getattr(self, f"{plot_kind}_popup", None)
        if popup is None:
            return
        if plot_kind == "traj":
            values_x = self._hist_orbit_x
            values_y = self._hist_orbit_y
            vals = self._hist_orbit
            error_x = self._hist_orbit_x_err
            error_y = self._hist_orbit_y_err
            error_all = self._hist_orbit_err
            ylabel = f"Residual norm [{self.bpm_unit}]"
            title = "Trajectory (Residual to response matrix reference)"
        elif plot_kind == "disp":
            values_x = self._hist_disp_x
            values_y = self._hist_disp_y
            vals = self._hist_disp
            error_x = self._hist_disp_x_err
            error_y = self._hist_disp_y_err
            error_all = self._hist_disp_err
            ylabel = f"Residual norm [{self.bpm_unit}]"
            title = "Dispersion (Residual to response matrix reference)"
        else:
            values_x = self._hist_wake_x
            values_y = self._hist_wake_y
            vals = self._hist_wake
            error_x = self._hist_wake_x_err
            error_y = self._hist_wake_y_err
            error_all = self._hist_wake_err
            ylabel = f"Residual norm [{self.bpm_unit}]"
            title = "Wakefield (Residual to response matrix reference)"

        ax = popup.ax
        ax.clear()

        if values_x:
            err_x = error_x if error_x is not None else None
            ax.errorbar(range(len(values_x)), values_x, yerr=err_x, marker="o", color="red", label="x", capsize=6,
                        elinewidth=2, capthick=2, markersize=4)
        if values_y:
            err_y = error_y if error_y is not None else None
            ax.errorbar(range(len(values_y)), values_y, yerr=err_y, marker="o", color="blue", label="y", capsize=6,
                        elinewidth=2, capthick=2, markersize=4)
        if vals:
            err_all = error_all if error_all is not None else None
            ax.errorbar(range(len(vals)), vals, yerr=err_all, linestyle="dashed", color="black", label="combined norm",
                        capsize=6, elinewidth=2, capthick=2, markersize=4)

        if values_x or values_y or vals:
            ax.legend(fontsize=9, loc="upper right")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        popup.canvas.draw_idle()

    def _refresh_all_plot_popups(self):
        self._refresh_plot_popup("traj")
        self._refresh_plot_popup("disp")
        self._refresh_plot_popup("wake")

    def _get_interface_initial_settings(self):
        interface_class_name = self.interface.__class__.__name__
        interface_module_name = self.interface.__class__.__module__

        for machine_interfaces in INTERFACE_SETUP.values():
            for interface_defaults in machine_interfaces:
                if (interface_defaults.get("class_name") == interface_class_name) and (
                        interface_defaults.get("module") == interface_module_name):
                    return interface_defaults
        return None

    def _get_interface_units(self):
        interface_defaults = self._get_interface_initial_settings()
        if interface_defaults is None:
            return {}, 0.01, "mm", ""
        units_settings = interface_defaults.get("units", {})
        sysid_kick = units_settings.get("sysid_corrector_kick", 0.01)
        bpm_unit = units_settings.get("bpm_position", "mm")
        corrs_unit = units_settings.get("corrector_strength", "T*mm")

        return units_settings, sysid_kick, bpm_unit, corrs_unit

    def _setup_beam_change_controls(self):
        beam_change = (self._get_interface_initial_settings() or {}).get("beam_change", {})
        self._beam_change_fields = []
        self.beam_change_group.setVisible(bool(beam_change))

        controls = {
            "energy": (
                self.energy_change_label,
                self.energy_nominal_label,
                self.energy_nominal_input,
                self.energy_test_label,
                self.energy_test_input,
                self.energy_change_tooltip,
            ),
            "intensity": (
                self.intensity_change_label,
                self.intensity_nominal_label,
                self.intensity_nominal_input,
                self.intensity_test_label,
                self.intensity_test_input,
                self.intensity_change_tooltip,
            ),
        }
        for kind, widgets in controls.items():
            title, nominal_label, nominal_input, test_label, test_input, tooltip = widgets
            settings = beam_change.get(kind)
            title.setVisible(settings is not None)
            tooltip.setVisible(settings is not None)
            if settings is None:
                for widget in (nominal_label, nominal_input, test_label, test_input):
                    widget.setVisible(False)
                continue
            title.setText(settings["label"])
            tooltip.setToolTip(settings["tooltip"])
            for slot, label, input_widget in (
                    ("nominal", nominal_label, nominal_input),
                    ("test", test_label, test_input),
            ):
                field = settings.get(slot)
                label.setVisible(field is not None)
                input_widget.setVisible(field is not None)
                if field is None:
                    continue
                label.setText(field["label"])
                value = getattr(self.interface, field["attribute"], field.get("default", ""))
                input_widget.setText("" if value is None else str(value))
                input_widget.setPlaceholderText(field.get("placeholder", ""))
                self._beam_change_fields.append((input_widget, field))

    def _apply_beam_change_controls(self):
        self._beam_change_values = {}
        for input_widget, field in self._beam_change_fields:
            text = input_widget.text().strip()
            if not text and field.get("allow_empty", False):
                setattr(self.interface, field["attribute"], None)
                self._beam_change_values[field["attribute"]] = None
                input_widget.setStyleSheet("")
                continue
            try:
                value = float(text)
            except ValueError:
                input_widget.setStyleSheet("QLineEdit { border: 1px solid #c62828; }")
                QMessageBox.warning(self, "Invalid beam-change setting", f"{field['label']} must be a number.")
                return False
            input_widget.setStyleSheet("")
            setattr(self.interface, field["attribute"], value)
            self._beam_change_values[field["attribute"]] = value
        return True

    def _load_beam_change_values(self, values):
        for input_widget, field in self._beam_change_fields:
            attribute = field["attribute"]
            if attribute not in values:
                continue
            value = values[attribute]
            input_widget.setText("" if value is None else str(value))
        return self._apply_beam_change_controls()

    def _restore_initial_settings(self):
        self.log("Restoring initial settings...")
        self._cancel = True
        self._running = False
        w1, w2, w3, rcond, iters, gain, beta, transmission_threshold= self._read_params()
        try:
            if w2 > 0:
                self.interface.reset_energy()
            if w3 > 0:
                self.interface.reset_intensity()
        except Exception:
            self.log(f"The machine wasn't restored to its nominal state.")
            QMessageBox.critical(self, "Restore error",
                                 f"Could not confirm the machine returned to its nominal energy/intensity.")
        if self.interface.restore_correctors_state(self.restore_state) is False:
            self.log("Warning: not every corrector was confirmed back at its saved current.")
            QMessageBox.warning(
                self, "Restore initial settings",
                "Some correctors were not confirmed back at their saved current within the "
                "readback tolerance. Check them on the machine before the next correction.",
            )
        self.reset_ref_orb = True
        self._hist_abs_rms_x.clear(), self._hist_abs_rms_y.clear(), self._hist_abs_rms_xy.clear()
        self._clear_graphs()
        self.log("Machine initial settings restored.")

    def _is_h_corrector(self, s):
        return str(s) in self.hcorrector_names

    def _is_v_corrector(self, s):
        return str(s) in self.vcorrector_names

    def _edit_bpm_weights(self, bpm):
        bpm_name = bpm.data(Qt.ItemDataRole.UserRole) or (bpm.text() or "")
        if not bpm_name:
            bpm_name = (bpm.text() or "").split("    [", 1)[0]
        bpm_window = BPM_weights(bpm_name=bpm_name, parent=self)
        if bpm_name in self.bpm_weights:
            w_orb, w_dfs, w_wfs = self.bpm_weights[bpm_name]
            bpm_window.set_values(w_orb, w_dfs, w_wfs)

        if bpm_window.exec() == QDialog.DialogCode.Accepted:
            self.bpm_weights[bpm_name] = bpm_window.get_values()
            self._update_bpm_weights(bpm)

    def _on_start_click(self):
        print("Starting button clicked...")
        self.log("Starting button clicked...")
        if not self._running:
            if not self._apply_beam_change_controls():
                return
            saved_state = self._save_machine_status()
            self._running = True
            self._step = True
            completed = False
            self._start_correction()
        else:
            self._step = True

    def _expand_data_path(self, path):
        home = os.path.expanduser("~")
        if path.startswith(home + os.sep):  # the character used by the operating system to separate pathname components
            return "~" + path[len(home):]
        return path

    def _setup_canvases(self):
        if FigureCanvas is None:
            self.traj_canvas = self.disp_canvas = self.wake_canvas = None
            self.traj_ax = self.disp_ax = self.wake_ax = None
            return

        def install(host):
            fig = Figure(figsize=(5, 2.4), tight_layout=True)
            canvas = FigureCanvas(fig)
            layout = host.layout()
            if layout is None:
                from PyQt6.QtWidgets import QVBoxLayout
                layout = QVBoxLayout(host)
                layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(canvas)
            ax = fig.add_subplot(111)
            return fig, canvas, ax

        self.traj_fig, self.traj_canvas, self.traj_ax = install(self.plot_widget_3)
        self.disp_fig, self.disp_canvas, self.disp_ax = install(self.plot_widget_4)
        self.wake_fig, self.wake_canvas, self.wake_ax = install(self.plot_widget_5)

    def _plot_series(self, ax, canvas, values_x, values_y, vals, title=None, error_x=None, error_y=None,
                     error_all=None):
        ylabel = f"Residual norm [{self.bpm_unit}]"
        if canvas is None or ax is None:
            return
        if getattr(self, "_suppress_main_plots", False):
            return
        if ax is self.traj_ax and getattr(self, "traj_transmission_ax", None) is not None:
            self.traj_transmission_ax.remove()
            self.traj_transmission_ax = None
        ax.clear()

        def matching_yerr(errors, values):
            if errors is None:
                return None
            try:
                if len(errors) == 0:
                    return None
                if len(errors) != len(values):
                    return None
            except TypeError:
                return errors
            return errors

        if values_x:
            err_x = matching_yerr(error_x, values_x)
            ax.errorbar(range(len(values_x)), values_x, yerr=err_x, marker="o", color='blue', label="x", capsize=6,
                        elinewidth=2, capthick=2,
                        markersize=4)  # yerr - height of the error bar on the plot, capsize - size of the top line on the error bar
        if values_y:
            err_y = matching_yerr(error_y, values_y)
            ax.errorbar(range(len(values_y)), values_y, yerr=err_y, marker="o", color='red', label="y", capsize=6,
                        elinewidth=2, capthick=2, markersize=4)
        if vals:
            err_all = matching_yerr(error_all, vals)
            ax.errorbar(range(len(vals)), vals, yerr=err_all, linestyle="dashed", color='black', label="combined norm",
                        capsize=6, elinewidth=2, capthick=2, markersize=4)
        if ax is self.traj_ax and self._hist_transmission:
            transmission = np.asarray(self._hist_transmission, dtype=float)
            if np.any(np.isfinite(transmission)):
                self.traj_transmission_ax = ax.twinx()
                self.traj_transmission_ax.plot(range(len(transmission)), transmission, "g--", label="Transmission")
                self.traj_transmission_ax.set_ylabel("Transmission [%]", color="green")
                self.traj_transmission_ax.set_ylim(bottom=0.0)
                self.traj_transmission_ax.tick_params(axis="y", colors="green")
                self.traj_transmission_ax.legend(fontsize=7, loc="lower right")
        if values_x or values_y:
            ax.legend(fontsize=7, loc="upper right")
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel("Iteration", fontsize=10)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.yaxis.get_offset_text().set_fontsize(10)
        ax.grid(True, alpha=0.3)
        canvas.draw_idle()

        if ax is self.traj_ax:
            self._refresh_plot_popup("traj")
        elif ax is self.disp_ax:
            self._refresh_plot_popup("disp")
        elif ax is self.wake_ax:
            self._refresh_plot_popup("wake")

    def _get_bpm_weights_text(self, bpm_name):
        wbpm_orb, wbpm_dfs, wbpm_wfs = self.bpm_weights.get(bpm_name, (1.0, 1.0, 1.0))
        return f"[w1 = {wbpm_orb:g}, w2 = {wbpm_dfs:g}, w3 = {wbpm_wfs:g}]"  # general format, removes reduntant zeros at the end etc.

    def _update_bpm_weights(self, item):
        bpm_name = item.data(Qt.ItemDataRole.UserRole) or (item.text() or "")  # it gives a clean name of the item, even if there is another text (like weights)
        item.setData(BpmWeightsDelegate.WEIGHTS_ROLE, self._get_bpm_weights_text(bpm_name))
        item.setText(bpm_name)

    def _populate_lists(self):
        corrs = self.initial_state.get_correctors()["names"]
        bpms = self.initial_state.get_bpms()["names"]
        self.correctors_list.insertItems(0, corrs)
        for bpm_name in bpms:
            bpm_name = str(bpm_name)
            item = QListWidgetItem(bpm_name)
            item.setData(Qt.ItemDataRole.UserRole, bpm_name)
            item.setData(BpmWeightsDelegate.WEIGHTS_ROLE, self._get_bpm_weights_text(bpm_name))
            self.bpms_list.addItem(item)

    def _get_selection(self):
        corrs_all = self.initial_state.get_correctors()["names"]
        bpms_all = self.initial_state.get_bpms()["names"]

        selected_corrs = []
        for i in range(self.correctors_list.count()):
            it = self.correctors_list.item(i)
            if it.isSelected():
                selected_corrs.append(it.text())
        corrs = selected_corrs or corrs_all

        selected_bpms = []
        for i in range(self.bpms_list.count()):
            it = self.bpms_list.item(i)
            if it.isSelected():
                selected_bpms.append(it.data(Qt.ItemDataRole.UserRole))
        bpms = selected_bpms or bpms_all

        return corrs, bpms

    def _force_triangular(self) -> bool:
        return bool(self.triangular_checkbox.isChecked())

    def _with_progress(self, total, title):
        prog = QProgressDialog(title, "Cancel", 0, total, self)
        prog.setWindowModality(Qt.WindowModality.ApplicationModal)  # user cant interact with the main window
        prog.setMinimumDuration(0)

        def cb(i, n, text):
            prog.setMaximum(n)
            prog.setValue(i)
            prog.setLabelText(text)
            QApplication.processEvents()
            return not prog.wasCanceled()

        return prog, cb

    def _calc_error(self, x_meas, y_meas, ref_x, ref_y, disp_x=None, disp_y=None):
        x_val = np.asarray(x_meas, dtype=float)
        y_val = np.asarray(y_meas, dtype=float)
        meanx = np.mean(x_val, axis=0)
        meany = np.mean(y_val, axis=0)
        faulty = (meanx == 0.0) & (meany == 0.0)
        if np.any(faulty):
            x_val[:, faulty] = np.nan
            y_val[:, faulty] = np.nan
        ref_x = np.asarray(ref_x, dtype=float).reshape(1, -1)
        ref_y = np.asarray(ref_y, dtype=float).reshape(1, -1)

        if disp_x is None:
            disp_x = 0
        if disp_y is None:
            disp_y = 0

        disp_x = np.asarray(disp_x, dtype=float).reshape(1, -1)
        disp_y = np.asarray(disp_y, dtype=float).reshape(1, -1)

        dx = x_val - ref_x - disp_x
        dy = y_val - ref_y - disp_y

        dx = np.nan_to_num(dx, nan=0, posinf=0, neginf=0)
        dy = np.nan_to_num(dy, nan=0, posinf=0, neginf=0)

        x_norm = np.linalg.norm(dx, axis=1)
        y_norm = np.linalg.norm(dy, axis=1)
        all_norm = x_norm + y_norm

        def std_calc(a):
            a = np.asarray(a, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan"), float("nan")
            if a.size == 1:
                return float(a[0]), 0
            return float(np.mean(a)), float(np.std(a, ddof=1))

        mean_x, std_x = std_calc(x_norm)
        mean_y, std_y = std_calc(y_norm)
        mean_all, std_all = std_calc(all_norm)
        return mean_x, mean_y, std_x, std_y, mean_all, std_all

    def _read_params(self):
        def getf(name, default):  # gets the text value and turns it into a float
            w = getattr(self, name, None)
            if w is None:
                return float(default)

            txt = (w.text() or "").strip()
            if txt:
                try:
                    return float(txt)
                except ValueError:
                    pass
            w.setText(f"{default:g}")
            return float(default)

        def geti(name, default):  # the same, but to an int
            w = getattr(self, name, None)
            if w is None:
                return int(default)

            txt = (w.text() or "").strip()
            if txt:
                try:
                    return int(float(txt))
                except ValueError:
                    pass
            w.setText(str(int(default)))
            return int(default)

        orbit_w = getf("lineEdit", 1.0)
        disp_w = getf("lineEdit_2", 10.0)
        wake_w = getf("lineEdit_3", 10.0)
        rcond = getf("lineEdit_4", 0.001)
        iters = geti("lineEdit_5", 10)
        gain = getf("lineEdit_6", 0.4)
        beta = getf("lineEdit_beta", 0.0)
        transmission_threshold = getf("transmission_value", 0.65) * 100
        return orbit_w, disp_w, wake_w, rcond, iters, gain, beta, transmission_threshold

    def _reset_reference_orbit(self):
        self.reset_ref_orb = True
        self.log("Resetting reference orbit")

    def _build_jitter_model_for_correction(self, actuators, bpms):
        if not self.subtract_jitter_checkbox.isChecked():
            self.jitter_model = None
            return None

        sequence = self.interface.get_sequence()
        refs, reason = explain_reference_selection(bpms, actuators, sequence, min_refs=2)

        if reason:
            self.log(f"Jitter subtraction disabled: {reason}")
            self.jitter_model = None
            return None

        old_nsamples = getattr(self.interface, "nsamples", None)
        fit_nsamples = 300

        try:
            if old_nsamples is not None:
                self.interface.nsamples = fit_nsamples
            bpms_snapshot = self.interface.get_bpms()
        finally:
            if old_nsamples is not None:
                self.interface.nsamples = old_nsamples

        targets = [str(bpm) for bpm in bpms if str(bpm) not in set(refs)]

        model, fit_reason = fit_jitter_model(bpms_list=[bpms_snapshot], reference_bpms=refs, target_bpms=targets)

        if model is None:
            self.log(f"Jitter subtraction disabled: {fit_reason}")
            self.jitter_model = None
            return None

        self.jitter_model = model
        self.log(
            "Jitter subtraction enabled with refs: "
            + ", ".join(refs)
            + f"; fitted from {fit_nsamples} fixed-config BPM samples"
        )
        return model

    def _start_correction(self, silent=False, preserve_plots=False):
        try:
            plot_snapshot = None
            if preserve_plots:
                plot_snapshot = {
                    "orbit_x": list(self._hist_orbit_x),
                    "orbit_y": list(self._hist_orbit_y),
                    "orbit": list(self._hist_orbit),
                    "disp_x": list(self._hist_disp_x),
                    "disp_y": list(self._hist_disp_y),
                    "disp": list(self._hist_disp),
                    "wake_x": list(self._hist_wake_x),
                    "wake_y": list(self._hist_wake_y),
                    "wake": list(self._hist_wake),
                    "orbit_x_err": list(self._hist_orbit_x_err),
                    "orbit_y_err": list(self._hist_orbit_y_err),
                    "orbit_err": list(self._hist_orbit_err),
                    "disp_x_err": list(self._hist_disp_x_err),
                    "disp_y_err": list(self._hist_disp_y_err),
                    "disp_err": list(self._hist_disp_err),
                    "wake_x_err": list(self._hist_wake_x_err),
                    "wake_y_err": list(self._hist_wake_y_err),
                    "wake_err": list(self._hist_wake_err),
                    "abs_rms_x": list(self._hist_abs_rms_x),
                    "abs_rms_y": list(self._hist_abs_rms_y),
                    "abs_rms_xy": list(self._hist_abs_rms_xy),
                }
            corrs, bpms = self._get_selection()

            self._build_jitter_model_for_correction(actuators=corrs, bpms=bpms)
            if self.jitter_model is not None:
                refs = set(self.jitter_model["reference_bpms"])
                bpms = [bpm for bpm in bpms if bpm not in refs]
                self.log("Removed jitter reference BPMs from correction targets")
            print("Starting correction...")
            self.log("Starting correction...")

            self._cancel = False
            w1, w2, w3, rcond, iters, gain, beta, transmission_threshold = self._read_params()
            wgt_orb, wgt_dfs, wgt_wfs = w1, w2, w3
            Cx = [s for s in corrs if self._is_h_corrector(s)]
            Cy = [s for s in corrs if self._is_v_corrector(s)]

            Axx, Ayy, Axy, Ayx, B0x, B0y, hcorrs, vcorrs, bpms_common = self._creating_response_matrices(selected_corrs=corrs, selected_bpms=bpms)
            print("hcorrs order =", hcorrs)
            print("vcorrs order =", vcorrs)
            print("Cx from GUI =", Cx)
            print("Cy from GUI =", Cy)

            Axx_base = np.array(Axx, copy=True)
            Ayy_base = np.array(Ayy, copy=True)
            Axy_base = np.array(Axy, copy=True)
            Ayx_base = np.array(Ayx, copy=True)

            if self.remove_coupling_checkbox.isChecked():
                Axy_base.fill(0.0)
                Ayx_base.fill(0.0)

            bpms = list(bpms_common)

            n = len(bpms)
            wbpm_orb_vec, wbpm_dfs_vec, wbpm_wfs_vec = [], [], []
            for bpm in bpms:
                wbpm_orb, wbpm_dfs, wbpm_wfs = self.bpm_weights.get(bpm, (1.0, 1.0, 1.0))
                wbpm_orb_vec.append(wbpm_orb)
                wbpm_dfs_vec.append(wbpm_dfs)
                wbpm_wfs_vec.append(wbpm_wfs)
            wbpm_orb_vec = np.array(wbpm_orb_vec, dtype=float).reshape(n, 1)
            wbpm_dfs_vec = np.array(wbpm_dfs_vec, dtype=float).reshape(n, 1)
            wbpm_wfs_vec = np.array(wbpm_wfs_vec, dtype=float).reshape(n, 1)

            w_parts = []
            if w1 > 0: w_parts.append(wbpm_orb_vec)
            if w2 > 0: w_parts.append(wbpm_dfs_vec)
            if w3 > 0: w_parts.append(wbpm_wfs_vec)

            W_x = np.vstack(w_parts)
            W_xy = np.vstack([W_x, W_x])  # because the weights are the same, B=vstack(bx,by)
            # W_xy=np.clip(W_xy, 0, 25) #idk, maybe later there's a need for clamp
            w_xy_bpms = np.sqrt(W_xy)

            self.setWindowTitle("BBA GUI - [Correction running]")
            target_disp_x, target_disp_y = self.interface.get_target_dispersion(bpms)
            max_curr_h = self.max_horizontal_current_spinbox.value()  # gauss * m
            max_curr_v = self.max_vertical_current_spinbox.value()  # gauss * m

            def clamp(val, max_val):
                val = np.asarray(val, dtype=float)
                max_val = np.asarray(max_val, dtype=float)
                result = val.copy()
                finite = np.isfinite(max_val) & (max_val > 0.0)
                result[finite] = np.clip(result[finite], -max_val[finite],
                                         max_val[finite])  # clip limits the values in an array
                return result

            plt.ion()

            self.use_adaptive_R = getattr(self, "use_adaptive_R", True)
            self._adaptive_R = getattr(self, "_adaptive_R", None)
            self._adaptive_R_prev_kick = None
            self._adaptive_R_prev_orbit = None
            adaptive_orbit_only = self.use_adaptive_R and w1 > 0 and w2 == 0 and w3 == 0
            if not adaptive_orbit_only:
                self._adaptive_R = None
            if w2 > 0:
                self.interface.reset_energy()
            if w3 > 0:
                self.interface.reset_intensity()

            samples_dir = os.path.join(self._session_dir, "BBA_states")
            os.makedirs(samples_dir, exist_ok=True)
            last_completed_iteration = None
            prev_Dx = None
            prev_Dy = None
            prev_applied_kick = None

            for it in range(iters):
                if self._cancel:
                    break
                self._step = False

                # nominal
                print("Measuring orbit")
                self.log("Measuring orbit")
                state0 = self.interface.get_state()
                state0 = self._apply_jitter_subtraction_to_state(state0)
                nominal_file = os.path.join(samples_dir, f"ITER_{it:04d}_nominal.pkl")
                state0.save(filename=nominal_file)
                if hasattr(self.interface, "chosen_ict"):
                    charge = np.asarray(state0.get_icts(self.interface.chosen_ict)["charge"], dtype=float).ravel()
                    if charge.size and np.isfinite(charge[0]) and charge[0] != 0:
                        if self.initial_charge_value is None:
                            self.initial_charge_value = charge[0]
                        reference = float(self.initial_charge_value)
                        self._hist_transmission.append(100.0 * charge[0] / reference)
                        if self._hist_transmission[-1] <= transmission_threshold:
                            self._stop_correction()
                            QMessageBox.warning(self, "Transmission below level!",
                                                "Transmission has reached the threshold. Stoping the correction now, and leaving correctors at current values.")
                    else:
                        self._hist_transmission.append(np.nan)
                if it == 0:
                    self.measurement_start_state = state0
                    screens0 = state0.get_screens()
                    print("Screen values before correction:")
                    print(f"Sigx: {screens0['sigx']}")
                    print(f"Sigy: {screens0['sigy']}")
                    print("Emittance before correction:")
                    for screen_name in screens0["names"]:
                        if hasattr(self.interface, "get_twiss_at_screen"):
                            tw = self.interface.get_twiss_at_screen(screen_name)
                            print(f"Emitt x for screen {screen_name}: {tw['emitt_x']}")
                            print(f"Emitt y for screen {screen_name}: {tw['emitt_y']}")
                        else:
                            pass
                O0 = state0.get_orbit(bpms)
                O0x = np.asarray(O0['x'], dtype=float).reshape(-1, 1)
                O0y = np.asarray(O0['y'], dtype=float).reshape(-1, 1)
                orbit_now = np.concatenate([O0x, O0y]).ravel()
                if adaptive_orbit_only and self._adaptive_R is not None and self._adaptive_R_prev_kick is not None:
                    self._adaptive_R.update(self._adaptive_R_prev_kick, orbit_now - self._adaptive_R_prev_orbit)
                bpms0 = state0.get_bpms(bpms)
                x0_vals = np.asarray(bpms0['x'], dtype=float)
                y0_vals = np.asarray(bpms0['y'], dtype=float)
                mean_x = np.mean(x0_vals, axis=0) if x0_vals.ndim == 2 else x0_vals
                mean_y = np.mean(y0_vals, axis=0) if y0_vals.ndim == 2 else y0_vals
                orbit_rms_x = float(np.sqrt(np.mean(mean_x ** 2)))
                orbit_rms_y = float(np.sqrt(np.mean(mean_y ** 2)))
                orbit_rms_xy = float(np.sqrt(np.mean(mean_x ** 2 + mean_y ** 2)))
                self._hist_abs_rms_x.append(orbit_rms_x)
                self._hist_abs_rms_y.append(orbit_rms_y)
                self._hist_abs_rms_xy.append(orbit_rms_xy)

                '''
                UNCOMMENT AFTER SANITY CHECKS 
                '''

                if it == 0:
                    if self.orbit_at_first_start_click_x is None and self.orbit_at_first_start_click_y is None:
                        B0x = O0x.copy()
                        B0y = O0y.copy()
                        self.orbit_at_first_start_click_x = B0x
                        self.orbit_at_first_start_click_y = B0y
                    else:
                        B0x = self.orbit_at_first_start_click_x
                        B0y = self.orbit_at_first_start_click_y

                if self.reset_ref_orb == True:
                    B0x = O0x.copy()
                    B0y = O0y.copy()
                    self.reset_ref_orb = False
                    self.log("Reference orbit reset to current orbit")

                # dfs
                if w2 > 0:
                    print("Measuring dispersion")
                    self.log("Measuring dispersion")
                    dP_P = self.interface.change_energy()
                    state1 = self.interface.get_state()
                    state1 = self._apply_jitter_subtraction_to_state(state1)
                    energy_file = os.path.join(samples_dir, f"ITER_{it:04d}_energy.pkl")
                    state1.save(filename=energy_file)
                    self.interface.reset_energy()
                    O1 = state1.get_orbit(bpms)
                    O1x = np.asarray(O1['x'], dtype=float).reshape(-1, 1)
                    O1y = np.asarray(O1['y'], dtype=float).reshape(-1, 1)
                    bpms1 = state1.get_bpms(bpms)
                    x1_vals = np.asarray(bpms1["x"], dtype=float)
                    y1_vals = np.asarray(bpms1["y"], dtype=float)
                    err_dx = np.sqrt(np.square(np.asarray(O0["stdx"], dtype=float)) / x0_vals.shape[0] + np.square(
                        np.asarray(O1["stdx"], dtype=float)) / x1_vals.shape[0])
                    err_dy = np.sqrt(np.square(np.asarray(O0["stdy"], dtype=float)) / y0_vals.shape[0] + np.square(
                        np.asarray(O1["stdy"], dtype=float)) / y1_vals.shape[0])
                    Dx = np.array([1e3 * dx * dP_P for dx in target_disp_x]).reshape(-1, 1)
                    Dy = np.array([1e3 * dy * dP_P for dy in target_disp_y]).reshape(-1, 1)
                    plt.clf()
                    plt.plot(Dx, '--', color='blue', label="target dispersion x")
                    plt.plot(Dy, '--', color='orange', label="target dispersion y")
                else:
                    O1x = O1y = None
                    Dx = Dy = None

                # wfs
                if w3 > 0:
                    print("Measuring wakefield")
                    self.log("Measuring wakefield")
                    self.interface.change_intensity()
                    state2 = self.interface.get_state()
                    state2 = self._apply_jitter_subtraction_to_state(state2)
                    intensity_file = os.path.join(samples_dir, f"ITER_{it:04d}_intensity.pkl")
                    state2.save(filename=intensity_file)
                    self.interface.reset_intensity()
                    O2 = state2.get_orbit(bpms)
                    O2x = np.asarray(O2['x'], dtype=float).reshape(-1, 1)
                    O2y = np.asarray(O2['y'], dtype=float).reshape(-1, 1)
                    bpms2 = state2.get_bpms(bpms)
                    x2_vals = np.asarray(bpms2["x"], dtype=float)
                    y2_vals = np.asarray(bpms2["y"], dtype=float)
                else:
                    O2x = O2y = None
                self.test_orbits_data = {
                    "selected_bpms": list(bpms),
                    "O0x": np.asarray(O0x).reshape(-1),  # because np.array is creating a copy
                    "O1x": None if O1x is None else np.asarray(O1x).reshape(-1),
                    "O2x": None if O2x is None else np.asarray(O2x).reshape(-1),
                    "O0y": np.asarray(O0y).reshape(-1),
                    "O1y": None if O1y is None else np.asarray(O1y).reshape(-1),
                    "O2y": None if O2y is None else np.asarray(O2y).reshape(-1),
                    "dfs_err_x": None if w2 <= 0 else np.asarray(err_dx).reshape(-1),
                    "dfs_err_y": None if w2 <= 0 else np.asarray(err_dy).reshape(-1),
                }
                if not hasattr(self, "rms_orbits_data") or self.rms_orbits_data is None:
                    self.rms_orbits_data = {}

                self.rms_orbits_data = {
                    "selected_bpms": list(bpms),
                    "start_x": np.asarray(x0_vals, dtype=float) if it == 0 else self.rms_orbits_data.get("start_x"),
                    "start_y": np.asarray(y0_vals, dtype=float) if it == 0 else self.rms_orbits_data.get("start_y"),
                    "current_x": np.asarray(x0_vals, dtype=float),
                    "current_y": np.asarray(y0_vals, dtype=float),
                    "final_x": self.rms_orbits_data.get("final_x"),
                    "final_y": self.rms_orbits_data.get("final_y"),
                    "x1_vals": None if w2 <= 0 else np.asarray(x1_vals, dtype=float),
                    "y1_vals": None if w2 <= 0 else np.asarray(y1_vals, dtype=float),
                    "x2_vals": None if w3 <= 0 else np.asarray(x2_vals, dtype=float),
                    "y2_vals": None if w3 <= 0 else np.asarray(y2_vals, dtype=float),
                    "nominal_x": None,
                    "nominal_y": None,
                }

                if self.nominal_state is not None:
                    nominal_bpms = self.nominal_state.get_bpms(bpms)
                    self.rms_orbits_data["nominal_x"] = np.asarray(nominal_bpms["x"], dtype=float)
                    self.rms_orbits_data["nominal_y"] = np.asarray(nominal_bpms["y"], dtype=float)

                Bx = []
                By = []
                if w1 > 0:
                    Bx.append(wgt_orb * (O0x - B0x))
                    By.append(wgt_orb * (O0y - B0y))
                if w2 > 0 and O1x is not None:
                    plt.errorbar(range(len(O1x)), (O1x - O0x).ravel(), yerr=err_dx,
                                 color='blue', label="measured x", capsize=3)
                    plt.errorbar(range(len(O1y)), (O1y - O0y).ravel(), yerr=err_dy,
                                 color='orange', label="measured y", capsize=3)
                    plt.xlabel("BPM index")
                    plt.ylabel(f"Orbit difference [{self.bpm_unit}]")
                    plt.title(f"DFS: measured orbit difference vs target dispersion (x, y): iteration {it + 1}/{iters}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    dfs_plot_ax = plt.gca()  # get current axis, don't mistake for other plot
                    Bx.append(wgt_dfs * ((O1x - O0x) - Dx))
                    By.append(wgt_dfs * ((O1y - O0y) - Dy))

                if w3 > 0 and O2x is not None:
                    By.append(wgt_wfs * (O2y - O0y))
                    Bx.append(wgt_wfs * (O2x - O0x))

                Axx_it = np.array(Axx_base, copy=True)
                Ayy_it = np.array(Ayy_base, copy=True)
                Axy_it = np.array(Axy_base, copy=True)
                Ayx_it = np.array(Ayx_base, copy=True)
                Bx = np.vstack(Bx)
                By = np.vstack(By)

                print("||Bx|| =", np.linalg.norm(Bx))
                print("||By|| =", np.linalg.norm(By))

                Axx_it[np.isnan(Axx_it)] = 0
                Ayy_it[np.isnan(Ayy_it)] = 0
                Axy_it[np.isnan(Axy_it)] = 0
                Ayx_it[np.isnan(Ayx_it)] = 0

                Axx_it[np.isnan(Bx.ravel()), :] = 0  # flattens an array into 1d
                Axy_it[np.isnan(Bx.ravel()), :] = 0  # flattens an array into 1d

                Ayy_it[np.isnan(By.ravel()), :] = 0
                Ayx_it[np.isnan(By.ravel()), :] = 0

                Bx[np.isnan(Bx)] = 0
                By[np.isnan(By)] = 0

                # A = U * Sigma * V^
                # A^+ = V * Sigma^+ * U^T

                filter_corr_x = np.all(np.isfinite(Axx_it), axis=0) & np.all(np.isfinite(Ayx_it), axis=0)
                filter_corr_y = np.all(np.isfinite(Ayy_it), axis=0) & np.all(np.isfinite(Axy_it), axis=0)

                Axx_it = Axx_it[:, filter_corr_x]
                Ayx_it = Ayx_it[:, filter_corr_x]

                Ayy_it = Ayy_it[:, filter_corr_y]
                Axy_it = Axy_it[:, filter_corr_y]

                Cy_cut = [corr for corr, true in zip(vcorrs, filter_corr_y) if true]
                Cx_cut = [corr for corr, true in zip(hcorrs, filter_corr_x) if true]

                A = np.block([[Axx_it, Axy_it],
                              [Ayx_it, Ayy_it]])

                if adaptive_orbit_only:
                    if self._adaptive_R is None or self._adaptive_R.R0.shape != A.shape:
                        self._adaptive_R = AdaptiveResponseMatrix(A)  # first use, or corrector/BPM selection changed
                    else:
                        A = self._adaptive_R.R

                B = np.vstack([Bx, By])

                A[np.isnan(A)] = 0
                B[np.isnan(B)] = 0

                # corrX = -gain * (np.linalg.pinv(Axx, rcond=rcond) @ Bx)  # theta = - gain * Axx^+ *Bx
                # corrY = -gain * (np.linalg.pinv(Ayy, rcond=rcond) @ By)

                A_weighted = w_xy_bpms * A
                B_weighted = w_xy_bpms * B

                # adding the (Aw.T*Aw+beta*I)-1
                if beta > 0:  # with beta, A.T@A+beta*I is always reversible, so we use solve, matrix is square and reversible
                    delta = -gain * np.linalg.solve(A_weighted.T @ A_weighted + beta * np.eye(A_weighted.shape[1]),
                                                    A_weighted.T @ B_weighted)  # without pinv, because we add beta so that singular values will not be near zero
                else:  # np.eye(n) singular matrix with shape= number of columns
                    delta = -gain * (np.linalg.pinv(A_weighted, rcond=rcond) @ B_weighted)
                print(f"Delta is {delta}")
                nh = len(Cx_cut)
                corrX = delta[:nh]  # horizontal changes
                corrY = delta[nh:]  # vertical changes

                delta_x = np.asarray(corrX.ravel(), dtype=float)  # asarray doesnt copy, like np.array
                delta_y = np.asarray(corrY.ravel(), dtype=float)
                delta_vals = np.concatenate([delta_x, delta_y])

                selected_correctors = Cx_cut + Cy_cut
                print("Cx_cut =", Cx_cut)
                print("Cy_cut =", Cy_cut)
                print("selected_correctors =", selected_correctors)
                current_corrs = self.interface.get_correctors(selected_correctors)
                returned_names = list(current_corrs["names"])
                returned_bdes = np.asarray(current_corrs["bdes"], dtype=float).ravel()
                returned_bact = np.asarray(current_corrs["bact"], dtype=float).ravel()

                bdes_map = {name: val for name, val in zip(returned_names, returned_bdes)}
                bact_map = {name: val for name, val in zip(returned_names, returned_bact)}
                current_bdes = np.array([bdes_map[name] for name in selected_correctors], dtype=float)
                current_bact = np.array([bact_map[name] for name in selected_correctors], dtype=float)
                readback_tolerance = 0.05
                if not np.allclose(current_bact, current_bdes, rtol=0.0, atol=readback_tolerance):
                    failed = [
                        f"{name} (bdes={target:.6g}, bact={actual:.6g})"
                        for name, target, actual in zip(selected_correctors, current_bdes, current_bact)
                        if abs(actual - target) > readback_tolerance
                    ]
                    raise RuntimeError(
                        "Corrector readback differs from its setpoint before correction: " + ", ".join(failed))

                max_vals_x = np.full(delta_x.shape, max_curr_h, dtype=float)
                max_vals_y = np.full(delta_y.shape, max_curr_v, dtype=float)
                max_vals = np.concatenate([max_vals_x, max_vals_y])

                new_bdes = current_bdes + delta_vals
                new_bdes = clamp(new_bdes, max_vals)
                for attempt in range(1, 4):
                    set_ok = self.interface.set_correctors(selected_correctors, new_bdes)
                    after_corrs = self.interface.get_correctors(selected_correctors)
                    after_names = list(after_corrs["names"])
                    after_bact_map = {
                        name: val for name, val in
                        zip(after_names, np.asarray(after_corrs["bact"], dtype=float).ravel())
                    }
                    after_bact = np.array([after_bact_map[name] for name in selected_correctors], dtype=float)
                    if set_ok is not False and np.allclose(after_bact, new_bdes, rtol=0.0, atol=readback_tolerance):
                        break
                    self.log(f"Corrector readback mismatch after BBA kick (attempt {attempt}/3)")
                else:
                    failed = [
                        f"{name} (target={target:.6g}, bact={actual:.6g})"
                        for name, target, actual in zip(selected_correctors, new_bdes, after_bact)
                        if abs(actual - target) > readback_tolerance
                    ]
                    raise RuntimeError(
                        "BBA stopped: correctors did not reach their requested currents (the correction is not reliable): " + ", ".join(
                            failed))

                after_names = list(after_corrs["names"])
                after_vals = np.asarray(after_corrs["bdes"], dtype=float).ravel()
                after_map = {name: val for name, val in zip(after_names, after_vals)}
                after_bdes = np.array([after_map[name] for name in selected_correctors], dtype=float)

                print("selected_correctors =", selected_correctors)
                print("current_bdes =", current_bdes)
                print("new_bdes =", new_bdes)
                print("after_bdes =", after_bdes)
                applied_delta = after_bdes - current_bdes
                print("applied_delta =", applied_delta)

                if w2 > 0 and O1x is not None and O1y is not None and prev_Dx is not None and prev_Dy is not None:
                    dfs_start = n if w1 > 0 else 0
                    dfs_rows = slice(dfs_start, dfs_start + n)  # slice of an array start:end
                    applied_x = applied_delta[:nh]
                    applied_y = applied_delta[nh:]
                    dfs_prediction_x = (
                                prev_Dx + (Axx_it[dfs_rows, :] @ applied_x + Axy_it[dfs_rows, :] @ applied_y) / wgt_dfs)
                    dfs_prediction_y = (
                                prev_Dy + (Ayx_it[dfs_rows, :] @ applied_x + Ayy_it[dfs_rows, :] @ applied_y) / wgt_dfs)
                    dfs_plot_ax.plot(range(n), dfs_prediction_x, color="blue", linestyle="--", label="R prediction x")
                    dfs_plot_ax.plot(range(n), dfs_prediction_y, color="orange", linestyle="--", label="R prediction y")
                    dfs_plot_ax.legend()
                    dfs_plot_ax.figure.canvas.draw_idle()
                if adaptive_orbit_only:
                    self._adaptive_R_prev_kick = applied_delta.copy()
                    self._adaptive_R_prev_orbit = orbit_now

                kicks_path = os.path.join(self._session_dir, "kicks.txt")
                write_header = not os.path.exists(kicks_path)
                with open(kicks_path, "a") as file:
                    if write_header:
                        file.write("time\titeration\tcorrector\tbdes_before\tapplied_kick\tbdes_after\n")
                    time = datetime.now().isoformat(timespec="seconds")
                    for corrector, before, kick, after in zip(selected_correctors, current_bdes, applied_delta,
                                                              after_bdes):
                        file.write(f"{time}\t{it + 1}\t{corrector}\t{before:.12g}\t{kick:.12g}\t{after:.12g}\n")
                # new bdes and after bdes should be the same

                vals = new_bdes - current_bdes

                # current_bdes + delta -> clamp(final_bdes) -> set_correctors(final_bdes)

                if w1 > 0:
                    mean_orbit_x, mean_orbit_y, err_x_orbit, err_y_orbit, mean_orbit_all, err_orbit_all = self._calc_error(
                        x0_vals, y0_vals, ref_x=B0x.ravel(), ref_y=B0y.ravel())  # ravel makes data a vector
                    self._hist_orbit_x.append(mean_orbit_x)
                    self._hist_orbit_y.append(mean_orbit_y)
                    self._hist_orbit.append(mean_orbit_all)
                    self._hist_orbit_x_err.append(err_x_orbit)
                    self._hist_orbit_y_err.append(err_y_orbit)
                    self._hist_orbit_err.append(err_orbit_all)

                if w2 > 0 and O1x is not None and O1y is not None:
                    dx_disp = x1_vals - x0_vals
                    dy_disp = y1_vals - y0_vals
                    prev_Dx = (O1x - O0x).ravel()
                    prev_Dy = (O1y - O0y).ravel()
                    mean_disp_x, mean_disp_y, err_disp_x, err_disp_y, mean_disp_all, err_disp_all = self._calc_error(
                        dx_disp, dy_disp, ref_x=np.zeros(dx_disp.shape[1]), ref_y=np.zeros(dy_disp.shape[1]),
                        disp_x=Dx.ravel(), disp_y=Dy.ravel())
                    self._hist_disp_x.append(mean_disp_x)
                    self._hist_disp_y.append(mean_disp_y)
                    self._hist_disp.append(mean_disp_all)
                    self._hist_disp_x_err.append(err_disp_x)
                    self._hist_disp_y_err.append(err_disp_y)
                    self._hist_disp_err.append(err_disp_all)

                if w3 > 0 and O2x is not None:
                    dx_wake = x2_vals - x0_vals
                    dy_wake = y2_vals - y0_vals
                    mean_wake_x, mean_wake_y, err_wake_x, err_wake_y, mean_wake_all, err_wake_all = self._calc_error(
                        dx_wake, dy_wake, ref_x=np.zeros(dx_wake.shape[1]), ref_y=np.zeros(dy_wake.shape[1]))
                    self._hist_wake_x.append(mean_wake_x)
                    self._hist_wake_y.append(mean_wake_y)
                    self._hist_wake.append(mean_wake_all)
                    self._hist_wake_x_err.append(err_wake_x)
                    self._hist_wake_y_err.append(err_wake_y)
                    self._hist_wake_err.append(err_wake_all)

                self._plot_series(ax=self.traj_ax, canvas=self.traj_canvas, values_x=self._hist_orbit_x,
                                  values_y=self._hist_orbit_y, vals=self._hist_orbit, error_x=self._hist_orbit_x_err,
                                  error_y=self._hist_orbit_y_err, error_all=self._hist_orbit_err, title=None)
                self._plot_series(ax=self.disp_ax, canvas=self.disp_canvas, values_x=self._hist_disp_x,
                                  values_y=self._hist_disp_y, vals=self._hist_disp, error_x=self._hist_disp_x_err,
                                  error_y=self._hist_disp_y_err, error_all=self._hist_disp_err, title=None)
                self._plot_series(ax=self.wake_ax, canvas=self.wake_canvas, values_x=self._hist_wake_x,
                                  values_y=self._hist_wake_y, vals=self._hist_wake, error_x=self._hist_wake_x_err,
                                  error_y=self._hist_wake_y_err, error_all=self._hist_wake_err, title=None)
                QApplication.processEvents()
                last_completed_iteration = it

            self.setWindowTitle("BBA GUI")
            if not silent:
                QMessageBox.information(self, "Correction", "Correction finished.")
            final_state = self.interface.get_state()
            final_machine_status_file = os.path.join(self._session_dir, "machine_status_after_correction.pkl")
            final_state.save(filename=final_machine_status_file)
            self.log(f"Saved machine status after correction: {os.path.basename(final_machine_status_file)}")
            final_state = self._apply_jitter_subtraction_to_state(final_state)
            if last_completed_iteration is not None:
                final_nominal_file = os.path.join(
                    samples_dir, f"ITER_{last_completed_iteration + 1:04d}_nominal.pkl")
                final_state.save(filename=final_nominal_file)
                self.log(f"Saved final nominal BBA state: {os.path.basename(final_nominal_file)}")
            screens_f = final_state.get_screens()
            print("Screen values after correction:")
            print(f"Sigx: {screens_f['sigx']}")
            print(f"Sigy: {screens_f['sigy']}")
            if hasattr(self.interface, "get_twiss_at_screen"):
                print("Emittance after correction:")
                for screen_name in screens_f["names"]:
                    tw = self.interface.get_twiss_at_screen(screen_name)
                    print(f"Emitt x for screen {screen_name}: {tw['emitt_x']}")
                    print(f"Emitt y for screen {screen_name}: {tw['emitt_y']}")
            else:
                pass
            final_bpms = final_state.get_bpms(bpms)
            final_x_vals = np.asarray(final_bpms["x"], dtype=float)
            final_y_vals = np.asarray(final_bpms["y"], dtype=float)

            if hasattr(self, "rms_orbits_data") and self.rms_orbits_data is not None:
                self.rms_orbits_data["final_x"] = final_x_vals
                self.rms_orbits_data["final_y"] = final_y_vals

            mean_final_x = np.mean(final_x_vals, axis=0) if final_x_vals.ndim == 2 else final_x_vals
            mean_final_y = np.mean(final_y_vals, axis=0) if final_y_vals.ndim == 2 else final_y_vals

            final_rms_x = float(np.sqrt(np.mean(mean_final_x ** 2)))
            final_rms_y = float(np.sqrt(np.mean(mean_final_y ** 2)))
            final_rms_xy = float(np.sqrt(np.mean(mean_final_x ** 2 + mean_final_y ** 2)))

            self._hist_abs_rms_x.append(final_rms_x)
            self._hist_abs_rms_y.append(final_rms_y)
            self._hist_abs_rms_xy.append(final_rms_xy)

            if silent:
                self.log("Internal orbit correction finished.")
            else:
                self.log("Correction finished.")
            if not silent:
                self.save_session_settings(w1, w2, w3, rcond, iters, gain, beta,
                                           self.max_horizontal_current_spinbox.value(),
                                           self.max_vertical_current_spinbox.value(),
                                           bool(self.triangular_checkbox.isChecked()), self.bpm_weights, Axx, Ayy, Axy,
                                           Ayx, Bx, By, bool(self.subtract_jitter_checkbox.isChecked()))
            if preserve_plots and plot_snapshot is not None:
                self._hist_orbit_x[:] = plot_snapshot["orbit_x"]
                self._hist_orbit_y[:] = plot_snapshot["orbit_y"]
                self._hist_orbit[:] = plot_snapshot["orbit"]
                self._hist_disp_x[:] = plot_snapshot["disp_x"]
                self._hist_disp_y[:] = plot_snapshot["disp_y"]
                self._hist_disp[:] = plot_snapshot["disp"]
                self._hist_wake_x[:] = plot_snapshot["wake_x"]
                self._hist_wake_y[:] = plot_snapshot["wake_y"]
                self._hist_wake[:] = plot_snapshot["wake"]
                self._hist_orbit_x_err[:] = plot_snapshot["orbit_x_err"]
                self._hist_orbit_y_err[:] = plot_snapshot["orbit_y_err"]
                self._hist_orbit_err[:] = plot_snapshot["orbit_err"]
                self._hist_disp_x_err[:] = plot_snapshot["disp_x_err"]
                self._hist_disp_y_err[:] = plot_snapshot["disp_y_err"]
                self._hist_disp_err[:] = plot_snapshot["disp_err"]
                self._hist_wake_x_err[:] = plot_snapshot["wake_x_err"]
                self._hist_wake_y_err[:] = plot_snapshot["wake_y_err"]
                self._hist_wake_err[:] = plot_snapshot["wake_err"]
                self._hist_abs_rms_x[:] = plot_snapshot["abs_rms_x"]
                self._hist_abs_rms_y[:] = plot_snapshot["abs_rms_y"]
                self._hist_abs_rms_xy[:] = plot_snapshot["abs_rms_xy"]
                self._plot_series(ax=self.traj_ax, canvas=self.traj_canvas, values_x=self._hist_orbit_x,
                                  values_y=self._hist_orbit_y, vals=self._hist_orbit, error_x=self._hist_orbit_x_err,
                                  error_y=self._hist_orbit_y_err, error_all=self._hist_orbit_err, title=None)
                self._plot_series(ax=self.disp_ax, canvas=self.disp_canvas, values_x=self._hist_disp_x,
                                  values_y=self._hist_disp_y, vals=self._hist_disp, error_x=self._hist_disp_x_err,
                                  error_y=self._hist_disp_y_err, error_all=self._hist_disp_err, title=None)
                self._plot_series(ax=self.wake_ax, canvas=self.wake_canvas, values_x=self._hist_wake_x,
                                  values_y=self._hist_wake_y, vals=self._hist_wake, error_x=self._hist_wake_x_err,
                                  error_y=self._hist_wake_y_err, error_all=self._hist_wake_err, title=None)

        except Exception as e:
            self.setWindowTitle("BBA GUI")
            QMessageBox.critical(self, "Correction error", str(e))
            self.log(f"Correction error: {e}")
            print_exception(e)  # it shows even the line that generated that error
        finally:
            self._running = False

    def _stop_correction(self):
        self._cancel = True
        QMessageBox.information(self, "Correction", "Stop requested. Finishing current iteration...")
        self.log("Stop requested. Finishing current iteration...")

    def _show_test_orbits(self):
        if self.test_orbits is None:
            self.test_orbits = TestOrbits(self)
        if not hasattr(self, "test_orbits_data") or self.test_orbits_data is None:
            QMessageBox.information(self, "No test orbits available", "No test orbits available.")
        else:
            self.test_orbits._plot_test_orbits(selected_bpms=self.test_orbits_data["selected_bpms"],
                                               O0x=self.test_orbits_data["O0x"], O0y=self.test_orbits_data["O0y"],
                                               O1x=self.test_orbits_data["O1x"], O1y=self.test_orbits_data["O1y"],
                                               O2x=self.test_orbits_data["O2x"], O2y=self.test_orbits_data["O2y"],
                                               dfs_err_x=self.test_orbits_data["dfs_err_x"],
                                               dfs_err_y=self.test_orbits_data["dfs_err_y"])
        self.test_orbits.show()
        self.test_orbits.raise_()
        self.test_orbits.activateWindow()

    def _show_console_log(self):
        if self.log_console is None:
            self.log_console = LogConsole(self)
        self.log_console.show()
        self.log_console.raise_()
        self.log_console.activateWindow()

    def log(self, text):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {text}"
        if self.log_console is None:
            self.log_console = LogConsole(self)
            # self.log_console.show()
        self.log_console.log(line)

    def _clear_graphs(self):
        # it doesnt do fresh start, it only clears the graphs
        self._cancel = True
        self._hist_orbit_x.clear(), self._hist_orbit_y.clear()
        self._hist_transmission.clear()
        self.initial_charge_value = None
        self._hist_disp_x.clear(), self._hist_disp_y.clear()
        self._hist_wake_x.clear(), self._hist_wake_y.clear()
        self._hist_orbit.clear(), self._hist_disp.clear(), self._hist_wake.clear()
        self._hist_orbit_x_err.clear(), self._hist_orbit_y_err.clear(), self._hist_orbit_err.clear()
        self._hist_disp_x_err.clear(), self._hist_disp_y_err.clear(), self._hist_disp_err.clear()
        self._hist_wake_x_err.clear(), self._hist_wake_y_err.clear(), self._hist_wake_err.clear()

        self._plot_series(self.traj_ax, self.traj_canvas, values_x=[], values_y=[], vals=[], title=None)
        self._plot_series(self.disp_ax, self.disp_canvas, values_x=[], values_y=[], vals=[], title=None)
        self._plot_series(self.wake_ax, self.wake_canvas, values_x=[], values_y=[], vals=[], title=None)
        self._refresh_all_plot_popups()

    def _refresh_metric_plots_for_mode(self):
        self.plot_widget_4.setEnabled(True)
        self.plot_widget_5.setEnabled(True)
        if self.disp_ax is not None:
            self.disp_ax.set_facecolor("white")
        if self.wake_ax is not None:
            self.wake_ax.set_facecolor("white")
        self._plot_series(self.traj_ax, self.traj_canvas, [], [], [], title=None)
        self._plot_series(self.disp_ax, self.disp_canvas, [], [], [], title=None)
        self._plot_series(self.wake_ax, self.wake_canvas, [], [], [], title=None)

    def _apply_jitter_subtraction_to_state(self, state):
        if self.jitter_model is None:
            return state
        state.bpms = apply_jitter_subtraction(state.get_bpms(), self.jitter_model)
        return state

    def handling(self, app_name, cwd=None, args=None):
        try:
            path = os.path.join(os.path.dirname(__file__), app_name)
            workdir = os.path.expanduser(os.path.expandvars(cwd))
            proc = QProcess(self)
            proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            proc.setWorkingDirectory(workdir)
            env = QProcessEnvironment.systemEnvironment()
            proc.setProcessEnvironment(env)
            argv = [path] + list(args or [])
            proc.start(sys.executable, argv)
            proc.readyReadStandardOutput.connect(
                lambda p=proc: print(bytes(p.readAllStandardOutput()).decode(errors="ignore")))
            self._procs.append(proc)
            print(workdir)

        except Exception as e:
            self.log(f"Error: {e}")
            print(f"Error in handling: {e}")

    def _display_response_matrix(self):
        orbit_dir = self.trajectory_response_3.text()
        dispersion_dir = self.dfs_response_3.text()
        wakefield_dir = self.wfs_response_3.text()
        selected_mode = None
        for rb in self.radio_buttons:
            if rb.isChecked():
                selected_mode = rb.text()
                break
        if selected_mode is None:
            selected_mode = self.modes[0]

        if selected_mode == self.modes[0]:
            mode = "orbit"
            args = ["--dir1", orbit_dir, "--compute"]

        elif selected_mode == self.modes[1]:
            mode = "dispersion"
            if not dispersion_dir:
                QMessageBox.warning(self, "Warning", "No dispersion directory selected")
                return
            args = ["--dir1", dispersion_dir, "--dir2", orbit_dir, "--diff", "--compute"]

        elif selected_mode == self.modes[2]:
            mode = "wakefield"
            if not wakefield_dir:
                QMessageBox.warning(self, "Warning", "No wakefield directory selected")
                return
            args = ["--dir1", wakefield_dir, "--dir2", orbit_dir, "--diff", "--compute"]

        self.handling('ComputeResponseMatrix_GUI.py', cwd=self.cwd, args=args)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    from Backend import SelectInterface

    dialog = SelectInterface.choose_acc_and_interface()
    if dialog is None:
        print("Selection cancelled.")
        sys.exit(1)

    I = dialog
    project_name = I.get_name()

    nominal_state = None
    start_state = I.get_state()

    print(f"Selected interface: {project_name}")
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"~/CERN-Flight_Simulator-Data/BBA_{I.get_name()}{time_str}_session_settings"
    dir_name = os.path.expanduser(os.path.expandvars(dir_name))
    w = MainWindow(interface=I, dir_name=dir_name, nominal_state=nominal_state, start_state=start_state)

    if hasattr(I, "log_messages"):
        I.log_messages(w.log)

    w.show()
    sys.exit(app.exec())
