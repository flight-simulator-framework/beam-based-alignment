import beam_based_alignment
import sys, os, glob, shutil, matplotlib, importlib, inspect, tempfile
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.optimize import lsq_linear
try:
    from PyQt6 import uic
    from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QEvent
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QListWidgetItem, QMessageBox, QVBoxLayout, QDialog, QStyledItemDelegate, QWidget, QInputDialog, QFileDialog)
    from PyQt6.QtGui import QPainter, QPixmap

    pyqt_version = 6
except ImportError:
    from PyQt5 import uic
    from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QEvent
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QListWidgetItem, QMessageBox, QVBoxLayout, QDialog, QStyledItemDelegate, QWidget, QInputDialog, QFileDialog)
    from PyQt5.QtGui import QPainter, QPixmap

    pyqt_version = 5
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from machine_interfaces.Backend.LogConsole import LogConsole
from machine_interfaces.Backend.SaveOrLoad import SaveOrLoad
from machine_interfaces.Backend.State import State
from response_matrix.ResponseMatrix_DFS_WFS import ResponseMatrix_DFS_WFS, AdaptiveResponseMatrix
import matplotlib.pyplot as plt
from beam_based_alignment.BBA_helpers.BPM_weights import BPM_weights
from beam_based_alignment.BBA_helpers.MeasureDispersionWakefields import MeasureDispersionWakefields
from machine_interfaces.Backend.ActuatorMode import ActuatorMode
from traceback import print_exception
from machine_interfaces.Interfaces.interface_setup import INTERFACE_SETUP, get_intensity_labels
from machine_interfaces.Backend.EnergyChangeMethod import EnergyChangeMethod
from machine_interfaces.Knobs.jitter_subtraction import (apply_jitter_subtraction, explain_reference_selection, fit_jitter_model)

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
        self._intensity_labels = get_intensity_labels(interface)
        self.dir_name = dir_name
        self.nominal_state = nominal_state
        self.start_state = start_state if start_state is not None else interface.get_state()
        self.restore_state = self.start_state
        self.initial_state = self.start_state
        self._cancel = False
        ui_path = os.path.join(os.path.dirname(beam_based_alignment.__file__), "UI files/BBA_GUI.ui")
        uic.loadUi(ui_path, self)
        self.history_content.setVisible(False)
        self.history_toggle_button.toggled.connect(self.history_content.setVisible)
        self.history_toggle_button.toggled.connect(lambda expanded: self.history_toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow))
        self._history_files = []
        self._correction_dir = None
        self.history_load_button.clicked.connect(lambda: self._load_history(load_settings=True))
        self.history_iteration_slider.valueChanged.connect(self.history_iteration_spinbox.setValue)
        self.history_iteration_spinbox.valueChanged.connect(self.history_iteration_slider.setValue)
        self.history_iteration_slider.valueChanged.connect(self._show_history_iteration)
        self.history_previous_button.clicked.connect(lambda: self.history_iteration_slider.setValue(self.history_iteration_slider.value() - 1))
        self.history_next_button.clicked.connect(lambda: self.history_iteration_slider.setValue(self.history_iteration_slider.value() + 1))
        self.history_restore_button.clicked.connect(self._restore_history_iteration)
        self._default_beam_tooltips = {
            name: getattr(self, name).toolTip()
            for name in ("subtract_jitter_checkbox", "jitter_subtraction_tooltip", "wfs_response_tooltip", "measure_wakefields_tooltip")
        }
        self._default_measure_texts = {"energy": self.measure_dispersion_button.text(),
                                       "intensity": self.measure_wakefields_button.text()}
        self._setup_intensity_labels()
        self._clock_zone_name = self._get_clock_zone()
        self._setup_machine_clock()
        self._load_logo()

        self.bpms_list.setItemDelegate(BpmWeightsDelegate(self.bpms_list))
        self.correctors_list.setItemDelegate(BpmWeightsDelegate(self.correctors_list))
        self._selection_before_click=[]
        self._weights_double_click = False
        self.bpms_list.viewport().installEventFilter(self)
        self.correctors_list.viewport().installEventFilter(self)

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
        self.traj_popup, self.disp_popup, self.wake_popup = None, None, None
        self._setup_canvases()
        self._plot_double_clicks()
        self.bpm_weights = {}
        self.corrector_weights = {}
        self._populate_lists()
        self._procs = []
        self.load_correctors_button.clicked.connect(self._load_correctors)
        self.load_bpms_button.clicked.connect(self._load_bpms)
        self.radio_buttons = [self.mode_orbit, self.mode_dispersion, self.mode_wakefield]
        self.pushButton_log.clicked.connect(self._show_console_log)
        self.session_database_3.setText(dir_name)
        self.pushButton_8.clicked.connect(self._pick_and_load_traj_data)
        self.pushButton_9.clicked.connect(self._pick_and_load_disp_data)
        self.pushButton_10.clicked.connect(self._pick_and_load_wake_data)
        self.clear_graphs_button.clicked.connect(self._clear_graphs)
        self.restore_initial_settings.clicked.connect(self._restore_initial_settings)
        self._running = False
        self.start_button.clicked.connect(self._on_start_click)
        self.preview_button.clicked.connect(self._preview_in_simulation)
        self._preview_history = None
        self.stop_button.clicked.connect(self._stop_correction)
        self.jitter_model = None
        self.subtract_jitter_checkbox.setChecked(False)
        self.actuator_mode = ActuatorMode.Kicker
        self._setup_corrector_controls()
        self._setup_beam_change_controls()
        self.restore_machine_status_button.clicked.connect(self._pick_and_load_machine_status_file)
        self.initial_charge_value = None
        self.orbit_at_first_start_click_x = None
        self.orbit_at_first_start_click_y = None
        self.method = self.beam_change_method_combo.currentData()
        self._test_orbit_results = {}
        self._measurement_stop_requested = False
        self.measure_dispersion_button.clicked.connect(lambda: self._measure_test_orbit("dfs"))
        self.measure_wakefields_button.clicked.connect(lambda: self._measure_test_orbit("wfs"))
        self.show_dfs_test_orbit_button.clicked.connect(lambda: self._display_test_orbit("dfs"))
        self.show_wfs_test_orbit_button.clicked.connect(lambda: self._display_test_orbit("wfs"))
        self.stop_dispersion_measurement_button.clicked.connect(self._stop_measurement)
        self.stop_wakefields_measurement_button.clicked.connect(self._stop_measurement)

    def _test_orbit_spec(self, kind):
        m = self.beam_change_method_combo.currentData()
        if kind == "dfs":
            name = "dispersion_quadrupole_scaling" if m == EnergyChangeMethod.QUADRUPOLE_SCALING else "dispersion_energy"
            return name, lambda: self.interface.change_energy(method=m), lambda: self.interface.reset_energy(method=m)
        if m == EnergyChangeMethod.BOTH:
            return ("dispersion_quadrupole_scaling", lambda: self.interface.change_energy(method=m, which_method=2),
                    lambda: self.interface.reset_energy(method=m, which_method=2))
        return "wakefields", self.interface.change_intensity, self.interface.reset_intensity

    def _stop_measurement(self):
        self._measurement_stop_requested = True
        self.log("Measurement stop requested. Finishing current iteration...")

    def _measure_test_orbit(self, kind):
        if self._running or not self._apply_beam_change_controls():
            return
        name, change, reset = self._test_orbit_spec(kind)
        n = int((self.no_of_meas_dfs_test_orbit if kind == "dfs" else self.no_of_meas_wfs_test_orbit).value())
        settings = self._beam_change_values.copy()  # in case anything changes in GUI line edits
        self._save_machine_status()
        _, bpms = self._get_selection()
        directory = os.path.join(self._session_dir, f"{name}_measurement")
        os.makedirs(directory, exist_ok=True)
        self._running, self._measurement_stop_requested = True, False
        dx, dy = [], []
        self.log(f"Measuring {name} test orbit ({n} measurements)...")
        try:
            for it in range(n):
                if self._measurement_stop_requested:
                    break
                state0 = self.interface.get_state()
                try:
                    change()
                    state1 = self.interface.get_state()
                finally:
                    reset()
                state0.save(filename=os.path.join(directory, f"{it:04d}_nominal.pkl"))
                state1.save(filename=os.path.join(directory, f"{it:04d}_changed.pkl"))
                O0, O1 = state0.get_orbit(bpms), state1.get_orbit(bpms)
                dx.append(np.asarray(O1["x"], dtype=float) - np.asarray(O0["x"], dtype=float))
                dy.append(np.asarray(O1["y"], dtype=float) - np.asarray(O0["y"], dtype=float))
                QApplication.processEvents()
        except Exception as e:
            QMessageBox.critical(self, "Measurement error", str(e))
            self.log(f"Measurement error: {e}")
            print_exception(e)
        finally:
            self._running = False
        if dx:
            self._test_orbit_results[name] = result = {
                "measurement_type": name, "method": settings["_method"], "settings": settings,
                "bpm_names": list(bpms), "bpm_unit": self.bpm_unit, "x_samples": np.array(dx), "y_samples": np.array(dy),
                "n_measurements": len(dx), "requested_measurements": n, "save_directory": directory,
            }
            self._save_test_orbit(result)
        self.log(f"{name} test orbit: {len(dx)}/{n} measurements saved in {directory}")

    def _save_test_orbit(self, r):
        cols = []
        for key in ("x_samples", "y_samples"):
            s = np.ma.masked_invalid(r[key])
            cols += [s.mean(axis=0).filled(np.nan), s.std(axis=0, ddof=1).filled(np.nan)] # average is also calculated from those measurements, so thats why we do N-1 (ddof=1)
        u = r["bpm_unit"]
        header = (f"{r['measurement_type']} test orbit: changed - nominal orbit over {r['n_measurements']}/{r['requested_measurements']} measurements\n"
                  f"method: {r['method']}\nsettings: {r['settings']}\n"
                  f"BPM\tdx_mean[{u}]\tdx_std[{u}]\tdy_mean[{u}]\tdy_std[{u}]")
        np.savetxt(os.path.join(r["save_directory"], f"{r['measurement_type']}_test_orbit.txt"),
                   np.column_stack([np.array(r["bpm_names"], dtype=object), *cols]),
                   fmt=["%s"] + ["%.6e"] * 4, delimiter="\t", header=header)

    def _display_test_orbit(self, kind):
        result = self._test_orbit_results.get(self._test_orbit_spec(kind)[0])
        if result is None:
            QMessageBox.information(self, "Test orbit", "No test orbit measured for the selected method yet.")
        else:
            MeasureDispersionWakefields(result, self).show()

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
        self.pushButton_11.clicked.connect(self.load_session_settings)
        self.setWindowTitle(f"BBA GUI - {self.interface.get_name()}")
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
        correctors = self.interface.get_correctors()
        correctors_list = correctors['names']
        units_settings, sysid_kick, bpm_unit, corrs_unit = self._get_interface_units()
        self.sysid_kick = sysid_kick
        self.bpm_unit = bpm_unit
        self.corrs_unit = corrs_unit
        self.current_groupbox_right.setTitle(f"Max strength ({self.corrs_unit})" if self.corrs_unit else "Max strength")
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
        base_dir = os.path.dirname(beam_based_alignment.__file__)
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
            popup = PlotPopup(self._plot_titles().get(plot_type, "Correction plot"), parent=self)
            setattr(self, popup_attr, popup)
        self._refresh_plot_popup(plot_type)
        popup.show()
        popup.raise_()
        popup.activateWindow()

    def _plot_titles(self):
        return {"traj": "Trajectory (Residual to response matrix reference)",
                "disp": "Dispersion (Residual to response matrix reference)",
                "wake": self._intensity_labels["plot_title"]}

    def _refresh_plot_popup(self, plot_kind):
        popup = getattr(self, f"{plot_kind}_popup", None)
        if popup is None:
            return
        title = self._plot_titles()[plot_kind]
        history = self._history({"traj": "orbit", "disp": "disp", "wake": "wake"}[plot_kind])
        popup.setWindowTitle(title)
        ax = popup.ax
        ax.clear()
        self._draw_series(ax, *history)
        if any(history[:3]):
            ax.legend(fontsize=9, loc="upper right")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"Residual norm [{self.bpm_unit}]")
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

    def _setup_intensity_labels(self):
        labels = self._intensity_labels
        self.groupBox_7.setTitle(labels["correction"])
        self.groupBox_4.setTitle(labels["plot_title"])
        self.mode_wakefield.setText(labels["mode_option_label"])
        self.label_3.setText(labels["weight"])
        self.label_3.setWordWrap(True)
        self.wakefield_weight_tooltip.setToolTip(labels["weight_tooltip"])
        for name in ("subtract_jitter_checkbox", "jitter_subtraction_tooltip"):
            getattr(self, name).setToolTip(labels.get("jitter_tooltip", self._default_beam_tooltips[name]))
        self.wfs_response_tooltip.setToolTip(labels.get("response_tooltip", self._default_beam_tooltips["wfs_response_tooltip"]))
        self.measure_wakefields_tooltip.setToolTip(labels.get("measurement_tooltip", self._default_beam_tooltips["measure_wakefields_tooltip"]))

    def _setup_beam_change_controls(self):
        beam_change = (self._get_interface_initial_settings() or {}).get("beam_change", {})
        self._beam_change_texts = {}
        self._beam_change_fields = []
        self.beam_change_method_combo.clear()
        for label in beam_change["methods"]:
            self.method = EnergyChangeMethod(beam_change["method_ids"][label])
            self.beam_change_method_combo.addItem(label, self.method)
        self._beam_change_method = self.beam_change_method_combo.currentText()
        self.beam_change_method_combo.currentTextChanged.connect(self._change_energy_method)
        self._refresh_beam_change_controls()

    def _beam_method_profile(self):
        beam_change = (self._get_interface_initial_settings() or {}).get("beam_change", {})
        return beam_change.get("method_settings", {}).get(self._beam_change_method, {})

    def _change_energy_method(self, method):
        if self._running:
            self.beam_change_method_combo.blockSignals(True)
            self.beam_change_method_combo.setCurrentText(self._beam_change_method)
            self.beam_change_method_combo.blockSignals(False)
            return
        for widget, field in self._beam_change_fields:
            self._beam_change_texts[field["attribute"]] = widget.text()
        self._beam_change_method = method
        self._refresh_beam_change_controls()
        for key, widget in (("dfs", self.dfs_response_3), ("wfs", self.wfs_response_3)):
            self._data_dirs[key] = None
            widget.clear()
        self._clear_graphs()

    def _refresh_beam_change_controls(self):
        beam_change = (self._get_interface_initial_settings() or {}).get("beam_change", {})
        profile = self._beam_method_profile()
        self._beam_change_fields = []
        self.beam_change_group.setVisible(bool(beam_change))
        self._intensity_labels = get_intensity_labels(self.interface, self._beam_change_method)
        self._setup_intensity_labels()

        controls = {
            "energy": (
                self.energy_change_group,
                self.energy_nominal_label,
                self.energy_nominal_input,
                self.energy_test_label,
                self.energy_test_input,
                self.energy_change_tooltip,
                self.measure_dispersion_button,
            ),
            "intensity": (
                self.intensity_change_group,
                self.intensity_nominal_label,
                self.intensity_nominal_input,
                self.intensity_test_label,
                self.intensity_test_input,
                self.intensity_change_tooltip,
                self.measure_wakefields_button,
            ),
        }
        for kind, widgets in controls.items():
            title, nominal_label, nominal_input, test_label, test_input, tooltip, measure_button = widgets
            settings = profile.get(kind, beam_change.get(kind))
            title.setVisible(settings is not None)
            tooltip.setVisible(settings is not None)
            if settings is None:
                for widget in (nominal_label, nominal_input, test_label, test_input):
                    widget.setVisible(False)
                continue
            title.setTitle(settings["label"])
            title.setStyleSheet(title.styleSheet())
            measure_button.setText(settings.get("measurement_button", self._default_measure_texts[kind]))
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
                input_widget.setText(self._beam_change_texts.get(field["attribute"], "" if value is None else str(value)))
                input_widget.setPlaceholderText(field.get("placeholder", ""))
                self._beam_change_fields.append((input_widget, field))

    def _apply_beam_change_controls(self):
        self._beam_change_values = {"_method": self._beam_change_method}
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
        method = values.get("_method")
        if method is not None:
            index = self.beam_change_method_combo.findText(method)
            if index < 0:
                raise ValueError(f"Energy-change method is unavailable: {method}")
            self.beam_change_method_combo.setCurrentIndex(index)
        for input_widget, field in self._beam_change_fields:
            attribute = field["attribute"]
            if attribute not in values:
                continue
            value = values[attribute]
            input_widget.setText("" if value is None else str(value))
        return self._apply_beam_change_controls()

    def _load_history(self, folder=None, load_settings=False):
        folder = folder or QFileDialog.getExistingDirectory(self, "Select BBA session", os.path.expanduser("~/CERN-Flight_Simulator-Data/"))
        files = sorted(glob.glob(os.path.join(folder, "BBA_states", "ITER_*_nominal.pkl"))) if folder else []
        if not files:
            return
        if State(filename=files[0]).get_interface_id() != f"{type(self.interface).__module__}.{type(self.interface).__name__}":
            QMessageBox.warning(self, "Iteration history", "This session was recorded on a different interface/machine.")
            return
        self._history_folder, self._history_files = folder, files
        last = len(files) - 1
        self.history_session_label.setText(os.path.basename(folder))
        self.history_iteration_count_label.setText(f"/ {last}")
        self.history_iteration_spinbox.setSpecialValueText("")
        for widget in (self.history_iteration_slider, self.history_iteration_spinbox):
            widget.setMaximum(last)
        for widget in (self.history_previous_button, self.history_next_button, self.history_iteration_slider, self.history_iteration_spinbox,
                       self.history_restore_button):
            widget.setEnabled(True)
        self.history_iteration_slider.setValue(last)
        self._show_history_iteration(last)
        if load_settings:
            self.load_session_settings(folder)
            self._redraw_history_plots()

    def _show_history_iteration(self, index):
        if not self._history_files:
            return
        state, first = State(filename=self._history_files[index]), State(filename=self._history_files[0])
        bpms = list(state.get_bpms()["names"])
        orbit, reference = state.get_orbit(bpms), first.get_orbit(bpms)
        rms = lambda dx, dy: f"{np.sqrt(np.nanmean(np.r_[dx, dy] ** 2)):.3f} {self.bpm_unit}"
        energy_file = self._history_files[index].replace("_nominal.pkl", "_energy.pkl")
        energy = State(filename=energy_file).get_orbit(bpms) if os.path.isfile(energy_file) else None
        ict = getattr(self.interface, "chosen_ict", None)
        charge = lambda s: np.nanmean(np.asarray(s.get_icts(ict)["charge"], dtype=float))
        self.history_timestamp_label.setText(f"Time: {state.get_timestamp():%Y-%m-%d %H:%M:%S}")
        self.history_trajectory_label.setText(f"Trajectory RMS: {rms(orbit['x'] - reference['x'], orbit['y'] - reference['y'])}")
        self.history_dispersion_label.setText(f"Dispersion RMS: {rms(energy['x'] - orbit['x'], energy['y'] - orbit['y']) if energy else '—'}")
        self.history_transmission_label.setText(f"Transmission: {100 * charge(state) / charge(first):.1f} %" if ict else "Transmission: —")

    def _restore_history_iteration(self):
        if self._running or not self._history_files:
            return
        index = self.history_iteration_slider.value()
        state = State(filename=self._history_files[index])
        movers = self.actuator_mode == ActuatorMode.QM
        question = (f"Set the {'quadrupole movers' if movers else 'correctors'} to iteration {index} "
                    f"({state.get_timestamp():%H:%M:%S}) of {os.path.basename(self._history_folder)}?")
        if QMessageBox.question(self, "Iteration history", question) != QMessageBox.StandardButton.Yes:
            return
        restored = self.interface.restore_quadrupoles_state(state) if movers else self.interface.restore_correctors_state(state)
        if restored is False:
            QMessageBox.warning(self, "Iteration history", "Not every corrector was confirmed at its saved value. Check them on the machine.")
        self.log(f"Machine set to iteration {index} of {self._history_folder}")

    def _restore_initial_settings(self):
        self.log("Restoring initial settings...")
        self._cancel = True
        self._running = False
        w1, w2, w3, rcond, iters, gain, beta, transmission_threshold= self._read_params()
        try:
            if w2 > 0:
                self.interface.reset_energy(method=self.method)
            if w3 > 0:
                self._reset_intensity()
        except Exception:
            self.log("The machine wasn't restored to its nominal state.")
            QMessageBox.critical(self, "Restore error",
                                 self._intensity_labels["restore_error"])
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

    def eventFilter(self, watched, event):
        for elements_list, edit_weights in ((self.bpms_list, self._edit_bpm_weights), (self.correctors_list, self._edit_corrector_weights)):
            if watched is not elements_list.viewport(): continue
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._weights_double_click = False
                self._selection_before_click = elements_list.selectedItems()
            elif event.type() == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
                element = elements_list.itemAt(event.pos())
                if element is not None:
                    self._weights_double_click = True
                    for i in range(elements_list.count()):
                        item = elements_list.item(i)
                        item.setSelected(item in self._selection_before_click)
                    edit_weights(element)
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease and self._weights_double_click:
                self._weights_double_click = False
                return True
        return super().eventFilter(watched, event)

    def _edit_bpm_weights(self, bpm):
        bpm_name = bpm.data(Qt.ItemDataRole.UserRole) or (bpm.text() or "")
        if not bpm_name:
            bpm_name = (bpm.text() or "").split("    [", 1)[0]
        bpm_window = BPM_weights(bpm_name=bpm_name, parent=self, intensity_label=self._intensity_labels["compact_label"])
        if bpm_name in self.bpm_weights:
            w_orb, w_dfs, w_wfs = self.bpm_weights[bpm_name]
            bpm_window.set_values(w_orb, w_dfs, w_wfs)

        if bpm_window.exec() == QDialog.DialogCode.Accepted:
            self.bpm_weights[bpm_name] = bpm_window.get_values()
            self._update_bpm_weights(bpm)

    def _edit_corrector_weights(self, corr):
        corr_name = corr.text()
        weight, ok = QInputDialog.getDouble(self, f"Corrector weight - {corr_name}", "Corrector weight (used when beta > 0): ", self.corrector_weights.get(corr_name, 1.0), 0.0, 100.0, 2)
        if ok:
            self.corrector_weights[corr_name] = weight
            self._update_corrector_weights(corr)


    def _on_start_click(self):
        self.log("Starting button clicked...")
        if self._running or not self._apply_beam_change_controls():
            return
        self.method = self.beam_change_method_combo.currentData()
        if self._correction_dir is None or self.actuator_mode == ActuatorMode.QM:
            self._save_machine_status()
            self._correction_dir = self._session_dir
        else:
            self._session_dir = self._correction_dir
            self.session_database_3.setText(self._session_dir)
        self._running = True
        self.beam_change_group.setEnabled(False)
        self._start_correction()
        self._load_history(self._session_dir)

    def _simulated_twin(self):
        name = type(self.interface).__module__.removesuffix("_RFTrack") + "_RFTrack"
        cls = getattr(importlib.import_module(name), name.rsplit(".", 1)[1])
        options = {"nsamples": getattr(self.interface, "nsamples", 1), "machine_offsets": True}
        twin = cls(**{key: value for key, value in options.items() if key in inspect.signature(cls).parameters})
        twin.log = self.log
        state = self.interface.get_state()
        if hasattr(twin, "load_machine_state"):
            twin.load_machine_state(state)
        else:
            twin.restore_quadrupoles_state(state)
            twin.restore_correctors_state(state)
        for _, field in self._beam_change_fields:
            if hasattr(twin, field["attribute"]) and getattr(self.interface, field["attribute"], None) is not None:
                setattr(twin, field["attribute"], getattr(self.interface, field["attribute"]))
        if hasattr(twin, "dfs_test_energy"):
            twin.dfs_test_energy = getattr(self.interface, "dfs_test_energy", 0.92)
        if hasattr(self.interface, "initial_quad_currents"):
            initial = dict(zip(self.interface.quadrupoles, np.ravel(self.interface.initial_quad_currents)))
            twin.initial_quad_currents = np.array([initial.get(name, current) for name, current in zip(twin.quadrupoles, twin.get_quadrupoles()["bdes"])])
        return twin

    def _preview_in_simulation(self):
        if self._running or not self._apply_beam_change_controls():
            return
        self.method = self.beam_change_method_combo.currentData()
        saved = {name: value for name, value in vars(self).items() if name.startswith("_hist_")}
        saved.update({name: getattr(self, name, None) for name in ("interface", "_session_dir", "_data_dirs", "initial_charge_value", "jitter_model",
                                                                  "orbit_at_first_start_click_x", "orbit_at_first_start_click_y", "reset_ref_orb")})
        session_text = self.session_database_3.text()
        self._preview_history = None
        self._running = True
        try:
            self.log("Preview in simulation: loading the machine settings into the simulated interface...")
            self.interface = self._simulated_twin()
            self._session_dir = tempfile.mkdtemp(prefix="BBA_preview_")
            self._data_dirs = {key: info and {**info, "dir": self._session_dir} for key, info in self._data_dirs.items()}
            for name in saved:
                if name.startswith("_hist_"):
                    setattr(self, name, [])
            self._start_correction()
            self._preview_history = {name: getattr(self, name) for name in saved if name.startswith("_hist_")}
        except Exception as e:
            QMessageBox.critical(self, "Preview in simulation", str(e))
            print_exception(e)
        finally:
            for name, value in saved.items():
                setattr(self, name, value)
            self._running = False
            self.session_database_3.setText(session_text)
            self._redraw_history_plots()

    def _expand_data_path(self, path):
        home = os.path.expanduser("~")
        if path.startswith(home + os.sep):  # the character used by the operating system to separate pathname components
            return "~" + path[len(home):]
        return path

    def _setup_canvases(self):
        def install(host):
            fig = Figure(figsize=(5, 2.4), tight_layout=True)
            canvas = FigureCanvas(fig)
            layout = host.layout()
            if layout is None:
                layout = QVBoxLayout(host)
                layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(canvas)
            return canvas, fig.add_subplot(111)

        self.traj_canvas, self.traj_ax = install(self.plot_widget_3)
        self.disp_canvas, self.disp_ax = install(self.plot_widget_4)
        self.wake_canvas, self.wake_ax = install(self.plot_widget_5)

    def _plot_series(self, ax, canvas, values_x, values_y, vals, title=None, error_x=None, error_y=None,
                     error_all=None):
        if canvas is None or ax is None or getattr(self, "_suppress_main_plots", False):
            return
        if ax is self.traj_ax and getattr(self, "traj_transmission_ax", None) is not None:
            self.traj_transmission_ax.remove()
            self.traj_transmission_ax = None
        ax.clear()
        self._draw_series(ax, values_x, values_y, vals, error_x, error_y, error_all)
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
        ax.set_ylabel(f"Residual norm [{self.bpm_unit}]", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.yaxis.get_offset_text().set_fontsize(10)
        ax.grid(True, alpha=0.3)
        canvas.draw_idle()
        for kind, plot_ax in (("traj", self.traj_ax), ("disp", self.disp_ax), ("wake", self.wake_ax)):
            if ax is plot_ax:
                self._refresh_plot_popup(kind)

    @staticmethod
    def _draw_series(ax, values_x, values_y, vals, error_x=None, error_y=None, error_all=None):
        for values, errors, style in ((values_x, error_x, dict(marker="o", color="blue", label="x")),
                                      (values_y, error_y, dict(marker="o", color="red", label="y")),
                                      (vals, error_all, dict(linestyle="dashed", color="black", label="combined norm"))):
            if values:  # yerr - height of the error bar on the plot, capsize - size of the top line on the error bar
                yerr = errors if errors is not None and len(errors) == len(values) else None
                ax.errorbar(range(len(values)), values, yerr=yerr, capsize=6, elinewidth=2, capthick=2, markersize=4, **style)

    def _get_bpm_weights_text(self, bpm_name):
        wbpm_orb, wbpm_dfs, wbpm_wfs = self.bpm_weights.get(bpm_name, (1.0, 1.0, 1.0))
        return f"[w1 = {wbpm_orb:g}, w2 = {wbpm_dfs:g}, w3 = {wbpm_wfs:g}]"  # general format, removes reduntant zeros at the end etc.

    def _update_bpm_weights(self, item):
        bpm_name = item.data(Qt.ItemDataRole.UserRole) or (item.text() or "")  # it gives a clean name of the item, even if there is another text (like weights)
        item.setData(BpmWeightsDelegate.WEIGHTS_ROLE, self._get_bpm_weights_text(bpm_name))
        item.setText(bpm_name)

    def _update_corrector_weights(self, item):
        item.setData(BpmWeightsDelegate.WEIGHTS_ROLE, f"[w = {self.corrector_weights.get(item.text(), 1.0):g}]")

    def _populate_lists(self):
        corrs = self.initial_state.get_correctors()["names"]
        bpms = self.initial_state.get_bpms()["names"]
        self.correctors_list.insertItems(0, corrs)
        for i in range(self.correctors_list.count()):
            self._update_corrector_weights(self.correctors_list.item(i))
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

    def _dataset_beam_signature(self, pairs):
        signature = ResponseMatrix_DFS_WFS._dataset_beam_signature(pairs)
        return signature if any(not key.startswith("quadrupoles.") for key in signature) else {}

    def _force_triangular(self) -> bool:
        return bool(self.triangular_checkbox.isChecked())

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
        def read(name, default, cast=float):  # gets the text value and turns it into a float
            widget = getattr(self, name)
            try:
                return cast(float(widget.text().strip()))
            except ValueError:
                widget.setText(f"{default:g}")
                return cast(default)

        return (read("lineEdit", 1.0), read("lineEdit_2", 10.0), read("lineEdit_3", 10.0) if (self._data_dirs.get("wfs") or {}).get("ok") else 0.0, read("lineEdit_4", 0.001),
                read("lineEdit_5", 10, int), read("lineEdit_6", 0.4), read("lineEdit_beta", 0.0), read("transmission_value", 0.65) * 100)

    def _reset_reference_orbit(self):
        self.reset_ref_orb = True
        self.log("Resetting reference orbit")

    def _build_jitter_model_for_correction(self, actuators, bpms):
        if not self.subtract_jitter_checkbox.isChecked():
            self.jitter_model = None
            return None

        _, w2, w3, *_ = self._read_params()
        if w2 > 0 or w3 > 0:
            message = (f"Jitter subtraction is not applied while DFS or {self._intensity_labels['compact_label']} is used (w2 or w3 > 0). "
                       "The energy or intensity change also moves the reference BPMs, so the subtraction would remove part of the measured signal. "
                       "The correction continues without jitter subtraction.")
            self.log(message)
            QMessageBox.warning(self, "Jitter subtraction", message)
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

    def _solve_correction(self, A_weighted, B_weighted, corrector_weights, gain, beta, rcond, lower_bounds, upper_bounds):
        A_solver = A_weighted.copy()
        b_solver = -gain * B_weighted.ravel()

        if beta > 0:
            L = np.diag(np.sqrt(corrector_weights * beta))
            A_solver = np.vstack((A_solver, L))
            b_solver = np.concatenate((b_solver, np.zeros(len(corrector_weights))))
        else:
            U, s, Vt = np.linalg.svd(A_solver, full_matrices=False)
            cutoff = rcond * s.max()
            keep = s > cutoff
            if not keep.any():
                raise RuntimeError("No response matrix directions remain after applying rcond.")

            s_keep = s[keep]
            U_keep = U[:, keep]
            Vt_keep = Vt[keep, :]
            A_solver = np.vstack((np.diag(s_keep) @ Vt_keep, s.max() * Vt[~keep, :]))
            b_solver = np.concatenate((U_keep.T @ b_solver, np.zeros(int((~keep).sum()))))
        result = lsq_linear(A_solver, b_solver, (lower_bounds, upper_bounds))
        if not result.success:
            raise RuntimeError(result.message)

        return result.x

    def _start_correction(self, preview = False):
        try:
            corrs, bpms = self._get_selection()
            corrs = [corr for corr in corrs if self.corrector_weights.get(corr, 1.0)  > 0]
            if not corrs: raise ValueError("All corrector weights are set to zero.")
            self.log("Starting correction...")
            self.interface.switch_off_sextupoles()
            self._build_jitter_model_for_correction(actuators=corrs, bpms=bpms)
            if self.jitter_model is not None:
                refs = set(self.jitter_model["reference_bpms"])
                bpms = [bpm for bpm in bpms if bpm not in refs]
                self.log("Removed jitter reference BPMs from correction targets")

            self._cancel = False
            w1, w2, w3, rcond, iters, gain, beta, transmission_threshold = self._read_params()

            Axx, Ayy, Axy, Ayx, B0x, B0y, hcorrs, vcorrs, bpms_common = self._creating_response_matrices(selected_corrs=corrs, selected_bpms=bpms)
            Axx_base, Ayy_base, Axy_base, Ayx_base = (np.array(M, copy=True) for M in (Axx, Ayy, Axy, Ayx))
            if self.remove_coupling_checkbox.isChecked():
                Axy_base.fill(0.0)
                Ayx_base.fill(0.0)

            bpms = list(bpms_common)
            n = len(bpms)
            bpm_weights = np.array([self.bpm_weights.get(bpm, (1.0, 1.0, 1.0)) for bpm in bpms], dtype=float).reshape(n, 3)
            W_x = np.vstack([bpm_weights[:, [k]] for k, w in enumerate((w1, w2, w3)) if w > 0])
            w_xy_bpms = np.sqrt(np.vstack([W_x, W_x]))  # because the weights are the same, B=vstack(bx,by)

            self.setWindowTitle("BBA GUI - [Correction running]")
            target_disp_x, target_disp_y = self.interface.get_target_dispersion(bpms)
            max_curr_h = self.max_horizontal_current_spinbox.value()  # gauss * m in RFT, in reality
            max_curr_v = self.max_vertical_current_spinbox.value()  # gauss * m in RFT, A in reality

            def clamp(val, max_val):
                val = np.asarray(val, dtype=float)
                max_val = np.asarray(max_val, dtype=float)
                result = val.copy()
                finite = np.isfinite(max_val) & (max_val > 0.0)
                result[finite] = np.clip(result[finite], -max_val[finite], max_val[finite])  # clip limits the values in an array
                return result
            plt.ion()

            self.use_adaptive_R = getattr(self, "use_adaptive_R", True)
            self._adaptive_R = None  # the matrix from files already includes earlier corrections
            self._adaptive_R_prev_kick = None
            self._adaptive_R_prev_B = None
            if w2 > 0:
                self.interface.reset_energy(method=self.method)
            if w3 > 0:
                self._reset_intensity()

            samples_dir = os.path.join(self._session_dir, "BBA_states")
            os.makedirs(samples_dir, exist_ok=True)
            first = max(len(glob.glob(os.path.join(samples_dir, "ITER_*_nominal.pkl"))) - 1, 0)
            state_file = lambda it, kind: os.path.join(samples_dir, f"ITER_{first + it:04d}_{kind}.pkl")
            prev_Dx = prev_Dy = None
            last_completed_iteration = None
            Bx=By=None
            previous_bdes = None
            previous_correctors = None
            for it in range(iters):
                if self._cancel:
                    break

                # nominal
                self.log("Measuring orbit")
                state0 = self._measure_state(state_file(it, "nominal"))
                if hasattr(self.interface, "chosen_ict"):
                    charge = np.asarray(state0.get_icts(self.interface.chosen_ict)["charge"], dtype=float).ravel()
                    if charge.size and np.isfinite(charge[0]):
                        if self.initial_charge_value is None:
                            if charge[0] ==0:
                                self._cancel = True
                                QMessageBox.warning(self, "No beam charge!", "Cannot establish transmission reference from zero charge. Correction stopped.")
                                break
                            self.initial_charge_value = charge[0]
                        reference = float(self.initial_charge_value)
                        self._hist_transmission.append(100.0 * charge[0] / reference)
                        if self._hist_transmission[-1] <= transmission_threshold:
                            self._cancel = True
                            if previous_bdes is not None and previous_correctors is not None:
                                if self.interface.set_correctors(previous_correctors, previous_bdes) is False:
                                    raise RuntimeError("Could not confirm restoration of corrector settings from iteration before. Restore them in the GUI.")
                                self.log("Correctors settings restored to the values from previous iteration due to low transmission.")
                            else: self.log("No correction step to undo in this iteration.")
                            QMessageBox.warning(self, "Transmission below level!", "Transmission has reached the threshold. Correction stopped. See the log for restoration status.")
                            break
                    else:
                        self._hist_transmission.append(np.nan)
                if it == 0:
                    self._print_screens(state0, "before")
                O0 = state0.get_orbit(bpms)
                O0x = np.asarray(O0['x'], dtype=float).reshape(-1, 1)
                O0y = np.asarray(O0['y'], dtype=float).reshape(-1, 1)
                bpms0 = state0.get_bpms(bpms)
                x0_vals = np.asarray(bpms0['x'], dtype=float)
                y0_vals = np.asarray(bpms0['y'], dtype=float)
                self._append_abs_rms(x0_vals, y0_vals)

                if it == 0:
                    if self.orbit_at_first_start_click_x is None and self.orbit_at_first_start_click_y is None:
                        B0x = O0x.copy()
                        B0y = O0y.copy()
                        self.orbit_at_first_start_click_x = B0x
                        self.orbit_at_first_start_click_y = B0y
                    else:
                        B0x = self.orbit_at_first_start_click_x
                        B0y = self.orbit_at_first_start_click_y

                if self.reset_ref_orb:
                    B0x = O0x.copy()
                    B0y = O0y.copy()
                    self.reset_ref_orb = False
                    self.log("Reference orbit reset to current orbit")

                if w1 > 0 and (O0x.shape != B0x.shape or O0y.shape != B0y.shape):
                    self.setWindowTitle("BBA GUI")
                    QMessageBox.warning(
                        self, "BPM selection changed",
                        "The number of selected BPMs differs from the saved reference orbit.\n"
                        "Restore the previous BPM selection or click Reset reference orbit, then start again.",
                    )
                    return

                # dfs
                if w2 > 0:
                    self.log("Measuring dispersion")
                    try:
                        dP_P = self.interface.change_energy(method=self.method)
                        state1 = self._measure_state(state_file(it, "energy"))
                    finally:
                        self.interface.reset_energy(method=self.method)
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
                    self.log(self._intensity_labels["measuring"])
                    try:
                        self._change_intensity()
                        state2 = self._measure_state(state_file(it, "intensity"))
                    finally:
                        self._reset_intensity()
                    O2 = state2.get_orbit(bpms)
                    O2x = np.asarray(O2['x'], dtype=float).reshape(-1, 1)
                    O2y = np.asarray(O2['y'], dtype=float).reshape(-1, 1)
                    bpms2 = state2.get_bpms(bpms)
                    x2_vals = np.asarray(bpms2["x"], dtype=float)
                    y2_vals = np.asarray(bpms2["y"], dtype=float)
                else:
                    O2x = O2y = None

                Bx = []
                By = []
                if w1 > 0:
                    Bx.append(w1 * (O0x - B0x))
                    By.append(w1 * (O0y - B0y))
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
                    Bx.append(w2 * ((O1x - O0x) - Dx))
                    By.append(w2 * ((O1y - O0y) - Dy))

                if w3 > 0 and O2x is not None:
                    By.append(w3 * (O2y - O0y))
                    Bx.append(w3 * (O2x - O0x))

                Axx_it, Ayy_it, Axy_it, Ayx_it = (np.array(M, copy=True) for M in (Axx_base, Ayy_base, Axy_base, Ayx_base))
                Bx = np.vstack(Bx)
                By = np.vstack(By)
                B_now = np.vstack([Bx, By]).ravel()

                for M in (Axx_it, Ayy_it, Axy_it, Ayx_it):
                    M[np.isnan(M)] = 0

                Axx_it[np.isnan(Bx.ravel()), :] = 0  # flattens an array into 1d
                Axy_it[np.isnan(Bx.ravel()), :] = 0
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

                if self.use_adaptive_R:
                    if self._adaptive_R is None or self._adaptive_R.R0.shape != A.shape:
                        self._adaptive_R = AdaptiveResponseMatrix(A)  # first use, or corrector/BPM selection changed
                    else:
                        self._adaptive_R.update(self._adaptive_R_prev_kick, B_now - self._adaptive_R_prev_B)
                        A = np.where(A != 0, self._adaptive_R.R, 0.0)

                B = np.vstack([Bx, By])

                A[np.isnan(A)] = 0
                B[np.isnan(B)] = 0

                A_weighted = w_xy_bpms * A
                B_weighted = w_xy_bpms * B

                corrector_weights = np.array([self.corrector_weights.get(corr, (1.0)) for corr in (Cx_cut + Cy_cut)], dtype=float)
                max_vals = np.concatenate([np.full(len(Cx_cut), max_curr_h, dtype=float), np.full(len(Cy_cut), max_curr_v, dtype=float)])
                max_vals = np.where(max_vals>0, max_vals, np.inf)
                selected_correctors = Cx_cut + Cy_cut
                current_bdes, current_bact = self._corrector_readings(selected_correctors)
                readback_mismatch = ~np.isclose(current_bact, current_bdes, rtol=0.0, atol=0.05)
                lower_bounds = -max_vals-current_bdes
                upper_bounds = max_vals-current_bdes
                if readback_mismatch.any():
                    raise RuntimeError("Corrector readback differs from its setpoint before correction: " + ", ".join(
                        f"{name} (bdes={target:.6g}, bact={actual:.6g})"
                        for name, target, actual, bad in zip(selected_correctors, current_bdes, current_bact, readback_mismatch) if bad))
                delta = self._solve_correction(A_weighted, B_weighted, corrector_weights, gain, beta, rcond, lower_bounds, upper_bounds)
                    #delta = -gain * (np.linalg.pinv(A_weighted, rcond=rcond) @ B_weighted)
                nh = len(Cx_cut)
                delta_x = np.asarray(delta[:nh].ravel(), dtype=float)  # horizontal changes
                delta_y = np.asarray(delta[nh:].ravel(), dtype=float)  # vertical changes
                delta_vals = np.concatenate([delta_x, delta_y])
                new_bdes = clamp(current_bdes + delta_vals, max_vals)
                if preview:
                    return

                previous_bdes = current_bdes.copy()
                previous_correctors = selected_correctors.copy()
                if self.interface.set_correctors(selected_correctors, new_bdes) is False:
                    raise RuntimeError("BBA stopped: correctors did not reach their requested currents (the correction is not reliable). "
                                       "See the log for the correctors that failed.")
                after_bdes, _ = self._corrector_readings(selected_correctors)  # new bdes and after bdes should be the same
                applied_delta = after_bdes - current_bdes

                if w2 > 0 and O1x is not None and O1y is not None and prev_Dx is not None and prev_Dy is not None:
                    dfs_start = n if w1 > 0 else 0
                    dfs_rows = slice(dfs_start, dfs_start + n)  # slice of an array start:end
                    applied_x = applied_delta[:nh]
                    applied_y = applied_delta[nh:]
                    dfs_prediction_x = (
                                prev_Dx + (Axx_it[dfs_rows, :] @ applied_x + Axy_it[dfs_rows, :] @ applied_y) / w2)
                    dfs_prediction_y = (
                                prev_Dy + (Ayx_it[dfs_rows, :] @ applied_x + Ayy_it[dfs_rows, :] @ applied_y) / w2)
                    dfs_plot_ax.plot(range(n), dfs_prediction_x, color="blue", linestyle="--", label="R prediction x")
                    dfs_plot_ax.plot(range(n), dfs_prediction_y, color="orange", linestyle="--", label="R prediction y")
                    dfs_plot_ax.legend()
                    dfs_plot_ax.figure.canvas.draw_idle()
                if self.use_adaptive_R:
                    self._adaptive_R_prev_kick = applied_delta.copy()
                    self._adaptive_R_prev_B = B_now

                kicks_path = os.path.join(self._session_dir, "kicks.txt")
                write_header = not os.path.exists(kicks_path)
                with open(kicks_path, "a") as file:
                    if write_header:
                        file.write("time\titeration\tcorrector\tbdes_before\tapplied_kick\tbdes_after\n")
                    now = datetime.now().isoformat(timespec="seconds")
                    for corrector, before, kick, after in zip(selected_correctors, current_bdes, applied_delta, after_bdes):
                        file.write(f"{now}\t{first + it + 1}\t{corrector}\t{before:.12g}\t{kick:.12g}\t{after:.12g}\n")

                if w1 > 0:
                    self._append_history("orbit", self._calc_error(x0_vals, y0_vals, ref_x=B0x.ravel(), ref_y=B0y.ravel()))  # ravel makes data a vector

                if w2 > 0 and O1x is not None and O1y is not None:
                    dx_disp = x1_vals - x0_vals
                    dy_disp = y1_vals - y0_vals
                    prev_Dx = (O1x - O0x).ravel()
                    prev_Dy = (O1y - O0y).ravel()
                    self._append_history("disp", self._calc_error(dx_disp, dy_disp, ref_x=np.zeros(dx_disp.shape[1]), ref_y=np.zeros(dy_disp.shape[1]),
                                                                  disp_x=Dx.ravel(), disp_y=Dy.ravel()))

                if w3 > 0 and O2x is not None:
                    dx_wake = x2_vals - x0_vals
                    dy_wake = y2_vals - y0_vals
                    self._append_history("wake", self._calc_error(dx_wake, dy_wake, ref_x=np.zeros(dx_wake.shape[1]), ref_y=np.zeros(dy_wake.shape[1])))

                self._redraw_history_plots()
                QApplication.processEvents()
                last_completed_iteration = it

            self.setWindowTitle("BBA GUI")
            final_state = self.interface.get_state()
            final_machine_status_file = os.path.join(self._session_dir, "machine_status_after_correction.pkl")
            final_state.save(filename=final_machine_status_file)
            self.log(f"Saved machine status after correction: {os.path.basename(final_machine_status_file)}")
            final_state = self._apply_jitter_subtraction_to_state(final_state)
            if last_completed_iteration is not None:
                final_nominal_file = state_file(last_completed_iteration + 1, "nominal")
                final_state.save(filename=final_nominal_file)
                self.log(f"Saved final nominal BBA state: {os.path.basename(final_nominal_file)}")
                if w2 > 0 and not self._cancel:
                    self.log("Measuring final dispersion")
                    try:
                        dP_P = self.interface.change_energy(method=self.method)
                        final_energy_state = self._measure_state(state_file(last_completed_iteration + 1, "energy"))
                    finally:
                        self.interface.reset_energy(method=self.method)
                    nominal_bpms = final_state.get_bpms(bpms)
                    energy_bpms = final_energy_state.get_bpms(bpms)
                    self._append_history("disp", self._calc_error(
                        np.asarray(energy_bpms["x"]) - np.asarray(nominal_bpms["x"]),
                        np.asarray(energy_bpms["y"]) - np.asarray(nominal_bpms["y"]),
                        ref_x=np.zeros(n), ref_y=np.zeros(n),
                        disp_x=1e3 * np.asarray(target_disp_x) * dP_P,
                        disp_y=1e3 * np.asarray(target_disp_y) * dP_P))
                    self._redraw_history_plots()
            if self.interface.reset_sextupoles():
                self._restore_orbit_after_sextupoles(final_state, corrs, bpms, rcond, max_curr_h, max_curr_v, samples_dir)
            # Add this correction's states to the response-matrix data it used, so the next fit includes them
            session = os.path.basename(self._session_dir)
            for key, kind in (("traj", "nominal"), ("dfs", "energy"), ("wfs", "intensity")):
                if self._data_dirs.get(key):
                    target = os.path.join(self._data_dirs[key]["dir"], "BBA_iterations")
                    for fn in glob.glob(os.path.join(samples_dir, f"ITER_*_{kind}.pkl")):
                        os.makedirs(target, exist_ok=True)
                        shutil.copy(fn, os.path.join(target, f"{session}_{os.path.basename(fn)}"))
            self._print_screens(final_state, "after")
            final_bpms = final_state.get_bpms(bpms)
            self._append_abs_rms(np.asarray(final_bpms["x"], dtype=float), np.asarray(final_bpms["y"], dtype=float))

            ending_message = ("Correction stopped" if self._cancel else "Correction finished") + (" in the simulated interface." if getattr(self.interface, "is_simulation", False) else ".")
            self.log(ending_message)
            self.save_session_settings(w1, w2, w3, rcond, iters, gain, beta,
                                       self.max_horizontal_current_spinbox.value(),
                                       self.max_vertical_current_spinbox.value(),
                                       bool(self.triangular_checkbox.isChecked()), self.bpm_weights, Axx, Ayy, Axy,
                                       Ayx, Bx, By, bool(self.subtract_jitter_checkbox.isChecked()))
            QMessageBox.information(self, "Correction", ending_message)

        except Exception as e:
            self.setWindowTitle("BBA GUI")
            QMessageBox.critical(self, "Correction error", str(e))
            self.log(f"Correction error: {e}")
            print_exception(e)  # it shows even the line that generated that error
        finally:
            self.interface.reset_sextupoles()
            self._running = False
            self.beam_change_group.setEnabled(True)

    def _measure_state(self, filename):
        state = self._apply_jitter_subtraction_to_state(self.interface.get_state())
        state.save(filename=filename)
        return state

    def _change_intensity(self):
        if self.method == EnergyChangeMethod.BOTH:
            return self.interface.change_energy(method=self.method, which_method=2)
        return self.interface.change_intensity()

    def _reset_intensity(self):
        if self.method == EnergyChangeMethod.BOTH:
            return self.interface.reset_energy(method=self.method, which_method=2)
        return self.interface.reset_intensity()

    def _restore_orbit_after_sextupoles(self, bba_state, corrs, bpms, rcond, max_h, max_v, samples_dir):
        Rxx, Ryy, Rxy, Ryx, _, _, hcorrs, vcorrs, bpms = self._compute_response_matrix(
            pairs=self._data_dirs["traj"]["pairs"], correctors=corrs, bpms=bpms, triangular=self._force_triangular())
        correctors = list(hcorrs) + list(vcorrs)
        inverse = np.linalg.pinv(np.nan_to_num(np.block([[Rxx, Rxy], [Ryx, Ryy]])), rcond=rcond)
        limits = np.r_[np.full(len(hcorrs), max_h), np.full(len(vcorrs), max_v)]
        limits = np.where(limits > 0, limits, np.inf)
        target = bba_state.get_orbit(bpms)
        for step in range(4):
            orbit = self._measure_state(os.path.join(samples_dir, f"SEXTUPOLES_ON_{step:04d}.pkl")).get_orbit(bpms)
            diff = np.nan_to_num(np.r_[orbit["x"] - target["x"], orbit["y"] - target["y"]])
            rms = np.sqrt(np.mean(diff ** 2))
            self.log(f"Orbit change after switching the sextupoles back on: rms {rms:.3f} {self.bpm_unit}")
            if rms <= 0.05 or step == 3:
                return
            current, _ = self._corrector_readings(correctors)
            if self.interface.set_correctors(correctors, np.clip(current - inverse @ diff, -limits, limits)) is False:
                raise RuntimeError("Could not confirm the corrector settings while restoring the orbit after switching the sextupoles back on.")

    def _corrector_readings(self, names):
        correctors = self.interface.get_correctors(names)
        index = [list(correctors["names"]).index(name) for name in names]
        return tuple(np.asarray(correctors[key], dtype=float).ravel()[index] for key in ("bdes", "bact"))

    def _print_screens(self, state, when):
        screens = state.get_screens()
        print(f"Screen values {when} correction:\nSigx: {screens['sigx']}\nSigy: {screens['sigy']}")
        if hasattr(self.interface, "get_twiss_at_screen"):
            print(f"Emittance {when} correction:")
            for screen_name in screens["names"]:
                twiss = self.interface.get_twiss_at_screen(screen_name)
                print(f"Emitt x for screen {screen_name}: {twiss['emitt_x']}\nEmitt y for screen {screen_name}: {twiss['emitt_y']}")

    def _append_abs_rms(self, x_vals, y_vals):
        mean_x = np.mean(x_vals, axis=0) if x_vals.ndim == 2 else x_vals
        mean_y = np.mean(y_vals, axis=0) if y_vals.ndim == 2 else y_vals
        self._hist_abs_rms_x.append(float(np.sqrt(np.mean(mean_x ** 2))))
        self._hist_abs_rms_y.append(float(np.sqrt(np.mean(mean_y ** 2))))
        self._hist_abs_rms_xy.append(float(np.sqrt(np.mean(mean_x ** 2 + mean_y ** 2))))

    def _history(self, kind):
        return [getattr(self, f"_hist_{kind}{suffix}") for suffix in ("_x", "_y", "", "_x_err", "_y_err", "_err")]

    def _append_history(self, kind, errors):
        mean_x, mean_y, std_x, std_y, mean_all, std_all = errors
        for values, value in zip(self._history(kind), (mean_x, mean_y, mean_all, std_x, std_y, std_all)):
            values.append(value)

    def _history_axes(self):
        return (("orbit", self.traj_ax, self.traj_canvas), ("disp", self.disp_ax, self.disp_canvas), ("wake", self.wake_ax, self.wake_canvas))

    def _redraw_history_plots(self):
        for kind, ax, canvas in self._history_axes():
            x, y, xy, x_err, y_err, xy_err = self._history(kind)
            self._plot_series(ax=ax, canvas=canvas, values_x=x, values_y=y, vals=xy, error_x=x_err, error_y=y_err, error_all=xy_err)
            if self._preview_history and self._preview_history[f"_hist_{kind}"]:
                for suffix, color, label in (("_x", "blue", "simulation x"), ("_y", "red", "simulation y"), ("", "gray", "simulation combined")):
                    values = self._preview_history[f"_hist_{kind}{suffix}"]
                    ax.plot(range(len(values)), values, linestyle="--", marker="x", alpha=0.6, color=color, label=label)
                ax.legend(fontsize=7, loc="upper right")
                canvas.draw_idle()

    def _stop_correction(self):
        self._cancel = True
        QMessageBox.information(self, "Correction", "Stop requested. Finishing current iteration...")
        self.log("Stop requested. Finishing current iteration...")

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
        for kind in ("orbit", "disp", "wake"):
            for values in self._history(kind):
                values.clear()
        self._hist_transmission.clear()
        self.initial_charge_value = None
        self._correction_dir = None
        self._preview_history = None
        self._redraw_history_plots()
        self._refresh_all_plot_popups()

    def _refresh_metric_plots_for_mode(self):
        self.plot_widget_4.setEnabled(True)
        self.plot_widget_5.setEnabled(True)
        self.disp_ax.set_facecolor("white")
        self.wake_ax.set_facecolor("white")
        for _, ax, canvas in self._history_axes():
            self._plot_series(ax, canvas, [], [], [])

    def _apply_jitter_subtraction_to_state(self, state):
        if self.jitter_model is None:
            return state
        state.bpms = apply_jitter_subtraction(state.get_bpms(), self.jitter_model)
        return state

    def handling(self, app_name, cwd=None, args=None):
        try:
            workdir = os.path.expanduser(os.path.expandvars(cwd or self.cwd))
            proc = QProcess(self)
            proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            proc.setWorkingDirectory(workdir)
            env = QProcessEnvironment.systemEnvironment()
            proc.setProcessEnvironment(env)
            argv = ["-m", os.path.splitext(os.path.basename(app_name))[0]] + list(args or [])
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
        selected_mode = next((index for index, button in enumerate(self.radio_buttons) if button.isChecked()), 0)
        if selected_mode == 0:
            args = ["--dir1", orbit_dir, "--compute"]
        else:
            data_dir, name = ((self.dfs_response_3.text(), "dispersion") if selected_mode == 1
                              else (self.wfs_response_3.text(), self._intensity_labels["directory"]))
            if not data_dir:
                QMessageBox.warning(self, "Warning", f"No {name} directory selected")
                return
            args = ["--dir1", data_dir, "--dir2", orbit_dir, "--diff", "--compute"]
        corrs, bpms = self._get_selection()
        self.handling('ComputeResponseMatrix_GUI.py', cwd=self.cwd, args=args + ["--correctors", *corrs, "--bpms", *bpms])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    from machine_interfaces.Backend import SelectInterface

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
