import os
import sys
from datetime import datetime

import matplotlib
import numpy as np
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ChangeBpmsWeights_BBA import ChangeBpmsWeights_BBA
from LogConsole_BBA import LogConsole
from SaveOrLoad_BBA import SaveOrLoad_BBA
from jitter_subtraction import explain_reference_selection, fit_jitter_model, apply_jitter_subtraction
from SelectInterface import choose_acc_and_interface

matplotlib.use("QtAgg")

OUTLIER_FACTOR = 10.0
NOMINAL_JITTER_SHOTS = 30


def reject_large_outliers(values, factor=OUTLIER_FACTOR):
    arr = np.asarray(values, dtype=float).copy()
    if arr.ndim != 2 or arr.size == 0:
        return arr
    med_abs = np.nanmedian(np.abs(arr), axis=0)
    threshold = factor * med_abs
    for j, thr in enumerate(threshold):
        if not np.isfinite(thr) or thr <= 0:
            continue
        mask = np.isfinite(arr[:, j]) & (np.abs(arr[:, j]) > thr)
        arr[mask, j] = np.nan
    return arr


def orbit_from_bpms(bpms, names=None):
    all_names = list(bpms.get("names", []))
    x_all = np.asarray(bpms.get("x", []), dtype=float)
    y_all = np.asarray(bpms.get("y", []), dtype=float)
    t_all = np.asarray(bpms.get("tmit", []), dtype=float)

    if names is not None:
        m = {str(n): i for i, n in enumerate(all_names)}
        idx = [m[str(n)] for n in names if str(n) in m]
        names_use = [all_names[i] for i in idx]
        x_all = x_all[:, idx]
        y_all = y_all[:, idx]
        t_all = t_all[:, idx]
    else:
        names_use = all_names

    x_all = reject_large_outliers(x_all)
    y_all = reject_large_outliers(y_all)
    x = np.mean(x_all, axis=0)
    y = np.mean(y_all, axis=0)
    stdx = np.std(x_all, axis=0)
    stdy = np.std(y_all, axis=0)
    tmit = np.mean(t_all, axis=0)
    faulty = np.isnan(x) | np.isnan(y)
    x[faulty] = np.nan
    y[faulty] = np.nan
    return {"names": names_use, "x": x, "y": y, "stdx": stdx, "stdy": stdy, "tmit": tmit, "faulty": faulty, "nbpms": len(names_use)}


class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 2.4), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)


class MainWindow(QMainWindow, SaveOrLoad_BBA):
    def __init__(self, interface, dir_name):
        super().__init__()
        self.cwd = os.getcwd()
        self.interface = interface
        self.dir_name = dir_name
        self._cancel = False
        self._running = False
        self.log_console = None
        self._bpm_names_cache = list(self.interface.get_bpms()["names"])
        self._qm_names_cache = list(self.interface.get_quadrupoles_names()) if hasattr(self.interface, "get_quadrupoles_names") else []
        self._hist_orbit = []
        self._hist_orbit_x = []
        self._hist_orbit_y = []
        self._hist_disp = []
        self._hist_disp_x = []
        self._hist_disp_y = []
        self._hist_wake = []
        self._hist_wake_x = []
        self._hist_wake_y = []

        ui_path = os.path.join(os.path.dirname(__file__), "BBA_GUI.ui")
        uic.loadUi(ui_path, self)
        self.setWindowTitle("BBA_QM_Sequential_GUI")

        self._setup_canvases()
        self._setup_controls()
        self._populate_lists()
        self._refresh_specific_bpm_candidates()
        self._refresh_metric_plots()

    def _setup_canvases(self):
        def install(host):
            canvas = MatplotlibWidget(host)
            layout = host.layout()
            if layout is None:
                layout = QVBoxLayout(host)
                layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(canvas)
            return canvas, canvas.axes

        self.traj_canvas, self.traj_ax = install(self.plot_widget_3)
        self.disp_canvas, self.disp_ax = install(self.plot_widget_4)
        self.wake_canvas, self.wake_ax = install(self.plot_widget_5)

    def _setup_controls(self):
        self.pushButton_log.clicked.connect(self._show_console_log)
        if hasattr(self, "clear_graphs_button"):
            self.clear_graphs_button.clicked.connect(self._clear_graphs)
        self.load_correctors_button.clicked.connect(self._load_correctors)
        self.load_bpms_button.clicked.connect(self._load_bpms)
        self.start_button.clicked.connect(self._on_start_click)
        self.stop_button.clicked.connect(self._stop_correction)
        self.compute_response_matrix_button.setVisible(False)

        for widget_name in ("pushButton_8", "pushButton_9", "pushButton_10"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setVisible(False)

        for widget_name in ("trajectory_response_3", "dfs_response_3", "wfs_response_3"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setVisible(False)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["Sequential QM"])
        row_mode.addWidget(self.mode_combo)
        self.verticalLayout_3.insertLayout(0, row_mode)

        row_specific = QHBoxLayout()
        self.specific_bpm_label = QLabel("Next target BPM")
        self.specific_bpm_combo = QComboBox(self)
        row_specific.addWidget(self.specific_bpm_label)
        row_specific.addWidget(self.specific_bpm_combo)
        self.verticalLayout_3.insertLayout(4, row_specific)

        self.bpms_list.itemSelectionChanged.connect(self._refresh_specific_bpm_candidates)
        self.correctors_list.itemSelectionChanged.connect(self._refresh_specific_bpm_candidates)

        self.label.setText("Trajectory hold weight")
        self.label_2.setText("BPM Absolute Mean")
        self.label_3.setText("Unused")
        self.current_label.setText("Max range [um]")
        self.horizontal_current_label.setText("X:")
        self.vertical_current_label.setText("Y:")
        self.max_horizontal_current_spinbox.setMaximum(1e6)
        self.max_vertical_current_spinbox.setMaximum(1e6)
        self.max_horizontal_current_spinbox.setDecimals(0)
        self.max_vertical_current_spinbox.setDecimals(0)
        self.max_horizontal_current_spinbox.setValue(1000.0)
        self.max_vertical_current_spinbox.setValue(1000.0)
        self.lineEdit.setText("1")
        self.lineEdit_2.setText("1")
        self.lineEdit_3.setText("0")
        self.lineEdit_4.setText("0.001")
        self.lineEdit_5.setText("2")
        self.lineEdit_6.setText("0.7")
        self.session_database_3.setText(self.dir_name)

    def _populate_lists(self):
        self.correctors_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.bpms_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.correctors_list.insertItems(0, self._qm_names_cache)
        self.bpms_list.insertItems(0, self._bpm_names_cache)

    def _read_params(self):
        def getf(name, default):
            widget = getattr(self, name)
            txt = (widget.text() or "").strip()
            try:
                return float(txt) if txt else float(default)
            except ValueError:
                widget.setText(f"{default:g}")
                return float(default)

        def geti(name, default):
            widget = getattr(self, name)
            txt = (widget.text() or "").strip()
            try:
                return int(float(txt)) if txt else int(default)
            except ValueError:
                widget.setText(str(int(default)))
                return int(default)

        w1 = getf("lineEdit", 1.0)
        w2 = getf("lineEdit_2", 1.0)
        iters = geti("lineEdit_5", 2)
        gain = getf("lineEdit_6", 0.7)
        return w1, w2, iters, gain

    def _get_selection(self):
        qcorrs_all = list(self._qm_names_cache)
        bpms_all = list(self._bpm_names_cache)
        qcorrs = [it.text() for it in self.correctors_list.selectedItems()] or qcorrs_all
        bpms = [it.text() for it in self.bpms_list.selectedItems()] or bpms_all
        return qcorrs, bpms

    def _maybe_cancel(self, where=""):
        QApplication.processEvents()
        if self._cancel:
            if where:
                self.log(f"Stopping after {where}.")
            else:
                self.log("Stopping.")
            return True
        return False

    def _qm_control_bpms(self, qcorrs, bpms):
        seq = self.interface.get_sequence()
        order = {str(name): idx for idx, name in enumerate(seq)}
        qpos = []
        for name in qcorrs:
            key = str(name)
            if key in order:
                qpos.append(order[key])
            elif f"M{key}" in order:
                qpos.append(order[f"M{key}"])
        if not qpos:
            return list(bpms)
        threshold = min(qpos)
        out = []
        for name in bpms:
            key = str(name)
            pos = order.get(key, order.get(f"M{key}", 10**9))
            if pos >= threshold:
                out.append(key)
        return out

    def _qm_reference_bpms(self, qcorrs, bpms):
        return explain_reference_selection(bpms, qcorrs, self.interface.get_sequence(), min_refs=2)

    def _refresh_specific_bpm_candidates(self):
        current = self.specific_bpm_combo.currentText() if hasattr(self, "specific_bpm_combo") else ""
        qcorrs, bpms_all = self._get_selection()
        bpms = self._qm_control_bpms(qcorrs, bpms_all)
        self.specific_bpm_combo.blockSignals(True)
        self.specific_bpm_combo.clear()
        self.specific_bpm_combo.addItems(bpms)
        idx = self.specific_bpm_combo.findText(current)
        if idx >= 0:
            self.specific_bpm_combo.setCurrentIndex(idx)
        self.specific_bpm_combo.blockSignals(False)

    def _plot_series(self, ax, canvas, values_x, values_y, vals, title=None, ylabel="[mm]"):
        ax.clear()
        if values_x:
            ax.plot(range(1, len(values_x) + 1), values_x, marker="o", color="red", label="x")
        if values_y:
            ax.plot(range(1, len(values_y) + 1), values_y, marker="o", color="blue", label="y")
        if vals:
            ax.plot(range(1, len(vals) + 1), vals, linestyle="dashed", color="black", label="combined norm")
        if values_x or values_y:
            ax.legend(fontsize=7)
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.grid(True, alpha=0.3)
        canvas.draw_idle()

    def _refresh_metric_plots(self):
        self._plot_series(self.traj_ax, self.traj_canvas, [], [], [], title="Trajectory Hold", ylabel="[mm]")
        self._plot_series(self.disp_ax, self.disp_canvas, [], [], [], title="BPM Absolute Mean", ylabel="[mm]")
        self._plot_series(self.wake_ax, self.wake_canvas, [], [], [], title="Unused In Sequential QM", ylabel="[mm]")

    def _clear_graphs(self):
        self._cancel = True
        self._hist_orbit_x.clear()
        self._hist_orbit_y.clear()
        self._hist_orbit.clear()
        self._hist_disp_x.clear()
        self._hist_disp_y.clear()
        self._hist_disp.clear()
        self._hist_wake_x.clear()
        self._hist_wake_y.clear()
        self._hist_wake.clear()
        self._refresh_metric_plots()

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
            self.log_console.show()
        self.log_console.log(line)

    def _acquire_nominal_jitter_snapshot(self):
        self.log(f"Acquiring nominal shots for jitter subtraction ({NOMINAL_JITTER_SHOTS} shots)")
        original = getattr(self.interface, "nsamples", None)
        try:
            if original is not None:
                self.interface.nsamples = NOMINAL_JITTER_SHOTS
            return self.interface.get_bpms()
        finally:
            if original is not None:
                self.interface.nsamples = original

    def _build_live_jitter_model(self, qcorrs, bpms_all):
        ref_bpms, ref_reason = self._qm_reference_bpms(qcorrs, bpms_all)
        if not ref_bpms:
            self.log(f"Jitter subtraction disabled: {ref_reason}")
            return None
        if self._maybe_cancel("reference BPM selection"):
            return None
        nominal_bpms = self._acquire_nominal_jitter_snapshot()
        if self._maybe_cancel("nominal jitter acquisition"):
            return None
        model, fit_reason = fit_jitter_model([nominal_bpms], ref_bpms, bpms_all)
        if model is None:
            self.log(f"Jitter subtraction disabled: {fit_reason}")
            return None
        self.log(f"Jitter subtraction enabled with refs: {', '.join(ref_bpms)}")
        return model

    def _usable_bpms_from_snapshot(self, bpm_snapshot, requested_bpms):
        all_names = list(bpm_snapshot.get("names", []))
        x_all = np.asarray(bpm_snapshot.get("x", []), dtype=float)
        y_all = np.asarray(bpm_snapshot.get("y", []), dtype=float)
        name_to_idx = {str(name): idx for idx, name in enumerate(all_names)}
        usable = []
        skipped = []
        for bpm in requested_bpms:
            idx = name_to_idx.get(str(bpm))
            if idx is None:
                skipped.append(str(bpm))
                continue
            x_col = x_all[:, idx] if x_all.ndim == 2 and idx < x_all.shape[1] else np.asarray([], dtype=float)
            y_col = y_all[:, idx] if y_all.ndim == 2 and idx < y_all.shape[1] else np.asarray([], dtype=float)
            finite_x = x_col[np.isfinite(x_col)]
            finite_y = y_col[np.isfinite(y_col)]
            x_zero = finite_x.size == 0 or np.all(np.isclose(finite_x, 0.0))
            y_zero = finite_y.size == 0 or np.all(np.isclose(finite_y, 0.0))
            if x_zero and y_zero:
                skipped.append(str(bpm))
                continue
            usable.append(str(bpm))
        return usable, skipped

    def _find_attached_bpm(self, qcorr, control_bpms):
        token = f"M{qcorr}"
        for bpm in control_bpms:
            if token in str(bpm):
                return str(bpm)
        return None

    def _attached_bpms_for_qcorrs(self, qcorrs, control_bpms):
        attached = []
        for qcorr in qcorrs:
            bpm = self._find_attached_bpm(qcorr, control_bpms)
            if bpm is not None and bpm not in attached:
                attached.append(bpm)
        return attached

    def _subset_orbit(self, orbit, names):
        if not names:
            return orbit
        name_to_idx = {str(name): idx for idx, name in enumerate(orbit["names"])}
        idx = [name_to_idx[str(name)] for name in names if str(name) in name_to_idx]
        if not idx:
            return {"names": [], "x": np.asarray([], dtype=float), "y": np.asarray([], dtype=float)}
        return {
            "names": [orbit["names"][i] for i in idx],
            "x": np.asarray(orbit["x"], dtype=float)[idx],
            "y": np.asarray(orbit["y"], dtype=float)[idx],
        }

    def _record_history(self, orbit, baseline, mean_bpms):
        O0x = orbit["x"].reshape(-1, 1)
        O0y = orbit["y"].reshape(-1, 1)
        B0x, B0y = baseline
        self._hist_orbit_x.append(float(np.linalg.norm(np.nan_to_num(O0x - B0x))))
        self._hist_orbit_y.append(float(np.linalg.norm(np.nan_to_num(O0y - B0y))))
        self._hist_orbit.append(self._hist_orbit_x[-1] + self._hist_orbit_y[-1])
        orbit_mean = self._subset_orbit(orbit, mean_bpms)
        mean_x = np.asarray(orbit_mean["x"], dtype=float).reshape(-1, 1)
        mean_y = np.asarray(orbit_mean["y"], dtype=float).reshape(-1, 1)
        self._hist_disp_x.append(float(np.nanmean(np.abs(mean_x))) if mean_x.size else np.nan)
        self._hist_disp_y.append(float(np.nanmean(np.abs(mean_y))) if mean_y.size else np.nan)
        self._hist_disp.append(0.5 * (self._hist_disp_x[-1] + self._hist_disp_y[-1]))
        self._hist_wake_x.append(0.0)
        self._hist_wake_y.append(0.0)
        self._hist_wake.append(0.0)
        self._plot_series(self.traj_ax, self.traj_canvas, self._hist_orbit_x, self._hist_orbit_y, self._hist_orbit, title="Trajectory Hold", ylabel="[mm]")
        self._plot_series(self.disp_ax, self.disp_canvas, self._hist_disp_x, self._hist_disp_y, self._hist_disp, title="BPM Absolute Mean", ylabel="[mm]")
        self._plot_series(self.wake_ax, self.wake_canvas, self._hist_wake_x, self._hist_wake_y, self._hist_wake, title="Unused In Sequential QM", ylabel="[mm]")
        QApplication.processEvents()

    def _on_start_click(self):
        if self._running:
            return
        self._running = True
        self.start_button.setEnabled(False)
        try:
            self._start_sequential_qm()
        finally:
            self._running = False
            self.start_button.setEnabled(True)

    def _start_sequential_qm(self):
        if not hasattr(self.interface, "get_quadrupoles") or not hasattr(self.interface, "apply_qmag_xyroll"):
            raise RuntimeError("Interface does not provide QM APIs")

        self._cancel = False
        w_hold, _w_abs, n_passes, gain = self._read_params()
        qcorrs, bpms_all = self._get_selection()
        control_bpms = self._qm_control_bpms(qcorrs, bpms_all)
        if not qcorrs:
            raise RuntimeError("No QM correctors selected")
        if not control_bpms:
            raise RuntimeError("No controllable downstream BPMs selected")

        self.log(f"Selected QMs: {', '.join(qcorrs)}")
        self.log(f"Control BPMs: {', '.join(control_bpms)}")
        mean_bpms = self._attached_bpms_for_qcorrs(qcorrs, control_bpms)
        self.log(f"BPM Absolute Mean targets: {', '.join(mean_bpms)}" if mean_bpms else "BPM Absolute Mean targets: none")
        jitter_model = self._build_live_jitter_model(qcorrs, bpms_all)
        if self._cancel:
            self.log("Sequential QM correction cancelled.")
            return

        max_x = float(self.max_horizontal_current_spinbox.value())
        max_y = float(self.max_vertical_current_spinbox.value())

        def clamp(val, lim):
            if lim <= 0:
                return float(val)
            return float(np.clip(val, -lim, lim))

        baseline_orbit = None
        for ipass in range(n_passes):
            if self._maybe_cancel(f"pass {ipass} setup"):
                break
            self.log(f"Starting sequential pass {ipass + 1}/{n_passes}")
            for qcorr in qcorrs:
                if self._maybe_cancel(f"{qcorr} start"):
                    break
                attached_bpm = self._find_attached_bpm(qcorr, control_bpms)
                if attached_bpm is None:
                    self.log(f"Skipping {qcorr}: no attached downstream BPM found")
                    continue
                bpm_snapshot = self.interface.get_bpms()
                if self._maybe_cancel(f"{qcorr} BPM acquisition"):
                    break
                usable_bpms, skipped_bpms = self._usable_bpms_from_snapshot(bpm_snapshot, control_bpms)
                if skipped_bpms:
                    self.log(f"Skipping zero BPMs for this shot: {', '.join(skipped_bpms)}")
                if attached_bpm not in usable_bpms:
                    self.log(f"Skipping {qcorr}: attached BPM {attached_bpm} is stuck at zero")
                    continue
                if not usable_bpms:
                    self.log(f"Skipping {qcorr}: no usable downstream BPMs in this shot")
                    continue
                if jitter_model is not None:
                    bpm_snapshot = apply_jitter_subtraction(bpm_snapshot, jitter_model)
                orbit = orbit_from_bpms(bpm_snapshot, usable_bpms)
                if baseline_orbit is None:
                    baseline_orbit = (orbit["x"].reshape(-1, 1).copy(), orbit["y"].reshape(-1, 1).copy())
                idx = orbit["names"].index(attached_bpm)
                bx_mm = float(orbit["x"][idx])
                by_mm = float(orbit["y"][idx])
                dx_um = clamp(gain * 1000.0 * bx_mm, max_x)
                dy_um = clamp(gain * 1000.0 * by_mm, max_y)
                q0 = self.interface.get_quadrupoles([qcorr])
                if len(q0.get("names", [])) == 0:
                    self.log(f"Skipping {qcorr}: quadrupole readback unavailable")
                    continue
                x_now = float(np.asarray(q0["xdes"], dtype=float)[0])
                y_now = float(np.asarray(q0["ydes"], dtype=float)[0])
                r_now = float(np.asarray(q0["rolldes"], dtype=float)[0])
                self.log(
                    f"{qcorr} -> {attached_bpm}: "
                    f"bpm_x={bx_mm:.4g} mm, bpm_y={by_mm:.4g} mm, "
                    f"move_x={dx_um:.3f} um, move_y={dy_um:.3f} um"
                )
                self.interface.apply_qmag_xyroll(qcorr, x_now + dx_um, y_now + dy_um, r_now, wait=True)
                if self._maybe_cancel(f"{qcorr} move"):
                    break
                bpm_after = self.interface.get_bpms()
                if self._maybe_cancel(f"{qcorr} post-move BPM acquisition"):
                    break
                usable_after, skipped_after = self._usable_bpms_from_snapshot(bpm_after, control_bpms)
                if skipped_after:
                    self.log(f"Skipping zero BPMs after move: {', '.join(skipped_after)}")
                if not usable_after:
                    self.log(f"Stopping history update after {qcorr}: no usable downstream BPMs after move")
                    continue
                if jitter_model is not None:
                    bpm_after = apply_jitter_subtraction(bpm_after, jitter_model)
                orbit_after = orbit_from_bpms(bpm_after, usable_after)
                if baseline_orbit is None or baseline_orbit[0].shape[0] != orbit_after["x"].reshape(-1, 1).shape[0]:
                    baseline_orbit = (orbit_after["x"].reshape(-1, 1).copy(), orbit_after["y"].reshape(-1, 1).copy())
                usable_mean_bpms = [bpm for bpm in mean_bpms if bpm in orbit_after["names"]]
                self._record_history(orbit_after, baseline_orbit, usable_mean_bpms)

        if self._cancel:
            self.log("Sequential QM correction stopped.")
            QMessageBox.information(self, "Sequential QM", "Sequential QM correction stopped.")
        else:
            self.log("Sequential QM correction finished.")
            QMessageBox.information(self, "Sequential QM", "Sequential QM correction finished.")

    def _stop_correction(self):
        self._cancel = True
        self.log("Stop requested. Will stop after the current blocking operation.")


if __name__ == "__main__":
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)

    interface = choose_acc_and_interface()
    if interface is None:
        sys.exit(0)

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"~/CERN-Flight_Simulator-Data/BBA_QM_Sequential_{interface.get_name()}{time_str}"
    dir_name = os.path.expanduser(os.path.expandvars(dir_name))
    os.makedirs(dir_name, exist_ok=True)

    window = MainWindow(interface, dir_name)
    window.show()
    sys.exit(app.exec())
