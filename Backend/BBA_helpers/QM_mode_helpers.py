import numpy as np

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QComboBox, QWidget, QMessageBox
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QComboBox, QWidget, QMessageBox

class QM_mode_helpers:

    def _setup_qm_controls(self):
        # row_mode = QHBoxLayout()
        # row_mode.addWidget(QLabel("Actuator mode"))
        #self.actuator_mode_combo = QComboBox(self)
        # actuator_mode_cls = self.actuator_mode.__class__
        # self.actuator_mode_combo.addItems([actuator_mode_cls.Kicker.value, actuator_mode_cls.QM.value])
        # row_mode.addWidget(self.actuator_mode_combo)
        # self.verticalLayout_3.insertLayout(0, row_mode)
        actuator_mode_cls = self.actuator_mode.__class__
        #self.actuator_mode_combo.clear()
        #self.actuator_mode_combo.addItems([actuator_mode_cls.Kicker.value, actuator_mode_cls.QM.value])
        #self.actuator_mode_combo.setCurrentText(self.actuator_mode.value)
        self.correctors_list.itemSelectionChanged.connect(self._refresh_specific_bpm_candidates)
        self.specific_bpm_row = QWidget(self)
        row_specific = QHBoxLayout(self.specific_bpm_row)
        row_specific.setContentsMargins(0, 0, 0, 0)
        self.specific_bpm_label = QLabel("Specific BPM (QM)")
        self.specific_bpm_combo = QComboBox(self)
        row_specific.addWidget(self.specific_bpm_label)
        row_specific.addWidget(self.specific_bpm_combo)
        self.verticalLayout_3.insertWidget(4, self.specific_bpm_row)
        self.specific_bpm_row.setFixedHeight(self.specific_bpm_row.sizeHint().height())

        #self.actuator_mode_combo.currentTextChanged.connect(self._on_actuator_mode_changed)
        self.bpms_list.itemSelectionChanged.connect(self._refresh_specific_bpm_candidates)
        self._refresh_specific_bpm_candidates()

    def _refresh_specific_bpm_candidates(self):
        actuator_mode_cls = self.actuator_mode.__class__
        if not hasattr(self, "specific_bpm_combo"):
            return
        current = self.specific_bpm_combo.currentText()
        bpms = []
        for i in range(self.bpms_list.count()):
            it = self.bpms_list.item(i)
            if it.isSelected():
                bpms.append(it.data(Qt.ItemDataRole.UserRole))
        if not bpms:
            bpms = list(self.initial_state.get_bpms()["names"])
        self.specific_bpm_combo.blockSignals(True)
        self.specific_bpm_combo.clear()
        if self.actuator_mode == actuator_mode_cls.QM:
            qcorrs = []
            for i in range(self.correctors_list.count()):
                it = self.correctors_list.item(i)
                if it.isSelected():
                    qcorrs.append(it.text())
            if not qcorrs:
                qcorrs = list(self.qm_corrs)
            bpms = self._qm_control_bpms(qcorrs, bpms)
        self.specific_bpm_combo.addItems([str(b) for b in bpms])
        idx = self.specific_bpm_combo.findText(current)
        if idx >= 0:
            self.specific_bpm_combo.setCurrentIndex(idx)
        self.specific_bpm_combo.blockSignals(False)

    def _update_qm_widgets_visibility(self):
        actuator_mode_cls = self.actuator_mode.__class__
        is_qm = self.actuator_mode == actuator_mode_cls.QM
        self.specific_bpm_row.setVisible(True)
        self.specific_bpm_label.setVisible(is_qm)
        self.specific_bpm_combo.setVisible(is_qm)

        if is_qm:
            self.groupBox_6.setTitle("DFS not used in QM mode")
            self.groupBox_7.setTitle("WFS not used in QM mode")
            self.label.setText("Trajectory hold weight")
            self.lineEdit.setText("0.0")
            self.lineEdit_2.setText("1.0")
            self.lineEdit_3.setText("0.0")
            self.lineEdit_4.setText("0.000001")
            self.label_2.setText("BPM->0 weight")
            self.label_3.setText("Specific BPM->0 weight")
            self.current_groupbox_right.setTitle("Max range [um]")
            self.horizontal_current_label.setText("X:")
            self.vertical_current_label.setText("Y:")
            self.max_horizontal_current_spinbox.setMaximum(1e6)
            self.max_vertical_current_spinbox.setMaximum(1e6)
            self.max_horizontal_current_spinbox.setDecimals(0)
            self.max_vertical_current_spinbox.setDecimals(0)
            self.max_horizontal_current_spinbox.setValue(1000.0)
            self.max_vertical_current_spinbox.setValue(1000.0)
        else:
            self.groupBox_6.setTitle("Dispersion-Free Steering")
            self.groupBox_7.setTitle("Wakefield-Free Steering")
            self.label.setText("Orbit weight")
            self.label_2.setText("Dispersion weight")
            self.label_3.setText("Wakefield weight")
            self.lineEdit.setText("1.0")
            self.lineEdit_2.setText("10.0")
            self.lineEdit_3.setText("10.0")
            self.lineEdit_4.setText("0.001")
            self.current_groupbox_right.setTitle(f"Max strength ({self.corrs_unit})")
            self.horizontal_current_label.setText("H:")
            self.vertical_current_label.setText("V:")
            self.max_horizontal_current_spinbox.setMaximum(99.99)
            self.max_vertical_current_spinbox.setMaximum(99.99)
            self.max_horizontal_current_spinbox.setDecimals(2)
            self.max_vertical_current_spinbox.setDecimals(2)
            self.max_horizontal_current_spinbox.setSingleStep(0.01)
            self.max_vertical_current_spinbox.setSingleStep(0.01)
            self.max_horizontal_current_spinbox.setValue(0.0)
            self.max_vertical_current_spinbox.setValue(0.0)

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

    def _start_qm_correction(self, silent=False, preserve_plots=False):
        if not hasattr(self.interface, "apply_qmag_xyroll"):
            raise RuntimeError("Interface does not support quadrupole mover correction")

        self._cancel = False
        if not preserve_plots:
            self._hist_orbit_x.clear()
            self._hist_orbit_y.clear()
            self._hist_orbit.clear()
            self._hist_disp_x.clear()
            self._hist_disp_y.clear()
            self._hist_disp.clear()
            self._hist_wake_x.clear()
            self._hist_wake_y.clear()
            self._hist_wake.clear()
            self._refresh_metric_plots_for_mode()

        w1, w2, w3, rcond, iters, gain, beta = self._read_params()

        qcorrs, bpms_selected = self._get_selection()
        self._build_jitter_model_for_correction(actuators=qcorrs, bpms=bpms_selected)
        bpms = self._qm_control_bpms(qcorrs, bpms_selected)
        if self.jitter_model is not None:
            refs = set(self.jitter_model["reference_bpms"])
            bpms = [bpm for bpm in bpms if bpm not in refs]
            self.log("Removed jitter reference BPMs from correction targets")

        if len(qcorrs) == 0:
            raise RuntimeError("No quadrupoles selected")

        if len(bpms) == 0:
            raise RuntimeError("No BPMs selected")

        spec_bpm = self.specific_bpm_combo.currentText().strip()
        response = self._creating_qm_response_matrices(selected_corrs=qcorrs, selected_bpms=bpms, triangular=True)

        if response is not None:
            qcorrs = list(response["qcorrs"])
            bpms = list(response["bpms"])
            self.log("Using measured QM response matrix from loaded trajectory data")
        else:
            self.log("No loaded QM response matrix data")
            return
        Rxx_raw = np.asarray(response["R_xx"], dtype=float) # + np.asarray(response["T_xx"], dtype=float)
        Rxy_raw = np.asarray(response["R_xy"], dtype=float)
        Ryx_raw = np.asarray(response["R_yx"], dtype=float)
        Ryy_raw = np.asarray(response["R_yy"], dtype=float) # + np.asarray(response["T_yy"], dtype=float)

        def _log_matrix_stats(name, mat):
            arr = np.asarray(mat, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                self.log(f"QM response {name}: all values are non-finite")
                return
            self.log(
                f"QM response {name}: shape={arr.shape}, "
                f"max|R|={float(np.max(np.abs(finite))):.6g} mm/um, "
                f"median|R|={float(np.median(np.abs(finite))):.6g} mm/um"
            )

        _log_matrix_stats("Rxx", Rxx_raw)
        _log_matrix_stats("Rxy", Rxy_raw)
        _log_matrix_stats("Ryx", Ryx_raw)
        _log_matrix_stats("Ryy", Ryy_raw)

        Rxx = np.nan_to_num(Rxx_raw, nan=0.0, posinf=0.0, neginf=0.0)
        Rxy = np.nan_to_num(Rxy_raw, nan=0.0, posinf=0.0, neginf=0.0)
        Ryx = np.nan_to_num(Ryx_raw, nan=0.0, posinf=0.0, neginf=0.0)
        Ryy = np.nan_to_num(Ryy_raw, nan=0.0, posinf=0.0, neginf=0.0)

        max_x = self.max_horizontal_current_spinbox.value()
        max_y = self.max_vertical_current_spinbox.value()

        B0x = None
        B0y = None

        for it in range(iters):
            if self._cancel:
                break

            state = self.interface.get_state()
            state = self._apply_jitter_subtraction_to_state(state)
            orbit = state.get_orbit(bpms)

            O0x = np.asarray(orbit["x"], dtype=float).reshape(-1, 1)
            O0y = np.asarray(orbit["y"], dtype=float).reshape(-1, 1)

            if B0x is None:
                B0x = O0x.copy()
                B0y = O0y.copy()

            blocks_A = []
            blocks_B = []

            if w1 > 0:
                top = np.hstack((w1 * Rxx, w1 * Rxy))
                bottom = np.hstack((w1 * Ryx, w1 * Ryy))

                blocks_A.extend((top, bottom))
                blocks_B.extend((w1 * (O0x - B0x), w1 * (O0y - B0y)))

            if w2 > 0:
                top = np.hstack((w2 * Rxx, w2 * Rxy))
                bottom = np.hstack((w2 * Ryx, w2 * Ryy))

                blocks_A.extend((top, bottom))
                blocks_B.extend((w2 * O0x, w2 * O0y))

            if w3 > 0 and spec_bpm in bpms:
                idx = bpms.index(spec_bpm)

                row_top = np.hstack((Rxx[idx:idx+1, :], Rxy[idx:idx+1, :]))
                row_bottom = np.hstack((Ryx[idx:idx+1, :], Ryy[idx:idx+1, :]))

                blocks_A.extend((w3 * row_top, w3 * row_bottom))
                blocks_B.extend((w3 * O0x[idx:idx+1, :], w3 * O0y[idx:idx+1, :]))

            A = np.vstack(blocks_A)
            B = np.vstack(blocks_B)

            A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
            B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

            if it == 0:
                try:
                    singular_values = np.linalg.svd(A, compute_uv=False)
                    if singular_values.size:
                        cutoff = float(rcond) * float(np.max(singular_values))
                        kept = int(np.sum(singular_values > cutoff))
                        self.log(
                            f"QM A SVD: shape={A.shape}, smax={float(np.max(singular_values)):.6g}, "
                            f"smin={float(np.min(singular_values)):.6g}, cutoff={cutoff:.6g}, kept={kept}/{singular_values.size}"
                        )
                except Exception as exc:
                    self.log(f"QM A SVD diagnostic failed: {exc}")

            delta = -gain * (np.linalg.pinv(A, rcond=rcond) @ B).reshape(-1)
            self.log(
                f"QM iteration {it}: ||B||={float(np.linalg.norm(B)):.6g}, "
                f"max|delta|={float(np.max(np.abs(delta))) if delta.size else 0.0:.6g} um"
            )

            nq = len(qcorrs)
            dx = delta[:nq]
            dy = delta[nq:nq + nq]

            qstate = self.interface.get_quadrupoles(qcorrs)

            x_now = np.asarray(qstate["xdes"], dtype=float)
            y_now = np.asarray(qstate["ydes"], dtype=float)
            r_now = np.asarray(qstate["rolldes"], dtype=float)

            x_target = np.clip(x_now + dx, -max_x, max_x)
            y_target = np.clip(y_now + dy, -max_y, max_y)

            self.log(
                f"QM iteration {it}: max|target_x|={float(np.max(np.abs(x_target))) if x_target.size else 0.0:.6g} um, "
                f"max|target_y|={float(np.max(np.abs(y_target))) if y_target.size else 0.0:.6g} um"
            )

            self.interface.apply_qmag_xyroll(qcorrs, x_target, y_target, r_now, wait=True)

            self._hist_orbit_x.append(float(np.linalg.norm(np.nan_to_num(O0x - B0x))))
            self._hist_orbit_y.append(float(np.linalg.norm(np.nan_to_num(O0y - B0y))))
            self._hist_orbit.append(self._hist_orbit_x[-1] + self._hist_orbit_y[-1])

            self._plot_series(ax=self.traj_ax, canvas=self.traj_canvas, values_x=self._hist_orbit_x, values_y=self._hist_orbit_y, vals=self._hist_orbit, title="QM - distance from initial trajectory")

            QApplication.processEvents()

        if not silent:
            self.save_session_settings_qm_correction(w1=w1, w2=w2, w3=w3, specific_bpm=spec_bpm, rcond=rcond, iters=iters, gain=gain, beta=beta, max_horizontal_range=max_x, max_vertical_range=max_y, is_triangular=bool(self.triangular_checkbox.isChecked()), bpm_weights=self.bpm_weights, response=response, is_jitter_subtraction_checked=bool(self.subtract_jitter_checkbox.isChecked()))
            QMessageBox.information(self, "QM correction", "QM correction finished")
