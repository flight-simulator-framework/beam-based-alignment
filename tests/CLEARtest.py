import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from BBA_GUI import MainWindow
from machine_interfaces.Backend.State import State

app = QApplication(sys.argv)

Interface = type(
    "CLEAR_real_machine",
    (),
    {"__module__": "machine_interfaces_clear.InterfaceCLEAR"},
)
interface = Interface()
interface.nsamples = 3
interface.rf_phase_nominal = 128.0
interface.rf_phase_test = 108.0
interface.uvatt2_test_steps = 1000.0
interface.quadrupole_scaling_factor = 0.5
interface.get_name = lambda: "CLEAR"
interface.get_state = lambda: State()
interface.get_hcorrectors_names = lambda: ["H"]
interface.get_vcorrectors_names = lambda: ["V"]
interface.get_correctors = lambda: {
    "names": ["H", "V"],
    "bdes": np.array([1.0, 1.0]),
}

window = MainWindow(interface, "", start_state=State())
window.show()
sys.exit(app.exec())