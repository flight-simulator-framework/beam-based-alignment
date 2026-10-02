import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from BBA_GUI import MainWindow
from machine_interfaces.Backend.State import State

app = QApplication(sys.argv)

Interface = type(
    'InterfaceATF2_Linac',
    (),
    {"__module__": 'machine_interfaces_atf2.InterfaceATF2_Linac'},
)
interface = Interface()
interface.nsamples = 3
interface.phase_kl1 = 0.0
interface.cm1l_test_phase = 5.0
interface.test_laser_intensity = 0.15
interface.get_name = lambda: 'ATF2_Linac'
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
