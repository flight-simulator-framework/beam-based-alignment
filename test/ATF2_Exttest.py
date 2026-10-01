import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from BBA_GUI import MainWindow
from machine_interfaces.Backend.State import State

app = QApplication(sys.argv)

Interface = type(
    'InterfaceATF2_Ext',
    (),
    {"__module__": 'machine_interfaces_atf2.InterfaceATF2_Ext'},
)
interface = Interface()
interface.nsamples = 3
interface.energy_frequency_offset_khz = 4.0
interface.nominal_laser_intensity = 0.1
interface.test_laser_intensity = 0.125
interface.get_name = lambda: 'ATF2_Ext'
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
