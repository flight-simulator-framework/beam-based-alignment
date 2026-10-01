import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from BBA_GUI import MainWindow
from machine_interfaces.Backend.State import State

app = QApplication(sys.argv)

Interface = type(
    'InterfaceESRF_TL2',
    (),
    {"__module__": 'machine_interfaces_esrf.InterfaceESRF_TL2'},
)
interface = Interface()
interface.nsamples = 3
interface.energy_delta = 1.0
interface.get_name = lambda: 'ESRF_TL2_Ext'
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
