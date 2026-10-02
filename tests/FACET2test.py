import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from BBA_GUI import MainWindow
from machine_interfaces.Backend.State import State

app = QApplication(sys.argv)

Interface = type(
    'InterfaceFACET2_Linac',
    (),
    {"__module__": 'machine_interfaces_facet2.InterfaceFACET2_Linac'},
)
interface = Interface()
interface.nsamples = 10
interface.bba_bc11_energy_offset_mev = -3.0
interface.bba_downstream_energy_offset_mev = -40.0
interface.bba_uvwp_offset_deg = -2.5
interface.get_name = lambda: 'FACET2_Linac'
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
