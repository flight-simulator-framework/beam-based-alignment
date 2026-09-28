# Beam-Based Alignment

Correct the beam trajectory using measured response matrices and either corrector magnets or quadrupole movers. The application calculates actuator changes, applies them iteratively, and displays the measured correction progress.

In corrector mode, the correction can combine orbit steering, dispersion-free steering (DFS) and wakefield-free steering (WFS). DFS uses measurements at different beam energies; WFS uses measurements at different beam intensities. Their weights control how strongly each contribution affects the correction.

All other parameters are explained in the GUI.
## Usage

1. Select a machine or simulation interface, then select the actuators and BPMs to use.
2. Choose the directories containing the measured response data. For corrector DFS/WFS, include the corresponding changed-energy or changed-intensity measurements.
3. Configure the correction weights, gain, number of iterations, matrix inversion settings and actuator limits. Adjust individual BPM weights if needed.
4. Start the correction and follow the orbit and correction-metric plots. The GUI provides controls to stop the correction and restore the initial settings.

Optional jitter subtraction uses upstream reference BPMs to estimate correlated beam motion before calculating the correction.

## Correction modes

- `BBA_GUI.py`: corrector-based orbit, DFS and WFS correction, with configurable beam energy and intensity settings.
- `BBA_QM_GUI.py`: correction using transverse quadrupole-mover positions. Its weights balance trajectory preservation, BPM centering and centering at a selected BPM; DFS and WFS are not used in this mode.

The plots show how the measured orbit and enabled correction objectives change over successive iterations. Session data and settings are saved for subsequent inspection.

## Installation and startup


```bash
git clone https://gitlab.cern.ch/flight-simulator/machine-interfaces.git
git clone https://gitlab.cern.ch/flight-simulator/response-matrix.git
git clone https://gitlab.cern.ch/flight-simulator/beam-based-alignment.git
cd beam-based-alignment
poetry update rf-track --lock
poetry install --extras clear
poetry run python BBA_GUI.py
```

Keep these repositories in the same parent directory; Poetry uses their local editable packages. The selected extra installs only that machine adapter and its dependencies. Available machine extras: `clear`, `atf2`, `facet2`, `esrf`; combine them with `--extras "clear atf2"` when needed.

Other entry files, using the same interpreter:

- `BBA_QM_GUI.py`
