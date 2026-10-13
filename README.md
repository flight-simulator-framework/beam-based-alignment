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

## Development

Development, issues and pull requests are hosted on [GitHub](https://github.com/flight-simulator-framework/beam-based-alignment). The [CERN GitLab repository](https://gitlab.cern.ch/flight-simulator/beam-based-alignment) receives updates from GitHub through a pull mirror.

## Installation and startup

On a control-room computer, install the applications with the deployment script of your machine: [CLEAR](https://github.com/flight-simulator-framework/machine-interfaces/blob/master/backends/clear/clear_deploy.md), [ATF2](https://github.com/flight-simulator-framework/machine-interfaces/blob/master/backends/atf2/atf2_deploy.md), [ESRF](https://github.com/flight-simulator-framework/machine-interfaces/blob/master/backends/esrf/esrf_deploy.md) or [FACET2](https://github.com/flight-simulator-framework/machine-interfaces/blob/master/backends/facet2/facet2_deploy.md). It creates a shared virtual environment and one launcher per GUI, e.g. `./BBA_GUI.sh`.

For a local installation, use Python 3.11 or newer and keep the repositories in one directory. The commands below clone from GitHub; to clone from CERN GitLab instead, replace `https://github.com/flight-simulator-framework/` with `https://gitlab.cern.ch/flight-simulator/`:

```bash
git clone https://github.com/flight-simulator-framework/machine-interfaces.git
git clone https://github.com/flight-simulator-framework/response-matrix.git
git clone https://github.com/flight-simulator-framework/beam-based-alignment.git
python3 -m venv venv
source venv/bin/activate
pip install -e machine-interfaces -e machine-interfaces/backends/esrf -e response-matrix -e beam-based-alignment
cd beam-based-alignment
python BBA_GUI.py
```

Replace `esrf` with `atf2` or `facet2` for another machine. For ATF2, also build the magnet current–strength conversion library from the directory containing the repositories; this requires `gcc`, `gfortran` and `make`:

```bash
make -C machine-interfaces/backends/atf2/machine_interfaces_atf2/libmagnet libmagnet.so CFLAG=-fPIC
```

The CLEAR control-system packages are available only from the CERN Acc-Py index; elsewhere, install only the CLEAR simulation:

```bash
pip install rf-track cpymad
pip install --no-deps -e machine-interfaces/backends/clear
```

Editable installs pick up `git pull` changes without reinstalling.

Other entry files, using the same environment:

- `BBA_QM_GUI.py`
