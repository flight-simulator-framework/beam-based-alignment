import sys
sys.path.append('/userhome/alatina/flight-simulator')

# from InterfaceATF2_Linac import InterfaceATF2_Linac
from Interfaces.ATF2.InterfaceATF2_Ext_RFTrack import InterfaceATF2_Ext_RFTrack
from Response import Response
from State import State

import matplotlib.pyplot as plt
import numpy as np

R0 = Response('response0.pkl')
R1 = Response('response2.pkl')

# The list of correctors to use
hcorrs = R0.hcorrs[1:10]
vcorrs = R0.vcorrs[1:10]

bpms = R0.bpms[1:20]

# Start correction
R0xx, R0yx = R0.submatrix_Rx(bpms, hcorrs)
R0xy, R0yy = R0.submatrix_Ry(bpms, vcorrs)

R1xx, R1yx = R1.submatrix_Rx(bpms, hcorrs)
R1xy, R1yy = R1.submatrix_Ry(bpms, vcorrs)

B0x, B0y = R0.submatrix_B(bpms)

# DFS parameters
gain = 0.2
wgt_orb = 1
wgt_dfsx = 10
wgt_dfsy = 10
rcond = 0.01

# Correction!
# I = InterfaceATF2_Linac(nsamples=3)
I = InterfaceATF2_Ext_RFTrack(jitter=0.0, bpm_resolution=0.0, nsamples=1)
S = State (interface=I)

norm_Orbit_x = []
norm_Orbit_y = []
norm_Disp_x = []
norm_Disp_y = []

# Turn on interactive plotting mode
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle('Convergence')

for iteration in range(15):

    # Nominal orbit
    print('Measuring trajectory...')
    S.pull(I)
    O0 = S.get_orbit(bpms)

    # Dispersive orbit
    print('Measuring dispersion...')
    I.change_energy()
    S.pull(I)
    I.reset_energy()
    O1 = S.get_orbit(bpms)

    # Python's transpose.......
    O0x = O0['x'].reshape(-1,1)
    O0y = O0['y'].reshape(-1,1)
    O1x = O1['x'].reshape(-1,1)
    O1y = O1['y'].reshape(-1,1)

    if iteration==1:
        B0x = O0x
        B0y = O0y

    # DFS system of equations
    B = np.vstack((wgt_orb * (O0x - B0x),
                   wgt_orb * (O0y - B0y),
                   wgt_dfsx * (O1x - O0x),
                   wgt_dfsy * (O1y - O0y)))

    Rx = wgt_orb * np.hstack((R0xx, R0xy))
    Ry = wgt_orb * np.hstack((R0yx, R0yy))
    Dx = wgt_dfsx * np.hstack((R1xx - R0xx, R1xy - R0xy))
    Dy = wgt_dfsy * np.hstack((R1yx - R0yx, R1yy - R0yy))

    R = np.vstack((Rx,Ry,Dx,Dy))
    corr = -gain * (np.linalg.pinv(R, rcond=rcond) @ B)

    # Apply correction
    I.vary_correctors(np.hstack((hcorrs,vcorrs)), corr)

    # Plots
    norm_Orbit_x = np.hstack((norm_Orbit_x, np.linalg.norm(O0x - B0x)))
    norm_Orbit_y = np.hstack((norm_Orbit_y, np.linalg.norm(O0y - B0y)))
    norm_Disp_x = np.hstack((norm_Disp_x, np.linalg.norm(O1x - O0x)))
    norm_Disp_y = np.hstack((norm_Disp_y, np.linalg.norm(O1y - O0y)))

    # Clear previous plots
    ax1.clear()
    ax2.clear()

    # Plot the updated data
    ax1.plot(range(iteration+1), norm_Orbit_x, label='X axis')
    ax1.plot(range(iteration+1), norm_Orbit_y, label='Y axis')
    ax1.set_title('Trajectory')
    ax1.set_xlabel ('Iteration [#]')
    ax1.set_ylabel ('Orbit [mm]')
    ax1.legend (loc='upper left')

    ax2.plot(range(iteration+1), norm_Disp_x, label='X axis')
    ax2.plot(range(iteration+1), norm_Disp_y, label='Y axis')
    ax2.set_title('Dispersion')
    ax2.set_xlabel ('Iteration [#]')
    ax2.set_ylabel ('Dispersion [mm]')
    ax2.legend (loc='upper left')

    # Redraw the plot
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)

print('Done!')

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot

