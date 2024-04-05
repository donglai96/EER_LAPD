from read_lapd import read_lapd_data
from bdot_process import emf2bw_areav
import numpy as np
import matplotlib.pyplot as plt
import pyspedas
from matplotlib import ticker, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import h5py
from bdot_process import emf2bw
import pickle 

datapath = '/data/BAPSF_Data/Energetic_Electron_Ring/jul2021/'
# filename = '04-bfield-p25-plane-He-1kG-uwave-l5ms-mirror-min-305G.hdf5'
# filename = '05-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-200G.hdf5'
# filename = '06-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-220G.hdf5'
# filename = '07-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-238G.hdf5'
# filename = '09-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-265G.hdf5'
# filename = '10-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-285G.hdf5'
# filename = '11-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-305G.hdf5'
#filename =  '12-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-320G.hdf5'
# filename = '13-bfield-bmirror-scan-p25-xline-uwave-l3ms-mirror-min-340G.hdf5'
# filename = '14-bfield-bmirror-scan-p25-xline-uwave-l3ms-mirror-min-320G.hdf5'
# filename = '15-bfield-bmirror-scan-p25-xline-uwave-l3ms-mirror-min-360G.hdf5'
# filename = '16-bfield-bmirror-scan-p25-xline-uwave-l3ms-mirror-min-380G.hdf5'
# filename = '17-bfield-p25-xline-uwave-l3ms-long-mirror-min-305G.hdf5'
# filename = '18-bfield-p25-xline-uwave-l3ms-asym-mirrorS-min-305G.hdf5'
# filename = '19-bfield-p25-xline-uwave-l3ms-asym-mirrorN-min-305G.hdf5'
filename = '20-bfield-tscan-p25-xline-uwave-l3ms-daq-9ms-mirror-min-305G.hdf5'
# filename = '21-bfield-tscan-p25-xline-uwave-13ms-daq-7ms-mirror-min-305G.hdf5'
# filename = '22-bfield-tscan-p25-xline-uwave-13ms-daq-5ms-mirror-min-305G.hdf5'
# filename = '23-bfield-tscan-p25-xline-uwave-13ms-daq-3ms-mirror-min-305G.hdf5'
# filename = '27-bfield-p25-xline-uwave-13ms-at-10ms-mirror-min-305G.hdf5'
# filename = '29-bfield-p25-xline-uwave-13ms-at-0ms-mirror-min-305G.hdf5'
runid = filename[0:2]
filepath = datapath + filename
data = read_lapd_data(filepath, rchan=[1], rshot=[0])

print(data['x'])

