from EER_read_utils import *
import matplotlib.pyplot as plt
import numpy as np
from read_lapd import read_lapd_data

if __name__ == "__main__":
  
  datapath = '/data/BAPSF_Data/Energetic_Electron_Ring/jul2021/'
  filename = datapath + \
        '04-bfield-p25-plane-He-1kG-uwave-l5ms-mirror-min-305G.hdf5'
  #savepath = '/data_old/xinan/July2021/'
# read data of one channel in one shot and one location
  data = read_lapd_data(filename, rchan=[1], rshot=[4], xrange=[20], yrange=[8])
