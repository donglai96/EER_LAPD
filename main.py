from EER_read_utils import *
import matplotlib.pyplot as plt
import numpy as np
from read_lapd import read_lapd_data
# from toolbox import bdot
# from bdot_calib import bint_fft
import numpy as np
from scipy import constants as const
from scipy import fft
import time
import matplotlib.pyplot as plt
from rmradio import rm_fft_pulse
from scipy.signal import butter, sosfiltfilt
if __name__ == "__main__":
  
  datapath = '/data/BAPSF_Data/Energetic_Electron_Ring/jul2021/'
  filename = datapath + \
        '04-bfield-p25-plane-He-1kG-uwave-l5ms-mirror-min-305G.hdf5'
  #savepath = '/data_old/xinan/July2021/'
# read data of one channel in one shot and one location
  data = read_lapd_data(filename, rchan=[1], rshot=[4], xrange=[20], yrange=[8])
  n0 = data['data'].shape[0]
  nfast = 2**(int(np.log2(n0-1))+1)
  emf_fft = fft.fft(data['data'], n=nfast, axis=0)
  emf_fft = np.squeeze(emf_fft)

  emf_fft2 = rm_fft_pulse(emf_fft, size=500)

  # ifft
  emf2 = fft.ifft(emf_fft2, n=nfast)

  fs = 1.0 / data['dt'][0]
  lowcut = 1e7
  highcut = 2e9


  def butter_bandpass_filter(data, lowcut, highcut, fs, order=6, axis=0):
      # Nyquist frequency
      nyq = 0.5 * fs
      # low and high cutoff frequencies
      low = lowcut / nyq
      high = highcut / nyq
      # Second-order sections representation of the IIR filter.
      sos = butter(order, [low, high], output='sos', btype='bandpass')
      # A forward-backward digital filter using cascaded second-order sections.
      # forward-backward filter has zero phase shift.
      y = sosfiltfilt(sos, data, axis=axis)
      return y

  emf2_filt = butter_bandpass_filter(np.real(emf2[:n0]), lowcut, highcut, fs)

  # fft
  emf2_filt_fft = fft.fft(emf2_filt, n=nfast)
  freq = fft.fftfreq(nfast, d=data['dt'][0])

  fig, axs = plt.subplots(2, 1, figsize=[9.6, 4.8])

  ax = axs[0]
  tarr = np.arange(n0) * data['dt'][0]
  ax.plot(tarr*1e6, np.real(emf2[:n0]))
  ax.plot(tarr*1e6, emf2_filt)

  ax.set_xlabel(r'$\mathrm{Time}\,[\mu s]$')
  ax.set_ylabel(r'$\varepsilon_{emf}$')

  ax = axs[1]
  ax.semilogy(freq/1e9, np.abs(emf_fft2), '.')
  ax.semilogy(freq/1e9, np.abs(emf2_filt_fft), '.')

  ax.set_xlabel('Frequency [GHz]')
  ax.set_ylabel(r'$\vert \varepsilon_{emf} \vert$')

  plt.tight_layout()
  plt.savefig('tst_bandpass.png')