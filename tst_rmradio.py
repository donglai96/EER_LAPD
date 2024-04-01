from read_lapd import read_lapd_data
# from toolbox import bdot
from bdot_calib import bint_fft
import numpy as np
from scipy import constants as const
from scipy import fft
import time
from scipy.ndimage import maximum_filter1d, uniform_filter1d
import matplotlib.pyplot as plt

datapath = '/data/BAPSF_Data/Energetic_Electron_Ring/jul2021/'
filename = datapath + \
        '04-bfield-p25-plane-He-1kG-uwave-l5ms-mirror-min-305G.hdf5'
savepath = './'
# read data of one channel in one shot and one location
data = read_lapd_data(filename, rchan=[1], rshot=[4], xrange=[20], yrange=[8])

# fft
n0 = data['data'].shape[0]
nfast = 2**(int(np.log2(n0-1))+1)
emf_fft = fft.fft(data['data'], n=nfast, axis=0)
emf_fft = np.squeeze(emf_fft)

emf_max = maximum_filter1d(np.abs(emf_fft), size=1000, mode='wrap')
emf_mean = uniform_filter1d(np.abs(emf_fft), size=1000, mode='wrap')
emf_ratio_median = np.median(emf_max / emf_mean)
emf_ratio_mean = np.mean(emf_max / emf_mean)
print('median(emf_max / emf_mean) = {}'.format(emf_ratio_median))
print('mean(emf_max / emf_mean) = {}'.format(emf_ratio_mean))

# plt.style.use('dark_background')

fig, axs = plt.subplots(2, 1, figsize=[9.6,4.8])

freq = fft.fftfreq(nfast, d=data['dt'][0])

ax = axs[0]
ax.semilogy(freq/1e9, np.abs(emf_fft), '.')
ax.set_ylabel(r'$\vert \varepsilon_{emf} \vert$')

ax = axs[1]
ax.semilogy(freq/1e9, emf_max/emf_mean, '.')
ax.axhline(y=emf_ratio_median)
ax.axhline(y=emf_ratio_mean)
ax.set_xlabel(r'$\mathrm{Frequency}\,[\mathrm{GHz}]$')
ax.set_ylabel('Max / Mean (filtered)')

plt.tight_layout()
plt.savefig(savepath+'tst_rmradio.png')
# plt.close()



def calcRegionBounds(bool_array):
    '''
    Returns the lower and upper bounds of contiguous regions.

    Parameters
    ==========
    bool_array    1-D Binary numpy array
    '''
    assert(bool_array.dtype=='bool')
    idx = np.diff(np.r_[0, bool_array, 0]).nonzero()[0]
    assert(len(idx)%2 == 0)
    return np.reshape(idx, (-1,2))

# index of short pulses
idx = (emf_max / emf_mean) > 4
# calculate bounds of short pulses
ibd = calcRegionBounds(idx)


emf_copy = emf_fft.copy()
for i in range(ibd.shape[0]):
    aux_arr = np.arange(ibd[i,0], ibd[i,1])
    # The first and last chunks are contiguous
    if i == 0 and ibd[0,0] == 0 and ibd[-1,1] == len(emf_fft):
        lwgt = (ibd[0,1] - aux_arr) \
                / (ibd[0,1] + ibd[-1,1] - ibd[-1,0] + 1)
        rwgt = (aux_arr + ibd[-1,1] - ibd[-1,0] + 1) \
                / (ibd[0,1] + ibd[-1,1] - ibd[-1,0] + 1)
        lval = emf_fft[ibd[-1,0]-1]
        rval = emf_fft[ibd[0,1]]
    elif i == ibd.shape[0]-1 and ibd[-1,1] == len(emf_fft) and ibd[0,0] == 0:
        lwgt = (ibd[-1,1] - aux_arr + ibd[0,1]) \
                / (ibd[-1,1] - ibd[-1,0] + ibd[0,1] +1)
        rwgt = (aux_arr - ibd[-1,0] + 1) \
                / (ibd[-1,1] - ibd[-1,0] + ibd[0,1] +1)
        lval = emf_fft[ibd[-1,0]-1]
        rval = emf_fft[ibd[0,1]]
    # Contiguous sections in between
    else:
        lwgt = (ibd[i,1] - aux_arr) / (ibd[i,1] - ibd[i,0] + 1)
        rwgt = (aux_arr - ibd[i,0] + 1) / (ibd[i,1] - ibd[i,0] + 1)
        lval = emf_fft[ibd[i,0]-1]
        rval = emf_fft[ibd[i,1]%len(emf_fft)]
    emf_copy[ibd[i,0]:ibd[i,1]] = lwgt*lval + rwgt*rval



fig, axs = plt.subplots(2, 1, figsize=[9.6,4.8])

freq = fft.fftfreq(nfast, d=data['dt'][0])

ax = axs[0]
ax.semilogy(freq/1e9, np.abs(emf_fft), '.')
ax.semilogy(freq/1e9, np.abs(emf_copy), '.')
ax.set_ylabel(r'$\vert \varepsilon_{emf} \vert$')

ax = axs[1]
ax.semilogy(freq/1e9, emf_max/emf_mean, '.')
ax.axhline(y=emf_ratio_median)
ax.axhline(y=emf_ratio_mean)
ax.set_xlabel(r'$\mathrm{Frequency}\,[\mathrm{GHz}]$')
ax.set_ylabel('Max / Mean (filtered)')

plt.tight_layout()
plt.savefig(savepath+'tst_rmradio2.png')
# plt.close()

# # index of short pulses
# idx = (emf_max / emf_mean) > 4
# idx1 = np.roll(idx, 1)
# # index of start and end of short pulses
# idx_xor = np.logical_xor(idx, idx1)
# # index of start of short pulses
# idx_and1 = np.logical_and(idx, idx_diff)
# idx_and2 = np.logical_and(np.roll(idx, 1), idx_diff)
# # write start and end indices into array
# idx_start = np.zeros_like(idx)
# idx_end = np.zeros_like(idx)
# iidx_start = np.where(idx_and1)[0]
# iidx_end = np.where(idx_and2)[0]
# assert iidx_start.shape[0] == iidx_end.shape[0], \
#         "start and end indices do not form pairs!"
# 
# if iidx_start[0] > iidx_end[0]:
#     offset = 1
# else:
#     offset = 0
# 
# npulse = iidx_start.shape[0]
# nt = idx.shape[0]
# for j in range(npulse):
#     istart = iidx_start[j]
#     iend = iidx_end[(j+offset)%npulse]
#     if istart < iend:
#         idx_start[istart:iend+1] = (istart - 1)%nt
#         idx_end[istart:iend+1] = (iend + 1)%nt

