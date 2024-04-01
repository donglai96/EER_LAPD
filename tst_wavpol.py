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
filename = '11-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-305G.hdf5'
# filename =  '12-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-320G.hdf5'
# filename = '13-bfield-bmirror-scan-p25-xline-uwave-l3ms-mirror-min-340G.hdf5'
# filename = '14-bfield-bmirror-scan-p25-xline-uwave-l3ms-mirror-min-320G.hdf5'
# filename = '15-bfield-bmirror-scan-p25-xline-uwave-l3ms-mirror-min-360G.hdf5'
# filename = '16-bfield-bmirror-scan-p25-xline-uwave-l3ms-mirror-min-380G.hdf5'
# filename = '17-bfield-p25-xline-uwave-l3ms-long-mirror-min-305G.hdf5'
# filename = '18-bfield-p25-xline-uwave-l3ms-asym-mirrorS-min-305G.hdf5'
# filename = '19-bfield-p25-xline-uwave-l3ms-asym-mirrorN-min-305G.hdf5'
# filename = '20-bfield-tscan-p25-xline-uwave-l3ms-daq-9ms-mirror-min-305G.hdf5'
# filename = '21-bfield-tscan-p25-xline-uwave-13ms-daq-7ms-mirror-min-305G.hdf5'
# filename = '22-bfield-tscan-p25-xline-uwave-13ms-daq-5ms-mirror-min-305G.hdf5'
# filename = '23-bfield-tscan-p25-xline-uwave-13ms-daq-3ms-mirror-min-305G.hdf5'
# filename = '27-bfield-p25-xline-uwave-13ms-at-10ms-mirror-min-305G.hdf5'
# filename = '29-bfield-p25-xline-uwave-13ms-at-0ms-mirror-min-305G.hdf5'
runid = filename[0:2]
filepath = datapath + filename
data = read_lapd_data(filepath, rchan=[0, 1, 2], rshot=[0], xrange=[0], yrange=[0])

# Not sure how to get this one, remember to ask Xin!

area = [1.889e-1, 1.843e-1, 1.501e-1]

fs = 1.0 / data['dt'][0]
lowcut = 1.5e8
highcut = 1e9
bx = emf2bw(np.squeeze(data['data'])[:, 0], area[0], \
        lowcut, highcut, fs, rm_radio=1)
by = emf2bw(np.squeeze(data['data'])[:, 1], area[1], \
        lowcut, highcut, fs, rm_radio=1)
bz = emf2bw(np.squeeze(data['data'])[:, 2], area[2], \
        lowcut, highcut, fs, rm_radio=1)
# bx = emf2bw_areav(np.squeeze(data['data'])[:, 0], areav[0], \
#         lowcut, highcut, fs, rm_radio=1)
# by = emf2bw_areav(np.squeeze(data['data'])[:, 1], areav[1], \
#         lowcut, highcut, fs, rm_radio=1)
# bz = emf2bw_areav(np.squeeze(data['data'])[:, 2], areav[2], \
#         lowcut, highcut, fs, rm_radio=1)

# truncate time series at the beginning and end sections
ntrunc = 100
bx = bx[ntrunc:-ntrunc]
by = by[ntrunc:-ntrunc]
bz = bz[ntrunc:-ntrunc]
tt = data['time'][ntrunc:-ntrunc]

# Number of points in FFT
nopfft = 1024
# The amount of overlap between successive FFT intervals
steplength = nopfft / 2
# No. of bins in frequency domain for averaging [1, 7]
bin_freq = 7

result = pyspedas.analysis.twavpol.wavpol(tt, bx, by, bz, nopfft=nopfft, steplength=steplength, bin_freq=bin_freq)
fig, axs = plt.subplots(5, 1, figsize=[6.4, 12.8],\
        sharex=True, constrained_layout=True)




(timeline, freqline, powspec, degpol, waveangle,
              elliptict, helict, pspec3, err_flag) = result
# with open('my_test_tuple.pkl', 'wb') as file:
#     pickle.dump(result, file)
tgrid, fgrid = np.meshgrid(timeline, freqline)
tfactor = 1e6

fig.subplots_adjust(right=0.82)

ax = axs[0]
ax.plot(tt*tfactor, bx, label=r'$x$')
ax.plot(tt*tfactor, by, label=r'$y$')
ax.plot(tt*tfactor, bz, label=r'$z$')
ax.set_xlim([tt[0]*tfactor, tt[-1]*tfactor])
ax.set_ylabel(r'Magnetic field [G]')
ax.legend(ncol=1, loc='center right', bbox_to_anchor=(1.2,0.5))

ax = axs[1]
cmap = plt.get_cmap('nipy_spectral')
logpow = np.log10(powspec)
vmax = np.max(logpow)
vmin = vmax - 8
vlvs = np.linspace(vmin, vmax, 101)
norm = colors.BoundaryNorm(vlvs, ncolors=cmap.N)
im = ax.pcolormesh(tgrid*tfactor, fgrid/1e9, logpow.T, cmap=cmap, norm=norm, shading = 'nearest')
ax.set_ylim([0, highcut/1e9])
ax.set_ylabel(r'$f$ [GHz]')
# colorbar
l = ticker.AutoLocator()
l.create_dummy_axis()
ticks = l.tick_values(vmin, vmax)
axins = inset_axes(ax, width="5%", height="100%", loc='lower left',\
        bbox_to_anchor=(1.01, 0, 1, 1), bbox_transform=ax.transAxes,\
        borderpad=0)
cb = fig.colorbar(im, cax=axins, ticks=ticks, orientation='vertical')
cb.ax.set_ylabel(r'$\log_{10} B^2$')

ax = axs[2]
cmap = plt.get_cmap('nipy_spectral')
vmax = 1
vmin = 0
vlvs = np.linspace(vmin, vmax, 101)
norm = colors.BoundaryNorm(vlvs, ncolors=cmap.N)
im = ax.pcolormesh(tgrid*tfactor, fgrid/1e9, degpol.T, cmap=cmap, norm=norm,shading = 'nearest')
ax.set_ylim([0, highcut/1e9])
ax.set_ylabel(r'$f$ [GHz]')
# colorbar
l = ticker.AutoLocator()
l.create_dummy_axis()
ticks = l.tick_values(vmin, vmax)
axins = inset_axes(ax, width="5%", height="100%", loc='lower left',\
        bbox_to_anchor=(1.01, 0, 1, 1), bbox_transform=ax.transAxes,\
        borderpad=0)
cb = fig.colorbar(im, cax=axins, ticks=ticks, orientation='vertical')
cb.ax.set_ylabel('Polarization')
cb.formatter.set_powerlimits((0, 0))
cb.update_ticks()

ax = axs[3]
cmap = plt.get_cmap('nipy_spectral')
vmax = 90
vmin = 0
vlvs = np.linspace(vmin, vmax, 101)
norm = colors.BoundaryNorm(vlvs, ncolors=cmap.N)
im = ax.pcolormesh(tgrid*tfactor, fgrid/1e9, np.rad2deg(waveangle).T, cmap=cmap, norm=norm,shading = 'nearest')
ax.set_ylabel(r'$f$ [GHz]')
ax.set_ylim([0, highcut/1e9])
# colorbar
l = ticker.AutoLocator()
l.create_dummy_axis()
ticks = l.tick_values(vmin, vmax)
axins = inset_axes(ax, width="5%", height="100%", loc='lower left',\
        bbox_to_anchor=(1.01, 0, 1, 1), bbox_transform=ax.transAxes,\
        borderpad=0)
cb = fig.colorbar(im, cax=axins, ticks=ticks, orientation='vertical')
cb.ax.set_ylabel(r'$\theta_{kB}$ [Deg]')

ax = axs[4]
cmap = plt.get_cmap('bwr')
vmax = 1
vmin = -1
vlvs = np.linspace(vmin, vmax, 101)
norm = colors.BoundaryNorm(vlvs, ncolors=cmap.N)
im = ax.pcolormesh(tgrid*tfactor, fgrid/1e9, elliptict.T, cmap=cmap, norm=norm,shading = 'nearest')
ax.set_ylim([0, highcut/1e9])
ax.set_xlabel(r'Time [$\mu$s]')
ax.set_ylabel(r'$f$ [GHz]')
# colorbar
l = ticker.AutoLocator()
l.create_dummy_axis()
ticks = l.tick_values(vmin, vmax)
axins = inset_axes(ax, width="5%", height="100%", loc='lower left',\
        bbox_to_anchor=(1.01, 0, 1, 1), bbox_transform=ax.transAxes,\
        borderpad=0)
cb = fig.colorbar(im, cax=axins, ticks=ticks, orientation='vertical')
cb.ax.set_ylabel('Ellipticity')


# plt.tight_layout()
plt.savefig('tst_wavpol_11.png', dpi=300)
plt.show()

