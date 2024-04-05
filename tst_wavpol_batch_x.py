
import pickle
from read_lapd import read_lapd_data
from bdot_process import emf2bw_areav
import numpy as np
import matplotlib.pyplot as plt
import pyspedas
from matplotlib import ticker, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import h5py
from bdot_process import emf2bw
import os

datapath = '/data/BAPSF_Data/Energetic_Electron_Ring/jul2021/'
filename = '04-bfield-p25-plane-He-1kG-uwave-l5ms-mirror-min-305G.hdf5'
#filename = '11-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-305G.hdf5'
# filename =  '16-bfield-bmirror-scan-p25-xline-uwave-l3ms-mirror-min-380G.hdf5'
#savedir = 'test_04_x_y/'

filepath = datapath + filename
os.mkdir('test_04_x20_y8' )
savedir = ('test_04_x20_y8' + '/')

for i in range(8):
        data = read_lapd_data(filepath, rchan=[0, 1, 2], rshot=[i], xrange=[20], yrange=[8])

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
        (timeline, freqline, powspec, degpol, waveangle,
                elliptict, helict, pspec3, err_flag) = result

        # make the figure
        fig, axs = plt.subplots(5, 1, figsize=[6.4, 12.8],\
                sharex=True, constrained_layout=True)
        tfactor = 1e6
        tgrid, fgrid = np.meshgrid(timeline, freqline)
        ax = axs[0]
        ax.plot(tt*tfactor, bx, label=r'$x$')
        ax.plot(tt*tfactor, by, label=r'$y$')
        ax.plot(tt*tfactor, bz, label=r'$z$')
        ax.set_xlim([tt[0]*tfactor, tt[-1]*tfactor])
        ax.set_ylabel(r'Magnetic field [G]')
        ax.legend(ncol=1, loc='center right', bbox_to_anchor=(1.2,0.5))


        cmap = plt.get_cmap('nipy_spectral')
        logpow = np.log10(powspec)
        vmax = np.max(logpow)
        vmin = vmax - 8
        vlvs = np.linspace(vmin, vmax, 101)
        norm = colors.BoundaryNorm(vlvs, ncolors=cmap.N)
        im = axs[1].pcolormesh(tgrid*tfactor, fgrid/1e9, logpow.T, cmap=cmap, norm=norm, shading = 'nearest')
        axs[1].set_ylim([0, highcut/1e9])
        axs[1].set_ylabel(r'$f$ [GHz]')
        # colorbar
        l = ticker.AutoLocator()
        l.create_dummy_axis()
        ticks = l.tick_values(vmin, vmax)
        axins = inset_axes(axs[1], width="5%", height="100%", loc='lower left',\
                bbox_to_anchor=(1.01, 0, 1, 1), bbox_transform=axs[1].transAxes,\
                borderpad=0)
        cb = fig.colorbar(im, cax=axins, ticks=ticks, orientation='vertical')
        cb.ax.set_ylabel(r'$\log_{10} B^2$')

        cmap = plt.get_cmap('nipy_spectral')
        vmax = 1
        vmin = 0
        vlvs = np.linspace(vmin, vmax, 101)
        norm = colors.BoundaryNorm(vlvs, ncolors=cmap.N)
        im = axs[2].pcolormesh(tgrid*tfactor, fgrid/1e9, degpol.T, cmap=cmap, norm=norm,shading = 'nearest')
        axs[2].set_ylim([0, highcut/1e9])
        axs[2].set_ylabel(r'$f$ [GHz]')
        # colorbar
        l = ticker.AutoLocator()
        l.create_dummy_axis()
        ticks = l.tick_values(vmin, vmax)
        axins = inset_axes(axs[2], width="5%", height="100%", loc='lower left',\
                bbox_to_anchor=(1.01, 0, 1, 1), bbox_transform=axs[2].transAxes,\
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
        file_id = savedir + 'shot_' + str(int(i))
        plt.savefig(file_id + 'tst_wavpol.png', dpi=300)

        plt.show()