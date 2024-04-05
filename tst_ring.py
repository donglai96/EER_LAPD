import numpy as np
import matplotlib.pyplot as plt
from read_lapd import read_lapd_data
# from toolbox import bdot
from bdot_calib import bint_fft
import numpy as np
from scipy import constants as const
from scipy import fft
import time
from scipy.ndimage import maximum_filter1d, uniform_filter1d


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


datapath = '/data/BAPSF_Data/Energetic_Electron_Ring/jul2021/'
filename = datapath + \
        '11-bfield-bmirror-scan-p25-xline-uwave-l5ms-mirror-min-305G.hdf5'

#fig, axs = plt.subplots(2, 1, figsize=[9.6,4.8])

# filename = datapath + \
#         '24-bfield-p25-xline-uwave-off-mirror-min-305G.hdf5'
#24-bfield-p25-xline-uwave-off-mirror-min-305G.hdf5
savepath = './'
# read data of one channel in one shot and one location

for xx in [0,15,30,45,60]:
    emf_fft_total = np.zeros((1048077,9))
    for s in range(9):
        data = read_lapd_data(filename, rchan=[1], rshot=[s], xrange=[xx]) 
    
    
        # fft
        n0 = data['data'].shape[0]
        nfast = 2**(int(np.log2(n0-1))+1)
        freq = fft.fftfreq(nfast, d=data['dt'][0])
        emf_fft = fft.fft(data['data'], n=nfast, axis=0)
        emf_fft = np.squeeze(emf_fft)
        print(emf_fft.shape,'test')
        emf_max = maximum_filter1d(np.abs(emf_fft), size=1000, mode='wrap')
        emf_mean = uniform_filter1d(np.abs(emf_fft), size=1000, mode='wrap')
        emf_ratio_median = np.median(emf_max / emf_mean)
        emf_ratio_mean = np.mean(emf_max / emf_mean)







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

        window_size = 500  # Adjust based on your smoothing requirements

        # Define a simple moving average function
        def moving_average(data, size):
            return np.convolve(data, np.ones(size), 'valid') / size

        # Apply the moving average filter
        smoothed_emf_fft = moving_average(np.abs(emf_fft), window_size)
        print(smoothed_emf_fft.shape, 'test!!')
        # Since the moving average reduces the array size, adjust the frequency array
        # We take the center point of each window for the frequency
        adjusted_freq = freq[window_size-1:]  # Adjusting freq array size to match smoothed signal
        
        emf_fft_total[:,s] = smoothed_emf_fft

    # For plotting, you might still want to decimate or just plot the smoothed data
    # Here, we just plot the smoothed data
    
    emf_fft_average = np.average(emf_fft_total,axis = 1)
    plt.semilogy(adjusted_freq/1e9/0.8538, emf_fft_average, '.',label = 'x = '+str(int(xx)))

    # Set the x-axis limit
    plt.xlim(0, 1)

    # Add labels and title for clarity (optional)
    plt.xlabel('f/fce')
            #plt.title('Microwave on')

plt.legend()
plt.savefig('tst_ring_11_x.png',format = 'png')
plt.show()