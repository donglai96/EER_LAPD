import numpy as np
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

def rm_fft_pulse(sig_fft, size=1000, threshold=4):
    '''
    Removes pulses in fft spectrum.

    Parameters
    ==========
    sig_fft      1-D complex array containing fft spectrum
    size         scalar defining the window length of max and mean filters;
                 size should be slightly larger than the pulse width;
    threshold    scalar defining the threshold of max/mean above which
                 the sig_fft array will be interpolated.
    '''
    # max filter
    sig_max = maximum_filter1d(np.abs(sig_fft), size=size, mode='wrap')
    # mean filter
    sig_mean = uniform_filter1d(np.abs(sig_fft), size=size, mode='wrap')
    # index of short pulses
    idx = (sig_max / sig_mean) > threshold
    # calculate bounds of short pulses
    ibd = calcRegionBounds(idx)

    sig_copy = sig_fft.copy()
    # interpolation
    for i in range(ibd.shape[0]):
        aux_arr = np.arange(ibd[i,0], ibd[i,1])
        # The first and last chunks are contiguous by wrapping
        if i == 0 and ibd[0,0] == 0 and ibd[-1,1] == len(sig_fft):
            lwgt = (ibd[0,1] - aux_arr) \
                    / (ibd[0,1] + ibd[-1,1] - ibd[-1,0] + 1)
            rwgt = (aux_arr + ibd[-1,1] - ibd[-1,0] + 1) \
                    / (ibd[0,1] + ibd[-1,1] - ibd[-1,0] + 1)
            lval = sig_fft[ibd[-1,0]-1]
            rval = sig_fft[ibd[0,1]]
        elif i == ibd.shape[0]-1 and ibd[-1,1] == len(sig_fft) and ibd[0,0] == 0:
            lwgt = (ibd[-1,1] - aux_arr + ibd[0,1]) \
                    / (ibd[-1,1] - ibd[-1,0] + ibd[0,1] +1)
            rwgt = (aux_arr - ibd[-1,0] + 1) \
                    / (ibd[-1,1] - ibd[-1,0] + ibd[0,1] +1)
            lval = sig_fft[ibd[-1,0]-1]
            rval = sig_fft[ibd[0,1]]
        # Contiguous sections in between
        else:
            lwgt = (ibd[i,1] - aux_arr) / (ibd[i,1] - ibd[i,0] + 1)
            rwgt = (aux_arr - ibd[i,0] + 1) / (ibd[i,1] - ibd[i,0] + 1)
            lval = sig_fft[ibd[i,0]-1]
            rval = sig_fft[ibd[i,1]%len(sig_fft)]
        sig_copy[ibd[i,0]:ibd[i,1]] = lwgt*lval + rwgt*rval

    return sig_copy
