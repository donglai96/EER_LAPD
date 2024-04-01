from scipy import fft
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

def bint_fft(emf, dt=1.0, axis=0, workers=32):
    if axis != 0:
        print('axis /= 0 in bint_fft is not supported!')
        sys.exit()
    # dimension of emf
    ndim = emf.shape
    # next fast length
    nfast = 2**(int(np.log2(ndim[axis]-1)) + 1)
    # fft
    emf_fft = fft.ifft(emf, n=nfast, axis=axis, workers=workers)
    # fft frequency
    f = fft.fftfreq(nfast, d=dt)
    # emf / (-j * 2 * pi * f)
    jw = -2.0j * np.pi * f
    # this step will have an error 'divide by zero'
    emf_fft = (emf_fft.T / jw).T
    # handling singularity at f=0
    emf_fft[0, ...] = emf_fft[1, ...]
    # ifft
    return np.real(fft.fft(emf_fft, n=nfast, axis=axis, \
            workers=workers))


def area_calib(data, g=10, r=5.5e-2, debug=0):
    # data is a dict that contains frequency and logmag
    # freq = data['freq']        # [Hz]
    # logmag = data['logmag']    # [dB]

    def fline(x, AA, BB):    # straight line
        return AA * x + BB
    
    # popt[0] =  gradient = mag / omega
    popt, _ = curve_fit(fline, 2.0*const.pi*data['freq'], \
            10.0**(data['logmag']/20.0))

    area = popt[0] * r / (32.0 * (4.0/5.0)**1.5 * g * const.mu_0)

    if debug != 0:
        fig, ax = plt.subplots(1, 1)
        ax.plot(data['freq'], 10.0**(data['logmag']/20.0), '.')
        ax.plot(data['freq'], fline(2.0*const.pi*data['freq'], *popt))
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('magnitude A/R')
        ax.legend(['original data', 'best fit line'])
        plt.tight_layout()

    return area    # in m^2
