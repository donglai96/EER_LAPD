from scipy import fft
from rmradio import rm_fft_pulse
import numpy as np
import scipy.constants as const
from bandpass import butter_bandpass_filter
from scipy.interpolate import interp1d

def emf2bw(emf, area, lowcut, highcut, fs, rm_radio=0):
    '''
    Input
    emf        1D array constaining emf data
    area       area of the bdot probe loop in cgs units
    lowcut     low-cutoff frequency
    highcut    high-cutoff frequency
    fs         sampling frequency

    Output:
    Bw         magnetic field in Gauss
    '''
    assert len(emf.shape) == 1, 'Input emf to emf2bw must be 1D array.'
    
    # speed of light in cgs units
    c_cgs = const.c * 1e2

    # fft
    n0 = emf.shape[0]
    nfast = 2**(int(np.log2(n0-1))+1)
    emf_fft = fft.fft(emf, n=nfast)

    # remove radio waves from emf fft
    if rm_radio != 0:
        emf_fft[:] = rm_fft_pulse(emf_fft)

    # convert emf to magnetic field in frequency space
    freq = fft.fftfreq(nfast, d=1/fs)
    Bw_fft = emf_fft * (1e-2/3) / (-2.0j * const.pi * freq) / area * (c_cgs)
    # handle singularity at f=0 by linear interpolation
    Bw_fft[0] = 0.5 * (Bw_fft[1] + Bw_fft[-1])

    # ifft
    Bw = np.real(fft.ifft(Bw_fft, n=nfast))

    # bandpass filter
    Bw[:n0] = butter_bandpass_filter(Bw[:n0], lowcut, highcut, fs)

    return Bw[:n0]



def emf2bw_areav(emf, areav, lowcut, highcut, fs, rm_radio=0):
    '''
    Input
    emf        1D array constaining emf data
    areav      areav of the bdot probe loop in cgs units;
               areav is a dict containing area as a function of freq.
    lowcut     low-cutoff frequency
    highcut    high-cutoff frequency
    fs         sampling frequency

    Output:
    Bw         magnetic field in Gauss
    '''
    assert len(emf.shape) == 1, 'Input emf to emf2bw must be 1D array.'
    
    # speed of light in cgs units
    c_cgs = const.c * 1e2

    # fft
    n0 = emf.shape[0]
    nfast = 2**(int(np.log2(n0-1))+1)
    emf_fft = fft.fft(emf, n=nfast)

    # remove radio waves from emf fft
    if rm_radio != 0:
        emf_fft[:] = rm_fft_pulse(emf_fft)

    # form fft freq
    freq = fft.fftfreq(nfast, d=1/fs)

    # interpolate area defined on its own freq to fft freq
    areafunc = interp1d(areav['freq'], areav['area'], kind='linear', \
            fill_value=(areav['area'][0], areav['area'][-1]), \
            bounds_error=False)
    area = areafunc(freq)

    # convert emf to magnetic field in frequency space
    Bw_fft = emf_fft * (1e-2/3) / (-2.0j * const.pi * freq) / area * (c_cgs)
    # handle singularity at f=0 by linear interpolation
    Bw_fft[0] = 0.5 * (Bw_fft[1] + Bw_fft[-1])

    # ifft
    Bw = np.real(fft.ifft(Bw_fft, n=nfast))

    # bandpass filter
    Bw[:n0] = butter_bandpass_filter(Bw[:n0], lowcut, highcut, fs)

    return Bw[:n0]



