from scipy.signal import butter, sosfiltfilt

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
