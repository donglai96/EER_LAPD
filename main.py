from EER_read_utils import *
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    # Example usage
    base_folder = '/data/BAPSF_Data/Energetic_Electron_Ring/jul2021'

    file_index = 12
    file_name = find_file_path(base_folder, file_index)
    print("The file_name is: ", file_name)
    
    print("Now reading file!\n")
    
    print("The file info: (in the Descroption)\n")
    print("!!!!!")
    
    file_info = get_lapd_fileinfo(file_name)
    
    print(file_info['Description'])
    print("!!!!!")
    print("Allert from Donglai Ma:")
    print('There are different names of the info, like Experiment description and Experiment set description... remember to check the file keys')
    
    board_number = 15
    chan_number = 5
    print("!!!!!")
    channel_data = get_channel_data(file_name, board_number, chan_number)
    # nt nx ny nshot nchannels
    print(channel_data.shape)
    fig, ax = plt.subplots(2,1)
    data_test = channel_data[:,0,0,0,0]
    ax[0].plot(data_test)
    
    
    
    dt = 2e-10
    fft_result = np.fft.fft(data_test)
    fft_freq = np.fft.fftfreq(data_test.size, d=dt)
    ax[1].semilogy(fft_freq, np.abs(fft_result))
    ax[1].set(xlabel='Frequency (Hz)', ylabel='Amplitude',
       title='FFT of the data')
    
    plt.savefig('test.png')