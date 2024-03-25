import numpy as np
from scipy import fft
from EER_read_utils import *
if __name__ == "__main__":
    # Example usage
    base_folder = '/data/BAPSF_Data/Energetic_Electron_Ring/jul2021'

    file_index = 4
    file_name = find_file_path(base_folder, file_index)
    print(file_name)
    file_info = get_lapd_fileinfo(file_name)
    print(file_info['Description'])
    
    board_number = 13
    chan_number = 5
    channel_data = get_channel_data(file_name, board_number, chan_number)
    print(channel_data.shape)
    fig, axs = plt.subplots(10,1,figsize = (10,20),sharex = True)
    
    for i, ax in enumerate(axs):
       
       
       
      data_test = channel_data[:,i*3,0,4,0]
      #ax[i].plot(data_test)
    
    
    
    
    
      dt = 2e-10
      fft_result = np.fft.fft(data_test)
      fft_freq = np.fft.fftfreq(data_test.size, d=dt)
      ax.semilogy(fft_freq, np.abs(fft_result))
      # ax.set(xlabel='Frequency (Hz)', ylabel='Amplitude',
      #    title='FFT of the data')
      
    plt.savefig('testnew.png')
