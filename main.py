from EER_read_utils import *

if __name__ == "__main__":
    # Example usage
    base_folder = '/data/BAPSF_Data/Energetic_Electron_Ring/jul2021'

    file_index = 25
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