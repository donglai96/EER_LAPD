"""
Helper function for reading the file
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import hdf_helpers as hdf
from itertools import product
from tqdm import tqdm





def check_devices(fname):
    # Check devices used to write data file (auxillary function) -------------
    # the indexing order is important
    devices = ['6K Compumotor', 'SIS crate', 'SIS 3302', 'SIS 3305',
               'SIS 3301', 'n5700', 'NI_XZ', 'NI_XYZ', 'Waveform']

    with h5py.File(fname, 'r') as ff:

    # Check if device exists from the name
        active_list = list(ff['Raw data + config'].keys())
        name_check = [(item in active_list) for item in devices]
    return name_check



def find_file_path(base_folder, index):
    """

    """
    # Format the index to a two-digit string
    index_str = f"{index:02}"
    
    # List all files in the base_folder
    files = os.listdir(base_folder)
    
    # Find the first file that starts with the specified index
    for file in files:
        
        if file.startswith(index_str):
            return os.path.join(base_folder, file)
    
    # If no file is found, return None
    return None

def get_lapd_fileinfo(file_name):
    """
    This Function load the available info
    1. how many slots
    2. for each slot, show the available channel
    """
        
    data_dict = hdf.get_group_attrs(file_name, 'Raw data + config')
    #print(data_dict)
    return data_dict

def get_channel_data(file_name, board_number, chan_number):
    """
    This function load the data for the given board number and chan number
    
    notice that the board_number is 13, 15 for sis3305. chan number is 1 for FPGA1 ch1 and chan number is 5 for FPGA2 ch1
    """
    if board_number not in [13, 15]:
        raise ValueError("Currently only support these slots, you might looking at 3302 board. In this experiment,\
                         for 3305 board we have two slots: 13 and 15 and each one has 2 channels")
        
    else:
        board_index = (board_number -13)//2
        print(f"Start reading the board: the board number is {board_number}")
        
        CH_number = chan_number%4
        FPGA_number = chan_number//4 + 1
        
        # get the config, check whether the given channel is available
        sis_group_name = 'Raw data + config/SIS crate'
        group_names = hdf.get_subgroup_names(file_name,sis_group_name)
        print(group_names)
        if len(group_names) == 1:
            shot_name = sis_group_name + '/' + group_names[0]
            sis_group_subname = shot_name + '/' + f'SIS crate 3305 configurations[{board_index}]'
        else:
            raise ValueError('It seems not only one group! Check it')
        
        attr_dict = hdf.get_group_attrs(file_name,sis_group_subname)
        print('The data we are looking at is:')
        attr_name = f'FPGA {FPGA_number} Data type {CH_number}'
        print(attr_dict[attr_name])
        print('returning data:')
        
        shot_data = read_sisboard_shot(file_name, shot_name=shot_name,slot_number=board_number,
                                       FPGA_number=FPGA_number, CH_number= CH_number)
        
        # an progress bar
    return None


def read_sisboard_shot(file_path, shot_name, slot_number, FPGA_number, CH_number = 1, sis_id = 3305):
    # Actual shot reading code is here ---------------------------------------
    #config_subgroup_list = get_subgroup_names(fname, config_name)

    if sis_id == 3302:
        print('Warning, not support 3302 yet...')

        dataset_config_name = shot_name  + \
            ' [Slot {0}: SIS 3302 ch {1}]'.format(slot_number, CH_number) # not support 3302 yet

    elif sis_id == 3305:
        
        fpga_id, fpga_ch = FPGA_number, CH_number  # helper function
        
        dataset_config_name = shot_name + \
            ' [Slot {0}: SIS 3305 FPGA {1} ch {2}]'.\
            format(slot_number, fpga_id, fpga_ch)
            
            
            
    # determine the index
    # read motion
    print('reading motion...')

    nx, ny, nz, nshots = read_motion(file_path)
    
    nchan = 1
    #create dataset
    
    # read file data
    with h5py.File(file_path, 'r') as file:
        ll = file[dataset_config_name]  # data contained as (nshots, dt)
        print(ll.shape)
        nt = ll.shape[1]  # extract size of time dimension
        dataset = np.zeros([nt, nx, ny, nshots, nchan])
        total_iterations = nx * ny * nshots
        for iix, iiy, iishot in tqdm(product(range(nx), range(ny), range(nshots)), total=total_iterations, desc="Processing"):
            #print(iix, iiy, iishot)
            index = get_shot_index(nshots, nx, ishot=iishot, ix=iix, iy=iiy)
            temp = np.zeros([1, nt])
            ll.read_direct(temp, np.s_[index, :])  # Assuming ll is your dataset or file handle with read_direct method
            temp_convert = [-tt * (2 / (2**10 - 1)) for tt in temp]
            dataset[:, iix, iiy, iishot, 0] = np.array(temp_convert)
    #print(dataset[:10,0,0,0,0])        
    return dataset
    
    
    
    
    return None
    # temp = np.zeros([1, dt])
    # ll.read_direct(temp, np.s_[index, :])  # read slice of data

    # # convert digitizer indices to voltage values
    # if sisid == 3302:
    #     return [ii*7.7241166e-5-2.531 for ii in temp]
    # # 3305: 10 bits (0 to 1023) and 2 Volt range
    # elif sisid == 3305:
    #     return [-ii*(2 / (2**10 - 1)) for ii in temp]
    # else:
    #     return None

def read_motion(file_path):
    name_check = check_devices(file_path)
    if name_check[0]:  # 6K Compumotor
         motion_attr_dict,motion_data_dict = lapd_6k_config(file_path, motionid=0)
    else:
        raise ValueError('No 6K Compumotor, other device need to be added in the future')
    nx = motion_attr_dict['Nx']
    ny = motion_attr_dict['Ny']
    nz = 1
    #nshot = 
    nshots = len(motion_data_dict['Shot number'])//(nx*ny*nz)
    if (nx > 1 and ny == 1 and nz == 1):
        geom = 'x-line'
    else:
        raise ValueError ('Currently only support x-line')
    print('geometry is::', geom)
    print(f'Nx: {nx}, Ny: {ny}, Nz: {nz}, NShot: {nshots}')
    return nx, ny, nz, nshots
    
    
    
def lapd_6k_config(file_path, motionid = 0):
    """
    Get the motion info and motion data
    """
    motion_group_name = '/Raw data + config/6K Compumotor/'
    motion_subgroup_name = motion_group_name + hdf.get_subgroup_names(file_path, motion_group_name)[0]
    motion_subdataset_name = motion_group_name + hdf.get_dataset_names(file_path, motion_group_name)[0]
    
    motion_attrs_dict =  hdf.get_group_attrs(file_path,motion_subgroup_name)
    motion_data_dict = hdf.get_value(file_path, motion_subdataset_name)
    #print(motion_data_dict)
    return motion_attrs_dict,motion_data_dict

def get_shot_index(nshots,nx, ishot=0, ix=0, iy=0):
        # data loop >> nshots -> xmotion -> ymotion
    return ishot + nshots*(ix + nx*iy)
    # data loop >> nshots -> xmotion -> ymotion -> extra variable steps
    # elif daqconfig == 1:
    #     return ishot + nshotss*(ix + nxx*(iy + nyy*istep))
    # # data loop >> nshots -> extra variable steps -> xmotion -> ymotion
    # elif daqconfig == 2:
    #     return ishot + nshotss*(istep + nsteps*(ix + nxx*iy))
    # # data loop >> nshots
    # elif daqconfig == 3:
    #     return ishot + nshotss*istep
    # # data loop >> nshots -> xmotion -> ymotion -> zmotion
    # elif daqconfig == 4:
    #     return ishot + nshotss*(ix + nxx*(iy + nyy*iz))
    # else:
    #     return 0
