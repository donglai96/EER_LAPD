"""
This file is the general helper for hdf5
"""

import h5py
import numpy as np


def get_group_attrs(hdf5_path, group_name):
    """
    Retrieves all attributes of a specified group in an HDF5 file.

    :param hdf5_path: Path to the HDF5 file.
    :param group_name: Name of the group within the HDF5 file.
    :return: Dictionary with attribute names as keys and attribute values as values.
    """
    attrs_dict = {}
    # Open the HDF5 file in read mode
    with h5py.File(hdf5_path, 'r') as file:
        # Access the specified group
        if group_name in file:
            group = file[group_name]
            # Iterate over attributes of the group and add them to the dictionary
            for attr_name, attr_value in group.attrs.items():
                
                if isinstance(attr_value, np.bytes_):
                    # Decode HDF5 bytes string to a regular Python string
                    #value = attr_value.decode('utf-8')
                    attr_value_new = attr_value.decode('ISO-8859-1') 
                    # for the lapd file this should be iso instead of utf-8 i guess, because this is alfven in the description
                    attrs_dict[attr_name] = attr_value_new
                    
                else:
                    attrs_dict[attr_name] = attr_value
        else:
            print(f"Group '{group_name}' not found in the HDF5 file.")
    return attrs_dict


def get_subgroup_names(hdf5_path, group_name):
    with h5py.File(hdf5_path, 'r') as file:
        ll = list(file[group_name].keys())  # keys are used to grab the object

        # values are used to check for type (this get list of objects (groups,
        #  datasets))
        lval = list(file[group_name].values())
        subgroup_names = []
        for ii in range(len(lval)):
            item = lval[ii]
            if isinstance(item, h5py.Group):
                subgroup_names.append(ll[ii])
    return subgroup_names


def get_dataset_names(hdf5_path, group_name): #headers are also dataset
    with h5py.File(hdf5_path, 'r') as file:
        
        ll = list(file[group_name].keys())  # keys are used to grab the object

        # values are used to check for type (this get list of objects (groups,
        #  datasets))
        lval = list(file[group_name].values())
        dataset_names = []
        for ii in range(len(lval)):
            item = lval[ii]
            if isinstance(item, h5py.Dataset):
                dataset_names.append(ll[ii])
    return dataset_names
           
           
def get_value(hdf5_path, dataset_name):

    with h5py.File(hdf5_path, 'r') as file:
        
        data_dict = {}
        dataset = file[dataset_name]
    # Loop through each field in the dataset's dtype
        for field in dataset.dtype.fields.keys():
            # For each field, extract the data and add it to the dictionary
            data_dict[field] = dataset[field][:]
    return data_dict
