# -*- coding : utf -8 -*-

'''
This module makes atomic information data
Raw data is csv file (atomic_data_20160603.csv)
'''

# import modules
import numpy as np
import pandas as pd
import os

class AtomInfo():
    # Set init
    def __init__(self):
        self.data = None

    # Read csv data
    def get_data(self, file_name):
        '''
        input
        ---------
        file name : str
            the name of data csv file
        ---------
        '''
        df = pd.read_csv(file_name)
        # Get valueable names
        # Exclude first value cuz it is name of atom
        val_name = df.columns[1:]

        # Make dictionary data
        data = df.values
        # Make empty dict
        dict_ = {}
        for row in range(data.shape[0]):
            vec = data[row]
            # Fist value is name of atom
            atom_num = vec[0]
            # Add data to dictionary
            dict_[atom_num] = vec[1:]

        # Set values
        self.val_name = val_name
        self.data = dict_

# Run script
DATA_DIR = '/home/ryuhei/zernike_moment/data/raw/'
FILE_NAME = 'atomic_data_20160603.csv'

if __name__ == '__main__':
    os.chdir(DATA_DIR)
    atom_Ins = AtomInfo()
    atom_Ins.get_data(FILE_NAME)
    np.save('/home/ryuhei/zernike_moment/data/working/atom_val_name.npy',
            atom_Ins.val_name)
    np.save('/home/ryuhei/zernike_moment/data/working/atom_info.npy',
            atom_Ins.data)
