'''
This program is run script of zernike moment
'''

# Import modules
import numpy as np
import os
from distutils.util import strtobool
from joblib import Parallel, delayed
from zernike import zernike_moment
import sys

# Define function
# Calc moment value
def run(zd, data, i):
    poscar_path = os.path.join(DATA_PATH, data, 'POSCAR')
    zernike = zd.get_moment_val(poscar_path, CUT_OFF)
    descriptor = zd.generate_descriptor(zernike)
    print('The number : ' + str(i) + ' is done')
    return descriptor, i


if __name__ == '__main__':
    '''
    ---------------------------------------------
    Parameter Setting
    '''
    args = sys.argv
    ORDER = int(args[1])
    CUT_OFF = int(args[2])
    with_atom = bool(strtobool(args[3]))
    # Calc the order of descriptors
    if ORDER % 2 == 0:
        DES_NUM = (ORDER / 2) + 1
    else:
        DES_NUM = (ORDER + 1) / 2
    '''

    ----------------------------------------------
    Load necessary information
    '''
    # Load data needed
    # Load coeff data
    coeff = np.zeros((1, 7))
    COEFF_PATH = '../../working/h_list/' + str(ORDER)
    for i in range(ORDER + 1):
        coeff_path = os.path.join(COEFF_PATH, str(i) + '.npy')
        each = np.load(coeff_path)
        coeff = np.vstack((coeff, each))
    coeff = coeff[1:]

    # Set atom_info
    atom_info = np.load('../../working/atom_info.npy')
    if with_atom == True:
        atom_info = atom_info.item()
        atom_lengh = atom_info['H'].shape[0]
        DES_LENGTH = int(atom_lengh * DES_NUM)
    else:
        atom_info = None
        DES_LENGTH = int(DES_NUM)

# Get path data to calc
    DATA_PATH = '../../raw/cohesive/descriptors/'
    f = open('../../raw/cohesive/compounds_name')
    data_path = f.read().split()
    print(len(data_path))
    print(data_path[0])

    '''
    -----------------------------------------------

    '''
    # Main part start
    # Genarate instance
    zd = zernike_moment.ZernikeDescriptor(ORDER)
    zd.make_nl_lst()
    zd.add_mvalue()
    zd.get_coeff_info(coeff)
    zd.get_atomic_info(atom_info)

    # Parallel calculation start
    print("Zerike Descriptor Calculation Start!!")
    print("Order = " + str(ORDER))
    print("Cut_Off Radius = " + str(CUT_OFF))
    print("With Atom = " + str(with_atom))
    print('-----------------------------------------------------------------')

    result = Parallel(n_jobs=-1, verbose=10)([delayed(run)(zd, data, i)
                                              for i, data
                                              in enumerate(data_path)])

    print('-----------------------------------------------------------------')
    print("Calc Done For Order " + str(ORDER))

    result.sort(key=lambda x: x[1])
    result_lst = [t[0] for t in result]
    # add descriptors to empty array
    _descriptor = np.zeros((1, int(DES_LENGTH)))
    for a_file in result_lst:
        _descriptor = np.vstack((_descriptor, a_file))
    descriptor = _descriptor[1:]

    if with_atom == True:
        SAVE_PATH = '../../descriptor/weighed/' + str(CUT_OFF)
        save_path = os.path.join(SAVE_PATH, str(ORDER) + '.npy')
        np.save(save_path, descriptor)
    else:
        SAVE_PATH = '../../descriptor/only_struct/' + str(CUT_OFF)
        save_path = os.path.join(SAVE_PATH, str(ORDER) + '.npy')
        np.save(save_path, descriptor)

    print('Descriptor shape is ' + str(descriptor.shape))
