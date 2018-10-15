'''
This module shapes descriptors
Using atom information and covariance
'''

# Import modules
import numpy as np
from scipy import stats as sp
import sys
import os
from distutils.util import strtobool


class Descriptor():
    def __init__(self):
        self.descriptor = None

    def sort_descriptor(self, des, atom_num_lst, element=None):
        # Sort descriptors to each struct
        # If you want to use atomic info, please set element
        if element is None:
            val = des
        else:
            val = np.hstack((des, element))

        # Exclude nan data
        ex_nan = val[:, ~np.isnan(val).any(axis=0)]

        # sort val by atom_num
        all_rep = []
        index = 0
        for i, atom_num in enumerate(atom_num_lst):
            fin = index + atom_num
            each = ex_nan[index:fin].reshape(atom_num, -1)
            all_rep.append(each)
            index += atom_num
        if sum(atom_num_lst) != index:
            raise Exception('Error!!')
        else:
            self.all_rep = np.array(all_rep)

    def generate_variance_des(self, icov):
        # If icov=True, use covarinace
        all_rep = self.all_rep
        descriptor = []
        for each_rep in all_rep:
            des = atomic_rep_to_compound_descriptor(each_rep, icov=icov)
            descriptor.append(des)
        working = np.array(descriptor).reshape(len(all_rep), -1)

        # Standlize array
        result = sp.zscore(working, axis=0)
        return result


def atomic_rep_to_compound_descriptor(rep, mom_order=2, icov=True):

    if (rep.ndim == 1):
        rep = np.reshape(rep, (-1, rep.shape[0]))

    d = []
    d.extend(np.mean(rep, axis=0))
    if (mom_order > 1):
        d.extend(np.std(rep, axis=0))
    if (mom_order > 2):
        d.extend(sp.stats.skew(rep, axis=0))
    if (mom_order > 3):
        d.extend(sp.stats.kurtosis(rep, axis=0))
    if (icov is True):
        if (rep.shape[0] == 1):
            cov = np.zeros((rep.shape[1], rep.shape[1]))
        else:
            cov = np.cov(rep.T)
        for i, row in enumerate(cov):
            for j, val in enumerate(row):
                if (i < j):
                    d.append(val)

    return np.array(d)


# Run part
if __name__ == '__main__':
    '''
    Setting Parameters
    ----------------------------------------------------------------
    '''
    args = sys.argv
    ORDER = int(args[1])
    CUT_OFF = int(args[2])
    WEIGHED = args[3]
    ICOV = bool(strtobool(args[4]))

    # If moment is weighed, there is no need to add ele_lst
    if WEIGHED == 'on':
        WITH_ELE = 'off'
    # If moment isnt weighed, set element_list
    elif WEIGHED == 'off':
        WITH_ELE = 'on'

    '''
    Print calc condition
    ------------------------------------------------------------------
    '''
    print('ORDER = ' + str(ORDER))
    print('CUT_OFF = ' + str(CUT_OFF))
    print('WEIGHED = ' + str(WEIGHED))
    print('WITH_ELE = ' + str(WITH_ELE))
    print('ICOV = ' + str(ICOV))

    '''
    Load necessary data
    ------------------------------------------------------------------
    '''
    # Load atom_num_data and element data
    num_lst = np.load('/home/ryuhei/zernike_moment/data/working/num_list.npy')
    ele_lst = np.load('/home/ryuhei/zernike_moment/data/working/'
                      'element_list.npy')

    # Load descriptor data
    DES_PATH = '/home/ryuhei/zernike_moment/data/descriptor/cohesive/'
    if WEIGHED == 'on':
        PATH = os.path.join(DES_PATH, 'weighed', str(CUT_OFF))
        min_val = 0
    elif WEIGHED == 'off':
        min_val = 1
        PATH = os.path.join(DES_PATH, 'only_struct', str(CUT_OFF))
    des = np.zeros((len(ele_lst), 1))
    for i in range(min_val, ORDER + 1):
        each_path = os.path.join(PATH, str(i) + '.npy')
        each_des = np.load(each_path)
        des = np.hstack((des, each_des))
    descriptor = des[:, 1:]

    # Generate Instance and calc descriptor
    des_ins = Descriptor()
    if WITH_ELE == 'on':
        ele = ele_lst
    elif WITH_ELE == 'off':
        ele = None
    des_ins.sort_descriptor(descriptor, num_lst, ele)
    result = des_ins.generate_variance_des(ICOV)
    print(result)
    print(result.shape)

    '''
    Save result
    -------------------------------------------------------------------
    '''
    SAVE_PATH = '/home/ryuhei/zernike_moment/input/cohesive/'
    if WEIGHED == 'on':
        SAVE_PATH = os.path.join(SAVE_PATH, 'weighed', str(CUT_OFF))
    elif WEIGHED == 'off':
        SAVE_PATH = os.path.join(SAVE_PATH, 'only_struct', str(CUT_OFF))

    if ICOV is True:
        save_path = os.path.join(SAVE_PATH, 'with_cov', str(ORDER) + '.npy')
    else:
        save_path = os.path.join(SAVE_PATH, 'wo_cov', str(ORDER) + '.npy')
    np.save(save_path, result)
