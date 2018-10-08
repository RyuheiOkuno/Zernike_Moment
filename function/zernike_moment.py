'''
This program calcs zernike moment and zernike descriptor
The coeff list is calculated before
'''

# Import modules
import numpy as np
from pymatgen.io.vasp import Poscar
import copy
import functools
import math

class ZernikeDescriptor():

    # Set init value
    def __init__(self, number):
        self.number = number
        self.atom_info = None

    # Make nl list on the conditon that (n - l) is even
    def make_nl_lst(self):
        # Get max order
        num = self.number
        lst = []
        for i in range(0, num + 1):
            if (num - i) % 2 == 0:
                each = [num, i]
                lst.append(each)
            else:
                pass
        # Set value
        self.nl_lst = lst

    # Make total nlm list to calculate moment
    def add_mvalue(self):
        # Get list
        lst = copy.copy(self.nl_lst)
        # Create empty list
        total_lst = []

        for a_file in lst:
            each_lst = []
            [n, l] = a_file
            for i in range(0, l + 1):
                nl = [n, l]
                nl.append(i)
                each_lst.append(nl)
            total_lst.append(each_lst)

        # Set value
        self.nlm_lst = total_lst

    # Get coeff list data of given order
    def get_coeff_info(self, all_coeff_data):
        '''
        input
        --------
        all_coeff_data : np.array or list (7 * N shape)
            all zernike coeff data arrays
        --------
        '''
        # Get list value to calc
        total_lst = self.nlm_lst

        total_calc_lst = []
        # Roop for nlm calc lst
        for a_file in total_lst:
            calc_lst = []
            for b_file in a_file:
                sorted_ = [list(each)[3:] for each in all_coeff_data
                           if list(each)[0:3] == b_file]
                calc_lst.append(sorted_)
            total_calc_lst.append(calc_lst)

        # Set value
        self.calc_lst = total_calc_lst

    # Get atomic information
    def get_atomic_info(self, atom_info):
        '''
        input
        --------
        atom_info : dict
            Dict data of atom info
            if you want to calc weighed moment, please set this value
            Ex) {'Al' : (1, 2, 1,61, 27)}
        --------
        '''
        self.atom_info = atom_info

    # Calc zernike moment of structure and atomic information
    # If you want to use atomic function, please set atom_info beforehead
    def get_moment_val(self, POSCAR_PATH, cut_off):
        '''
        calc zernike moment from POSCAR file
        only using structural information
        input
        -------------
        POSCAR_PATH : str
            the path of POSCAR to be calc
        cut_off : float
            the cut off radius
        -------------
        '''
        # Get self value
        calc_lst = self.calc_lst
        atom_info = self.atom_info
        # Create empty list to add
        zernike_moments = []
        # Roop for all calc_lst
        for each in calc_lst:
            # Calc each component of zernike moment
            zerval = list(map(functools.partial(calc_zernike_moment,
                                                POSCAR_PATH=POSCAR_PATH,
                                                cut_off=cut_off,
                                                atom_info=atom_info), each))
            zernike_moments.append(np.array(zerval))
        # Set value
        self.zernike = zernike_moments

    def generate_descriptor(self):
        '''
        generate zernike descriptor from zernike moment value
        '''
        # Get zernike moment and nlm_lst value
        zermom = self.zernike
        nlm_lst = self.nlm_lst

        atom_num = np.array(zermom[0]).shape[1]
        descriptors = np.zeros((atom_num, 1))
        # Get -m value
        for i, lst in enumerate(nlm_lst):
            tval = zermom[i]
            all_moment = []
            for j, a_file in enumerate(lst):
                each_matrix = tval[j]
                if a_file[-1] == 0:
                    all_moment.append(each_matrix)
                else:
                    all_moment.append(each_matrix)
                    conj = ((-1) ** a_file[-1]) * np.conj(each_matrix)
                    all_moment.append(conj)
        self.all_moment = all_moment
            #descriptor = np.linalg.norm(np.array(all_moment), axis=0)
            #descriptors = np.hstack((descriptors, descriptor))
        #self.descriptors = descriptors[:, 1:]

'''
Define function to be used in this program
'''

def calc_zernike_moment(coeffs, POSCAR_PATH, cut_off, atom_info=None):
    '''
    input
    ---------
    coeffs : all [n, l, m, h] to calc
        The length of each lst must be 4
        Ex)coeffs = [[1,2,3,-i], [1,1,3,2+i]]
    ---------
    '''
    # split rst_list and h_list
    rst = np.array(coeffs)[:, 0:3]
    h_val = np.array(coeffs)[:, -1].reshape(rst.shape[0], 1)
    geomom = list(map(functools.partial(calc_geomet_moment,
                                        POSCAR_PATH=POSCAR_PATH,
                                        cut_off=cut_off,
                                        atom_info=atom_info), rst))
    zermom = [geomom[i] * h_val[i] for i in range(len(geomom))]
    zermom = (3 * np.sum(np.array(zermom), axis=0))/(4 * math.pi)
    return zermom

def calc_geomet_moment(lst, POSCAR_PATH, cut_off, atom_info=None):
    '''
    This function callulates geomet moment
    weighs each componets value such as electronegativity and so on
    input
    ---------
    lst : list
        exponent of each values list
    POSCAR_PATH : str
        The path of POSCAR to be calculated
    cut_off : flota
        cut_off radius
    atom_info : dict (default is None)
        Dict data of atom info
        if you want to calc weighed moment, please set this value
        Ex) {'Al' : (1, 2, 1,61, 27)}
    '''
    # Get Poscar information
    poscar = Poscar.from_file(POSCAR_PATH)
    st = poscar.structure
    sites = st.sites

    # Create empty array to add result value
    if atom_info == None:
        result_arr = np.zeros((1, 1))
    else:
        result_arr = np.zeros((1, len(atom_info['H'])))

    for each in sites:
        # Get neighbor atoms's coords information
        neighbors = st.get_neighbors(each, r=cut_off)
        neigh_coord = [a_file[0].coords - each.coords for a_file in neighbors]
        trans_arr = np.array(neigh_coord)
        # Mapping array by cut_off radius
        # Use this array in calculation
        mapped_array = trans_arr/cut_off

        # Main part
        # Calc geometrical moment
        val = [calc_exponet_val(coord, lst) for coord in mapped_array]
        # If atom_info = None, calc geomet moment only from strucutre
        if atom_info == None:
            result = np.array(np.mean(val)).reshape(1, 1)
            result_arr = np.vstack((result_arr, result))
        # If atom_info is set, use atom_info too
        else:
            arr = np.array(val).reshape(1, len(val))/len(val)
            # Get neighbor atom's property information
            # If atom_info isn't set, pass this step
            neigh_name = [a_file[0].species_string for a_file in neighbors]
            neigh_info = np.array([atom_info[i] for i in neigh_name])
            # Get weighed value
            result = np.dot(arr, neigh_info)
            result_arr = np.vstack((result_arr, result))
    return result_arr[1:]

# Calc exponet value of coords
def calc_exponet_val(coords, exponent):
    if len(coords) != len(exponent):
        print('The Size Must Be Same!!!')
    else:
        x, y, z = coords
        r, s, t = exponent
        x_val = x ** r
        y_val = y ** s
        z_val = z ** t
        val = x_val * y_val * z_val
    return val
