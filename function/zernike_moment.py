'''
This program calcs zernike moment and zernike descriptor
The coeff list is calculated before
'''

# Import modules
import numpy as np
from pymatgen.io.vasp import Poscar
import copy

class ZernikeDescriptor():

    # Set init value
    def __init__(self, number):
        self.descriptor = None
        self.number = number

    # Make nl list on the conditon that (n - l) is even
    def make_nl_list(self):
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
        self.nl_list = lst

    # Make total nlm list to calculate moment
    def make_total_list(self):
        # Get list
        lst = copy.copy(self.nl_list)
        # Create empty list
        total_lst = []

        for a_file in lst:
            [n, l] = a_file
            for i in range(0, l + 1):
                nl = [n, l]
                nl.append(i)
                total_lst.append(nl)

        # Set value
        self.calc_lst = total_lst

    # Generate descriptor of only structure
    def generate_struct_descriptor(self, POSCAR_PATH, cut_off):
        '''
        generate zernike descriptor

        input
        -------------
        POSCAR_PATH : str
            the path of POSCAR to be calc
        cut_off : float
            the cut of radius
        -------------
        '''








# Define functuion to be used in this program
def calc_weighed_moment(lst, POSCAR_PATH, cut_off, atom_info=None):
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

    poscar = Poscar.from_file(POSCAR_PATH)
    st = poscar.structure
    sites = st.sites

    # Set result value to append result of calculation
    result_arr = np.zeros((1, len(atom_info['H'])))

    for each in sites:
        # Get neighbor atoms's coords information
        neighbors = st.get_neighbors(each, r=cut_off)
        neigh_coord = [a_file[0].coords - each.coords for a_file in neighbors]
        trans_arr = np.array(neigh_coord)
        # Mapping array by cut_off radius
        # Use this array in calculation
        mapped_array = trans_arr/cut_off

        # Get neighbor atom's property information
        # If atom_info isnt set, pass this step
        neigh_name = [a_file[0].species_string for a_file in neighbors]
        neigh_info = np.array([atom_info[i] for i in neigh_name])

        # Main part
        # Calc geometrical moment
        val = [calc_exponet_val(coord, lst) for coord in mapped_array]
        arr = np.array(val).reshape(1, len(val))/len(val)

        # Get weighed value
        result = np.dot(arr, neigh_info)
        result_arr = np.vstack((result_arr, result))
    return result_arr[1:]

# Calc normal geometric moment
def calc_geomet_moment(lst, POSCAR_PATH, cut_off):
    result_list = []
    poscar = Poscar.from_file(POSCAR_PATH)
    st = poscar.structure
    sites = st.sites
    for each in sites:

        # Get neighbor atoms's coords information
        neighbors = st.get_neighbors(each, r=cut_off)
        neigh_coord = [a_file[0].coords - each.coords for a_file in neighbors]
        trans_arr = np.array(neigh_coord)
        # Mapping array by cut_off radius
        # Use this array in calculation
        mapped_array = trans_arr/cut_off

        # Main part
        val = [calc_exponet_val(coord, lst) for coord in mapped_array]
        result = np.mean(val)
        result_list.append(result)
    return np.array(result_list)

# Calc exponet value of coords
def calc_exponet_val(coords, exponent):
    if len(coords) != len(exponent):
        print('The Size Must Be Same!!!')
    else:
        x, y, z = coords
        r, s, t = exponent
        if x == 0:
            x_val = 0
        else:
            x_val = x ** r
        if y == 0:
            y_val = 0
        else:
            y_val = y ** s
        if z == 0:
            z_val = 0
        else:
            z_val = z ** t
        val = x_val * y_val * z_val
    return val
