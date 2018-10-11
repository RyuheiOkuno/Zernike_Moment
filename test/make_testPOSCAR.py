'''
This program makes POSCAR file for test
'''

# Import modules
import numpy as np
import random
import math
import copy
from pymatgen.core.structure import Structure
from pymatgen.core.operations import SymmOp


class TestPOSCAR():
    # Set init value
    def __init__(self):
        self.POSCAR = None
        self.trans_POSCAR = None

    # Set structure info
    def set_struct(self, crys_type='sc', length=4, spe='Al'):
        coords = np.array(([0, 0, 0])).reshape(1, 3)
        lattice = np.eye(3) * length
        species = [spe] * coords.shape[0]
        struct = Structure(lattice=lattice, species=species, coords=coords)
        self.struct = struct

    def rotate(self, frac=False):
        R = rotate_matrix()
        arr1 = np.zeros(3)
        R = np.vstack((R, arr1))
        arr2 = np.array([0, 0, 0, 1])
        R = np.vstack((R.T, arr2))
        affine_matrix = R.T
        sym = SymmOp(affine_matrix)
        init_st = copy.copy(self.struct)
        if frac is True:
            init_st.apply_operation(sym, fractional=True)
        else:
            init_st.apply_operation(sym, fractional=False)
        self.rotate = init_st


def rotate_matrix():
    px = random.uniform(0, 2 * math.pi)
    py = random.uniform(0, 2 * math.pi)
    pz = random.uniform(0, 2 * math.pi)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(px), np.sin(px)],
                   [0, -np.sin(px), np.cos(px)]])
    Ry = np.array([[np.cos(py), 0, -np.sin(py)],
                   [0, 1, 0],
                   [np.sin(py), 0, np.cos(py)]])
    Rz = np.array([[np.cos(pz), np.sin(pz), 0],
                   [-np.sin(pz), np.cos(pz), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R


if __name__ == '__main__':
    test = TestPOSCAR()
    test.set_struct()
    test.struct.to(filename='POSCAR', fmt='POSCAR')
    test.rotate()
    test.rotate.to(filename='rotate_POSCAR', fmt='POSCAR')
