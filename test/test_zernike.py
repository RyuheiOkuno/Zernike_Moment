'''
This is test script of zernike moment and zernike descriptor
Test is done for function listed below
1.make nlm list
2.calc_geomet_moment
3.calc_zernike_moment
4.generate zernike descriptor(cofirm the rotation invariants)
'''

# Import modules
from zernike import zernike_moment
import numpy as np
import os
import unittest
import math

# load data
INFO = np.load('/home/ryuhei/zernike_moment/data/working/atom_info.npy')


class TestZernike(unittest.TestCase):
    '''
    test class of zernike
    '''

    def test_make_list(self):
        COEFF_PATH = '/home/ryuhei/zernike_moment/data/working/h_list/3/'
        coeff = np.zeros((1, 7))
        for i in range(4):
            coeff_path = os.path.join(COEFF_PATH, str(i) + '.npy')
            each = np.load(coeff_path)
            coeff = np.vstack((coeff, each))
        coeff = coeff[1:]
        zd = zernike_moment.ZernikeDescriptor(3)
        zd.make_nl_lst()
        zd.add_mvalue()
        actual = zd.nlm_lst
        expect = [[[3, 1, 0], [3, 1, 1]],
                  [[3, 3, 0], [3, 3, 1], [3, 3, 2], [3, 3, 3]]]
        self.assertEqual(actual, expect)

    def test_calc_normal_geomom(self):
        lst = [0, 2, 2]
        POSCAR_PATH = './POSCAR'
        cut_off = 6
        atom_info = None
        actual = zernike_moment.calc_geomet_moment(lst, POSCAR_PATH, cut_off,
                                                   atom_info)
        expect = np.array([32/729]).reshape(1, 1)
        self.assertEqual(actual, expect)

    def test_calc_weighed_geomom(self):
        lst = [0, 2, 2]
        POSCAR_PATH = './POSCAR'
        cut_off = 6
        atom_info = INFO.item()
        actual = zernike_moment.calc_geomet_moment(lst, POSCAR_PATH, cut_off,
                                                   atom_info)
        expect = (atom_info['Al'] * (32/729)).reshape(1, 24)
        act = actual[~np.isnan(np.array(actual, dtype=float))]
        exp = expect[~np.isnan(np.array(expect, dtype=float))]
        np.testing.assert_almost_equal(act, exp, decimal=5)

    def test_calc_zernike_moment(self):
        COEFF_PATH = '/home/ryuhei/zernike_moment/data/working/h_list/2/'
        coeff = np.zeros((1, 7))
        for i in range(3):
            coeff_path = os.path.join(COEFF_PATH, str(i) + '.npy')
            each = np.load(coeff_path)
            coeff = np.vstack((coeff, each))
        coeff = coeff[1:]
        zd = zernike_moment.ZernikeDescriptor(2)
        zd.make_nl_lst()
        zd.add_mvalue()
        zd.get_coeff_info(coeff)
        coeffs = zd.calc_lst[0][0]
        print(zd.nlm_lst[0][0])
        actual = zernike_moment.calc_zernike_moment(coeffs,
                                                    './POSCAR', 4).item()
        expect = (3 * np.sqrt(7/3))/(4 * math.pi)
        self.assertAlmostEqual(actual, expect)

    def test_generate_descriptor(self):
        COEFF_PATH = '/home/ryuhei/zernike_moment/data/working/h_list/4/'
        coeff = np.zeros((1, 7))
        for i in range(5):
            coeff_path = os.path.join(COEFF_PATH, str(i) + '.npy')
            each = np.load(coeff_path)
            coeff = np.vstack((coeff, each))
        coeff = coeff[1:]
        zd = zernike_moment.ZernikeDescriptor(4)
        zd.make_nl_lst()
        zd.add_mvalue()
        zd.get_coeff_info(coeff)
        zd.get_atomic_info(INFO.item())
        # Calc value1
        zernike1 = zd.get_moment_val('./POSCAR', 6)
        val1 = zd.generate_descriptor(zernike1)
        # Calc value2
        zernike2 = zd.get_moment_val('./rotate_POSCAR', 6)
        val2 = zd.generate_descriptor(zernike2)
        np.testing.assert_almost_equal(val1, val2, decimal=3)


if __name__ == '__main__':
    unittest.main()
