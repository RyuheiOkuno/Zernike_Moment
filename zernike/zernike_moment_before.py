import numpy as np
import math
import os
from function import combination
from function import calc_gm
from pymatgen.io.vasp import Poscar
from function import calc_each_gm
from function import get_unique_list

def calc_rstlist(lst, number):
    n = lst[0]
    l = lst[1]
    m = lst[2]
    alpha = lst[3]
    beta = lst[4]
    u = lst[5]
    mu = lst[6]
    ita = lst[7]
    nyu = lst[8]
    if (n - l) % 2 == 0 and l <= n and m <= l and nyu <= (n - l)/2 and u <= m:
        if mu <= int((l - m)/2) and alpha <= nyu:
            if beta <= (nyu - alpha) and ita <= mu:
                r = 2 * (ita + alpha) + u
                s = 2 * (mu - ita + beta) + m - u
                t = 2 * (nyu - alpha - beta - mu) + l - m
                num = r + s + t
                if num <= n and num == number:
                    return [n, l, m, r, s, t, alpha, beta, u, mu, ita, nyu]


def calc_zernike_coeff(lst, sum_of_rst):    #lst is a nlmrst_list
    n = lst[0]
    l = lst[1]
    m = lst[2]
    k = (n - l)/2
    PATH = "/home/ryuhei/zernike_moment/zernike_coeff/coeff_list/"
    LST_PATH = os.path.join(PATH, str(n)) 
    coeff_list = []
    result = 0
    result_list = []
    path = os.path.join(LST_PATH, str(sum_of_rst) + ".npy")
    coeff = np.load(path)
    for a_file in coeff:
        nlmrst_list = list(a_file[0:6])
        if nlmrst_list != lst:
            pass
        else:
            coeff_list.append(a_file)
    for each in coeff_list:
        [alpha, beta, u, mu, ita, nyu] = list(each[6:])
        a = combination(mu, ita) * combination((l - mu), (m + mu))
        b = ((-1.0) ** mu) * (2.0 ** (-2 * mu)) * combination(l, mu)
        c = ((-1.0)**(m - u)) * combination(m, u) * (1j)**u
        d = combination((nyu - alpha), beta) * combination(nyu, alpha)
        e = (-1.0) ** (k + nyu) * combination(2 * k, k) * combination(k, nyu)
        f = combination((2 * (k + l + nyu) + 1), 2 * k)/(2.0 ** (2 * k))
        g = np.sqrt((2 * l + 4 * k + 3)/3) / (combination((k + l + nyu), k))
        q = e * f * g
        h = np.sqrt((2.0*l + 1) * math.factorial((l+m)) * math.factorial((l-m)))
        i = (2.0 ** ((-1) * m))/math.factorial(l)
        result = result + a*b*c*d*q*h*i
    coeffs = lst[0:6]
    coeffs.append(result)
    if result == 0:
        pass
    else:
        result_list.extend(coeffs)
    return result_list

def calc_zernike_moment(lst, POSCAR_PATH):   ##lst is a nlm_list
    PATH = "/home/ryuhei/zernike_moment/zernike_coeff/h_list/"
    [n, l, m] = lst
    COEFF_PATH = os.path.join(PATH, str(n))
    coeff_array = np.zeros(7)
    rsth_list = []
    c = 0
    for i in range(n + 1):
        each_path = os.path.join(COEFF_PATH, str(i) + ".npy")
        each_coeff = np.load(each_path)
        coeff_array = np.vstack((coeff_array, each_coeff))
    coeff_array = coeff_array[1:]   #exclude zeros in first line
    for each in coeff_array:
        each_list = list(each)
        nlm_list = each_list[0:3]
        if nlm_list == lst:
            rsth_list.append(each_list[3:])
    for a_file in rsth_list:
        geomet_moment = calc_gm(a_file[0:3], POSCAR_PATH)
        d = geomet_moment * np.conj(a_file[3])
        c = c + d
    result = 3 * c/(4 * math.pi)
    return result

def calc_each_zernike(lst, POSCAR_PATH, cut_off):
    PATH = "/home/ryuhei/zernike_moment/data/working/h_list/"
    [n, l, m] = lst
    COEFF_PATH = os.path.join(PATH, str(n))
    coeff_array = np.zeros(7)
    rsth_list = []
    data_path = os.path.join(POSCAR_PATH, "POSCAR")
    poscar = Poscar.from_file(data_path)
    atom_num_list = poscar.structure.atomic_numbers
    num = len(atom_num_list)
    c = np.zeros(num)
    for i in range(n + 1):
        each_path = os.path.join(COEFF_PATH, str(i) + ".npy")
        each_coeff = np.load(each_path)
        coeff_array = np.vstack((coeff_array, each_coeff))
    coeff_array = coeff_array[1:]   #exclude zeros in first line
    for each in coeff_array:
        each_list = list(each)
        nlm_list = each_list[0:3]
        if nlm_list == lst:
            rsth_list.append(each_list[3:])
    for a_file in rsth_list:
        geomet_moment = calc_each_gm(a_file[0:3], POSCAR_PATH, cut_off=cut_off)
        d = geomet_moment * np.conj(a_file[3])
        c = c + d
    result = 3 * c/(4 * math.pi)
    return result

def make_nlm_list(MAX_ORDER):
    DATA_PATH = "/home/ryuhei/zernike_moment/zernike_coeff/coeff_list/"
    PATH = DATA_PATH + str(MAX_ORDER)
    lst = []
    for i in range(MAX_ORDER + 1):
        each_path = os.path.join(PATH, str(i) + ".npy")
        each = np.load(each_path)
        for a_file in each:
            nlm_list = list(a_file[0:3])
            lst.append(nlm_list)
    result = get_unique_list(lst)
    return result


