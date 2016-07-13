from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

h = 6.62607004 * 10 ** -34
c = 2.99792458 * 10 ** 8
eV = 1.60218 * 10 ** -19


def e_to_w(energy):
    """ Takes photon energy (keV) -> returns wavelength (m) """
    energy_j = np.array(energy) * 1000 * eV
    wavelength = h * c / energy_j
    return wavelength


def w_to_e(wavelength):
    """ Takes wavelength (m) -> returns photon energy (keV) """
    energy_j = h * c / np.array(wavelength)
    energy_kev = energy_j / (eV * 1000)
    return energy_kev


def e_to_q(energy, two_theta):
    """ Takes energy (keV) and 2theta (rad) -> returns q (A-1) """
    wavelength = e_to_w(energy)
    q_per_m = np.sin(two_theta / 2) * 4 * np.pi / wavelength
    q_per_A = q_per_m / (10**10)  # convert to A^-1
    return q_per_A


def q_to_e(q, two_theta):
    """ Takes q (A-1) and 2theta (rad) -> returns energy (keV) """
    q_per_m = np.array(q) * (10**10)
    wavelength = np.sin(two_theta / 2) * 4 * np.pi / q_per_m
    return w_to_e(wavelength)


def q_to_2th(q, energy):
    """ Takes q (A-1) and energy (keV) -> returns 2theta (rad) """
    wavelength = e_to_w(energy)
    q_per_m = q * (10**10)
    two_theta = 2 * np.arcsin(q_per_m * wavelength / (4 * np.pi))
    return two_theta


if __name__ == '__main__':
    print(q_to_e(5, np.pi*(1/36)))
