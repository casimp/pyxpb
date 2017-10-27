from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import matplotlib.pyplot as plt

from pyxpb.conversions import tth_to_q, e_to_q, h, c, eV, q_to_tth
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from pyxpb.peaks import Peaks, Rings


class EnergyDetector(Peaks):
    def __init__(self, phi, two_theta, energy_bins, flux=None,
                 gauge_param=None, delta_energy=None, F=0.13, e_f=0.003):
        """ Creates instance of energy dispersive x-ray detector.

        Creates EDXD detector (array) with associated detector parameters
        (i.e. 2theta, flux distribution, energy and energy resolution).

        Inherits from Peaks class, allowing for the calculation/estimation and
        visualisation of intensity profiles for materials or combinations
        of materials/phases.

        Args:
            phi (ndarray): Angle for each of the detectors in detector array
            two theta (float): Slit angle (rad)
            energy_bins (ndarray): Energy bins (keV)
            flux (tuple): Tuple containing (energy, flux) values
            gauge_param (tuple): Tuple containing gauge param (a, b, c, e and
                                 h from Knowles et al.)
            delta_energy (float, tuple): Energy resolution of detector or
                                         tuple with energy v resolution
        """
        self.method = 'edxd'
        self.two_theta = two_theta
        self.phi = phi
        self.energy = energy_bins
        self.q = e_to_q(self.energy, two_theta)

        # Calc error in 2theta (alpha) - use I12 setup params if none supplied
        if gauge_param is None:
                gauge_param = (0.15, 0.25, 1455, 553, 0, np.pi/36)
        gauge_temp = tuple(gauge_param ) + (False, )
        alpha = energy_gauge(*gauge_temp)[1]

        # Energy resolution - use 500eV if none, convert to ndarray if tuple
        if delta_energy is None:
            delta_energy = 0.5
        elif isinstance(delta_energy, (tuple, list)):
            delta_energy = np.stack((delta_energy[0], delta_energy[1]), 1)

        self._fwhm = fwhm_polyest_e(delta_energy, two_theta, alpha, F, e_f)

        # Flux wrt energy - use const. if none, convert to ndarray if tuple
        if isinstance(flux, (tuple, list)):
            flux = np.stack((flux[0], flux[1]), 1)
        elif flux is None:
            flux = np.array([[0, 1], [self.q.max, 1]])

        q_f = e_to_q(flux[:, 0], two_theta)
        self.flux_q = InterpolatedUnivariateSpline(q_f, flux[:, 1], k=1, ext=3)

        # Parameter store for save/reload
        self._det_param = {'phi': phi, 'two_theta': two_theta,
                           'energy_bins': energy_bins,
                           'gauge_param': gauge_param,
                           'delta_energy': delta_energy,
                           'flux': flux, 'F': F, 'e_f': e_f}

        # Empty dicts for storing peaks / materials
        self.a, self.fwhm, self.q0 = {}, {}, {}
        self.materials, self.hkl = {}, {}
        self._back = np.ones(1)


class MonoDetector(Rings):
    def __init__(self, shape, pixel_size, sample_detector, energy,
                 delta_energy):
        """ Creates instance of monochromatic (area) XRD detector.

        Creates XRD detector with correct geometry and experimental setup
        (i.e. sample to detector distance, energy and energy resolution).

        Inherits from Rings/Peaks classes, allowing for the
        calculation/estimation and visualisation of intensity profiles and
        Debye-Scherrer rings for materials or combinations of materials/phases.

        Args:
            shape (tuple): Detector shape (x, y) in pixels
            pixel_size (float): Pixel size (mm)
            sample_detector (float): Sample to detector distance (mm).
            energy (float): X-ray energy (keV)
            energy_sigma (float): Energy resolution (keV)
        """
        self.method = 'mono'
        self.energy = energy

        #  Instantiate position array (r, phi, 2theta, q) for detector
        y, x = np.ogrid[:float(shape[0]), :float(shape[1])]
        x, y = x - (shape[0] - 1) / 2, y - (shape[1] - 1) / 2
        r = (x ** 2 + y ** 2) ** .5
        self.phi = np.arctan(y / x)
        self.phi[np.logical_and(x < 0, y > 0)] += np.pi
        self.phi[np.logical_and(x < 0, y < 0)] -= np.pi
        self.two_theta = np.arctan(r * pixel_size / sample_detector)
        self.q = tth_to_q(self.two_theta, energy)

        # Beam energy variation (used to estimate FWHM)
        tth = np.linspace(0, self.two_theta.max(), 100)
        fwhm_q = e_to_q(delta_energy, tth)
        fwhm_tth = q_to_tth(fwhm_q, energy)

        # FWHM should vary approx. with Caglioti polynomial
        self._fwhm = [0.01, 0.01]#np.polyfit(np.tan(tth / 2), fwhm_tth **2, 3)

        # Flux wrt. energy/q - i.e. no variation for mono.
        self._flux = np.array([[0, 1], [self.q.max(), 1]])
        # assert np.array_equal(self._flux[:, 0], [0, self.q.max()])
        self.flux_q = interp1d(self._flux[:, 0], self._flux[:, 1])

        # Parameter store for save/reload
        self._det_param = {'shape': shape, 'pixel_size': pixel_size,
                           'sample_detector': sample_detector,
                           'energy': energy, 'delta_energy': delta_energy}

        # Empty dicts for storing peaks / materials
        self.a, self.fwhm, self.q0 = {}, {}, {}
        self.materials, self.hkl = {}, {}
        self._back = np.ones(1)


def energy_gauge(a, b, c, e, h, ttheta, plot=True):
    """ As per Rowles, M (2011) - parallel model. """
    alpha = np.arctan((a + b) / (2 * c))
    beta = ttheta - alpha
    gamma = np.pi - ttheta - alpha
    d = a * c / (a + b)
    f = (d + e) * np.sin(alpha) / np.sin(beta)
    g = (d + e) * np.sin(alpha) / np.sin(gamma)

    n = h / (2 * np.tan(beta))
    p = - h / (2 * np.tan(gamma))
    l_total = (f + n) + (g + p)
    x = [0, 2 * n, l_total, l_total - 2 * p, 0]
    y = [0, h, h, 0, 0]
    if plot:
        plt.plot(x, y)
    return l_total, alpha


def fwhm_polyest_e(delta_energy, two_theta, alpha, F, e_f):
    """ Estimate the fwhm profile in an energy dispersive detector.

    Calculates the fwhm wrt. the measurement accuracy of the detectors,
    the Fano factor contribution and angle variation (due to slit size
    and positioning.

    Args:
        delta_energy (ndarray): Energy resolution wrt. energy (keV)
        two_theta (float): 2theta in radians
        alpha (float): Half the full angular variation.
        F (float): Fano factor (approx. 0.13 for Germanium)
        e_f (float): Energy to make electron-hole pair (approx. 3e-3 keV)

    Returns:
        ndarray: 1d array containing the estimated polynomial (k=2).
    """
    # d_E^2 used to calc FWHM -> calc polynomial such that d_E^2 = A*E + B
    if isinstance(delta_energy, (int, float)):
        res_sq = [0, delta_energy ** 2]
    else:
        e, res = delta_energy[:, 0], delta_energy[:, 1]
        res_sq = np.polyfit(e, [i ** 2 for i in res], 1)

    # Polynomial terms that should fit fwhm squared
    fw_base = [(2 * alpha / np.tan(two_theta)) ** 2, F * e_f * 2.35 ** 2, 0]
    fw_e_sq = np.polyadd(fw_base, res_sq)

    # Conversion factor to put fwhm squared in terms of q
    e_q = 1000 * eV * 4 * np.pi * np.sin(two_theta / 2) / (h * c * 1e10)
    return [fw_e_sq[0], fw_e_sq[1] * e_q, fw_e_sq[2] * e_q ** 2]


filename = os.path.join(os.path.dirname(__file__), 'data/i12_flux.csv')
i12_flux = np.loadtxt(filename, delimiter=',')


def i12_energy():
    return EnergyDetector(phi=np.linspace(-np.pi, 0, 23),
                          two_theta=np.pi * (5 / 180),
                          energy_bins=np.linspace(0, 180.5, 4096),
                          flux=i12_flux,
                          gauge_param=(0.15, 0.25, 1455, 553, 0, np.pi/36),
                          delta_energy=([50, 150], [0.35, 0.6]))
