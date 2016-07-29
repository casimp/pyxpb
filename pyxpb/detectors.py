from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import matplotlib.pyplot as plt

from pyxpb.conversions import tth_to_q, e_to_q, e_to_w, q_to_e
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from pyxpb.peaks import Peaks, Rings


class EnergyDetector(Peaks):
    def __init__(self, phi, two_theta, energy_bins, energy_v_flux,
                 gauge_param, energy_res):
        """ Creates instance of energy dispersive x-ray detector.

        Creates EDXD detector (array) with associated detector parameters
        (i.e. 2theta, flux distribution, energy and energy resolution).

        Inherits from Peaks class, allowing for the calculation/estimation and
        visualisation of intensity profiles for materials or combinations
        of materials/phases.

        Args:
            phi (np.ndarray): Angle for each of the detectors in detector array
            two theta (float): Slit angle (rad)
            energy_bins (np.ndarray): Energy bins (keV)
            energy_v_flux (tuple): Tuple containing energy v flux measurements
            energy_sigma (float, tuple): Energy resolution of detector or
                                         tuple with energy v resolution
        """
        self.method = 'edxd'
        self.two_theta = two_theta
        self.phi = phi
        self.energy = energy_bins
        self.q = e_to_q(self.energy, two_theta)

        # Error in angle (alpha)
        alpha = energy_gauge(*gauge_param, False)[1]

        # Energy resolution wrt. energy (used to define FWHM)
        if isinstance(energy_res, (int, float)):
            e, res = [energy_bins[0], energy_bins[-1]], [energy_res] * 2
        else:
            e, res = energy_res[0], energy_res[1]
        e_res = InterpolatedUnivariateSpline(e, res, k=1, ext=3)

        self.sigma_q = fwhm_energy(e_res, two_theta, alpha)

        # Flux wrt. energy/q
        q, flux = e_to_q(energy_v_flux[0], two_theta), energy_v_flux[1]
        self.flux_q = InterpolatedUnivariateSpline(q, flux, k=1, ext=3)

        # Empty dicts for storing peaks / materials
        self.a, self.sigma, self.q0 = {}, {}, {}
        self.materials, self.hkl = {}, {}


class MonoDetector(Rings):
    def __init__(self, shape, pixel_size, sample_detector, energy,
                 energy_sigma):
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
        self.shape, self.pixel_size = shape, pixel_size
        self.energy, self.sample_detector = energy, sample_detector

        #  Instantiate position array (r, phi, 2theta, q) for detector
        y, x = np.ogrid[:float(shape[0]), :float(shape[1])]
        x, y = x - (shape[0] - 1) / 2, y - (shape[1] - 1) / 2
        r = (x ** 2 + y ** 2) ** .5
        self.phi = np.arctan(y / x)
        self.phi[np.logical_and(x < 0, y > 0)] += np.pi
        self.phi[np.logical_and(x < 0, y < 0)] -= np.pi
        self.two_theta = np.arctan(r * self.pixel_size / sample_detector)
        self.q = tth_to_q(self.two_theta, energy)

        # Beam energy variation (used to define FWHM)
        sigma = e_to_q(energy_sigma, np.array([0, self.two_theta.max()]))
        self.sigma_q = interp1d([0, self.q.max()], sigma)

        # Flux wrt. energy/q - i.e. no variation for mono.
        self.flux_q = interp1d([0, self.q.max()], [1, 1])

        # Empty dicts for storing peaks / materials
        self.a, self.sigma, self.q0 = {}, {}, {}
        self.materials, self.hkl = {}, {}

filename = os.path.join(os.path.dirname(__file__), 'data/i12_flux.csv')
i12_flux = np.loadtxt(filename, delimiter=',')


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


def fwhm_energy(e_res, ttheta, alpha):
    C = 4 * np.pi / 1e10

    def fwhm(q):
        energy = q_to_e(q, ttheta)
        d_e = e_res(energy) / 2

        return C * ((np.sin((ttheta + alpha)/2) / e_to_w(energy + d_e)) -
                    (np.sin((ttheta - alpha)/2) / e_to_w(energy - d_e)))

    return fwhm


# def i12_fwhm_b(energy, p=1):
#     C = 4 * np.pi / 1e10
#     ttheta = np.pi * 5 / 180
#     alpha = i12_gauge(0, False)[1]
#     d_e = i12_e_res(energy)
#     print((alpha / np.tan(ttheta)))
#     #return p * np.sqrt((2 * d_e / energy)**2 + (2.354**2)*(0.13*2.960/(energy*1000)) + (2 * alpha / np.tan(ttheta))**2) * energy
#     return p * np.sqrt((2 * d_e / energy) ** 2 + (2 * alpha / np.tan(ttheta)) ** 2) * energy


i12_energy = EnergyDetector(phi=np.linspace(-np.pi, 0, 23),
                            two_theta=np.pi * (5 / 180),
                            energy_bins=np.linspace(0, 180.5, 4096),
                            energy_v_flux=(i12_flux[:, 0], i12_flux[:, 1]),
                            gauge_param=(0.15, 0.25, 1455, 553, 0, np.pi/36),
                            energy_res=([50, 150], [0.5*0.007, 0.5*0.004]))
