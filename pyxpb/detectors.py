from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np

from xrdpb.conversions import tth_to_q, e_to_q
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from xrdpb.peaks import Peaks, Rings


class EnergyDetector(Peaks):
    def __init__(self, phi, two_theta, energy_bins, energy_v_flux,
                 energy_sigma):
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

        # Energy resolution wrt. energy (used to define FWHM)
        if isinstance(energy_sigma, (int, float)):
            self.sigma_q = interp1d([0, self.q.max()], [energy_sigma] * 2)
        else:
            q, sigma = e_to_q(energy_sigma[0], two_theta), energy_sigma[1]
            self.sigma_q = InterpolatedUnivariateSpline(q, sigma, k=1, ext=3)

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
        self.phi = np.arctan(y / x)# - np.arccos(x / r)  # np.arcsin(y / r)  # np.arctan(y / x)
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

i12_energy = EnergyDetector(phi=np.linspace(-np.pi, 0, 23),
                            two_theta=np.pi * (5 / 180),
                            energy_bins=np.linspace(0, 200, 4000),
                            energy_v_flux=(i12_flux[:, 0], i12_flux[:, 1]),
                            energy_sigma=([50, 150], [0.5*0.007, 0.5*0.004]))
