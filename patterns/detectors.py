from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from patterns.conversions import tth_to_q, q_to_tth, e_to_q, q_to_e
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from patterns.peaks import Peaks, Rings


# fname = os.path.join(os.path.dirname(__file__), 'data/i12_energy_distribution.csv')
# i12_e_flux = np.loadtxt(fname, delimiter=',')
#
#
# i12 = {'energy': i12_e_flux[:, 0],
#        'flux': i12_e_flux[:, 1],
#        'res_e': {'energy': [50, 150],
#                  'delta': [0.5 * 0.007, 0.5 * 0.004]},
#        'bins': 4000}
#
# edxd_info = {'i12': i12, 'id15': i12}
#
# energy_i12 = BaseEnergyDetector()


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
            energy_flux (tuple): Tuple containing energy v flux measurements
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
        x, y = x - shape[0] / 2, y - shape[1] / 2
        r = (x ** 2 + y ** 2) ** .5
        self.phi = np.arctan(y / x)
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




