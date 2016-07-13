from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from multiplicity import peak_details
from beamline_details import edxd_info
from conversions import e_to_q, q_to_e, q_to_2th
from intensity_factors import scattering_factor, temperature_factor, lp_factor
from detectors import extract_q

plt.style.use('ggplot')


class Intensity(object):

    label_dict = {'q': r'q (A$^{-1}$)',
                  '2theta': r'$2\theta$',
                  'energy': 'Energy (keV)'}

    def __init__(self):
        self.energy, self.two_theta = None
        self.q_range = None
        self.q0, self.hkl = None, None
        self.a, self.sigma = None, None
        self.intensity = None
        self.sigma_q, self.flux_q = None, None
        self.method = None

    def add_peaks(self, material, b=1, weight=1):
        """
        Find peaks and relative intensities based on material and Debye-Waller
        or B factor (generally between 0.5 and 1.5 in inorganic materials).
        Deduce associated fit parameters for gaussian intensity distribution.
        """
        # Find peaks and multiplicity
        q0, m, hkl = peak_details(np.max(self.q_range), material)
        self.q0[material], self.hkl[material] = q0, hkl

        # Intensity factors
        method = self.method
        i_lp = lp_factor(q_to_2th(q0, self.energy)) if method == 'mono' else 1
        i_sf = scattering_factor(material, q0)  # consider adding complex
        i_tf = temperature_factor(q0, b)
        flux = self.flux_q(q0) if method == 'energy' else 1
        integrated_intensity = m * i_sf * i_tf * i_lp * flux * weight

        # Gaussian fit parameters
        sigma = self.sigma_q(q0)
        self.sigma[material] = sigma
        peak_height = integrated_intensity / (sigma * (2 * np.pi) ** 0.5)
        self.a[material] = peak_height

        # Intensity profiles wrt. q
        i = np.zeros_like(self.q_range)
        for idx, q in enumerate(q0):
            i += peak_height[idx] * np.exp(-(self.q_range - q) ** 2 /
                                            (2 * sigma[idx] ** 2))
        self.intensity['total'] += i
        self.intensity[material] = i

    def plot_intensity(self, x_axis='q', plot_type='all'):
        """
        Plot normalised intensity against 'q' or 'energy' / '2theta'.

        It there are multiple materials then 'separate', 'total'
        or 'all' intensity profiles can be visualized.
        """
        # Select label and conversion options
        x_label = self.label_dict[x_axis]
        error = "Can't plot wrt. {} in {} mode".format(x_axis, self.method)
        if self.method == 'mono':
            assert x_axis != 'energy', error
            convert = lambda x: q_to_2th(x, self.energy) * 180 / np.pi
        else:
            assert x_axis != '2theta', error
            convert = lambda x: q_to_e(x, self.two_theta)

        x = self.q_range if x_axis == 'q' else convert(self.q_range)
        i_max = np.max(self.intensity['total'])

        if plot_type != 'total':
            for mat in self.q0:
                q0, hkl, a = self.q0[mat], self.hkl[mat], self.a[mat]
                intensity = self.intensity[mat]
                plt.plot(x, intensity / i_max, '-+', label=mat)

                for idx, q in enumerate(q0):
                    q_ = q if x_axis == 'q' else convert(q)
                    plt.annotate(hkl[idx], xy=(q_, a[idx] / i_max),
                                 xytext=(0, 0), textcoords='offset points',
                                 ha='center', va='bottom')

            if len(self.q0) > 1 and plot_type in ['total', 'all']:
                intensity = self.intensity['total']

        plt.plot(x, intensity / i_max, 'k:', label='total')
        legend = plt.legend()
        legend.get_frame().set_color('white')
        plt.xlabel(x_label)
        plt.ylabel('Intensity')
        plt.show()


class MonochromaticIntensity(Intensity):
    def __init__(self, detector_shape=(2000, 2000), pixel_size=0.2,
                 sample_to_detector=1000, energy=100, delta_energy=0.5):

        self.energy = energy
        self.sigma_q = lambda q: e_to_q(delta_energy, q_to_2th(q, self.energy))
        self.q_range = extract_q(energy, detector_shape, pixel_size,
                                 sample_to_detector)
        self.two_theta = q_to_2th(self.q_range, self.energy)
        self.q0, self.hkl = {}, {}
        self.a, self.sigma = {}, {}
        self.intensity = {'total': np.zeros_like(self.q_range)}
        self.method = 'mono'


class EnergyDispersiveIntensity(Intensity):

    def __init__(self, two_theta=np.pi*(1/36), beamline='i12'):
        self.two_theta = two_theta
        self.energy = edxd_info[beamline]['energy']

        e_res = edxd_info[beamline]['res_e']
        delta_e = [e * d for e, d in zip(e_res['energy'], e_res['delta'])]
        q_res = [[e_to_q(e, two_theta) for e in e_res['energy']],
                 [e_to_q(d, two_theta) for d in delta_e]]
        self.sigma_q = InterpolatedUnivariateSpline(*q_res, k=1, ext=3)

        q = e_to_q(self.energy, two_theta)
        self.flux_q = interp1d(q, edxd_info[beamline]['flux'])
        self.q_range = np.linspace(0, np.max(q), edxd_info[beamline]['bins'])
        self.q0, self.q_res, self.hkl = {}, {}, {}
        self.a, self.sigma = {}, {}
        self.intensity = {'total': np.zeros_like(self.q_range)}
        self.method = 'edxd'


if __name__ == '__main__':
    test = MonochromaticIntensity(energy=100, delta_energy=2)
    test.add_peaks('Fe', weight=1)
    test.add_peaks('Cu')
    test.plot_intensity(x_axis='2theta', plot_type='all')
