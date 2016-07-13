from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from conversions import e_to_w, w_to_e, e_to_q, q_to_e, q_to_2th
from intensity_factors import scattering_factor, scattering_factor_complex, temperature_factor, lorentz_polarization_factor
from multiplicity import q0_multiplicity

plt.style.use('ggplot')

i12_dist = os.path.join('data', 'i12_energy_distribution.csv')
i12_energy_dist = np.loadtxt(i12_dist, delimiter=',')


edxd_info = {'i12': {'energy': i12_energy_dist[:, 0],
                     'flux': i12_energy_dist[:, 1],
                     'res_e': {'energy': [50, 150],
                               'delta': [0.5 * 7 * 10 ** -3,
                                         0.5 * 4 * 10 ** -3]},
                     'bins': 4000}}


def extract_q(energy_keV, detector_shape=(2000, 2000), pixel_size=0.2,
                    sample_to_detector=1000):

    n_steps = ((0.5 * detector_shape[0]) ** 2 +
               (0.5 * detector_shape[1]) ** 2) ** 0.5

    r_max = n_steps * pixel_size

    theta_max = np.arctan(r_max / sample_to_detector)
    q_max = (4 * np.pi * np.sin(theta_max)) / e_to_w(energy_keV)
    return np.linspace(0, q_max, n_steps) / (10**10)


class MonochromaticIntensity(object):
    def __init__(self, detector_shape=(2000, 2000), pixel_size=0.2,
                 sample_to_detector=1000, energy=100, delta_energy=0.5):

        self.energy = energy
        self.delta_energy = delta_energy
        self.q_range = extract_q(energy, detector_shape, pixel_size,
                                  sample_to_detector)
        self.q_max = np.max(self.q_range)
        self.q0, self.hkl, = {}, {}
        self.a, self.sigma = {}, {}
        self.intensity = {'total': np.zeros_like(self.q_range)}

    def add_peaks(self, material, elements=None, B=1, weight=1):

        q0, M, hkl = q0_multiplicity(material, self.q_max)
        self.q0[material] = q0
        self.hkl[material] = hkl

        if elements is not None:
            i_sf = scattering_factor_complex(elements, self.q0)
        else:
            i_sf = scattering_factor(material, q0)
        i_tf = temperature_factor(q0, B)
        i_lp = lorentz_polarization_factor(q_to_2th(q0, self.energy))
        integrated_intensity = M * i_sf * i_tf * i_lp * weight

        sigma = e_to_q(self.delta_energy, q_to_2th(q0, self.energy))
        self.sigma[material] = sigma
        peak_height = integrated_intensity / (sigma * (2 * np.pi) ** 0.5)
        self.a[material] = peak_height

        i = np.zeros_like(self.q_range)
        for idx, q in enumerate(q0):
            i += peak_height[idx] * np.exp(-(self.q_range - q) ** 2 /
                                            (2 * sigma[idx] ** 2))
        self.intensity['total'] += i
        self.intensity[material] = i

    def plot_intensity(self, x_axis='q', plot_type='both'):

        two_theta = q_to_2th(self.q_range, self.energy) * 180 / np.pi
        x = self.q_range if x_axis == 'q' else two_theta
        i_max = np.max(self.intensity['total'])
        x_label = r'q (A$^{-1}$)' if x_axis == 'q' else r'$2\theta$'

        if plot_type != 'total':
            for material in self.q0:
                # Load data
                q0 = self.q0[material]
                hkl = self.hkl[material]
                a = self.a[material]
                intensity = self.intensity[material]

                if plot_type == 'individual':
                    plt.figure()
                plt.plot(x, intensity/i_max, '-+', label=material)
                legend = plt.legend()
                legend.get_frame().set_color('white')
                plt.xlabel(x_label)
                plt.ylabel('Intensity')

                for idx, q in enumerate(q0):

                    theta = q_to_2th(q, self.energy) * 180 / np.pi
                    q_ = q if x_axis == 'q' else theta
                    plt.annotate(hkl[idx], xy=(q_, a[idx]/i_max),
                                 xytext=(0, 0), textcoords='offset points',
                                 ha='center', va='bottom')

            plot_total = any([plot_type == 'total', plot_type == 'all'])
            if len(self.q0) > 1 and plot_total:
                intensity = self.intensity['total']
                plt.plot(x, intensity / i_max, 'k:', label='total')
                legend = plt.legend()
                legend.get_frame().set_color('white')
                plt.xlabel(x_label)
                plt.ylabel('Intensity')

        plt.show()



class EnergyDispersiveIntensity(object):
    def __init__(self, two_theta=np.pi*(1/36), beamline='i12'):

        self.two_theta = two_theta
        self.energy = edxd_info[beamline]['energy']
        self.flux = edxd_info[beamline]['flux']
        bins = edxd_info[beamline]['bins']

        e_res = edxd_info[beamline]['res_e']
        delta_e = [e * d for e, d in zip(e_res['energy'], e_res['delta'])]
        q_res = [[e_to_q(e, two_theta) for e in e_res['energy']],
                 [e_to_q(d, two_theta) for d in delta_e]]
        self.f_q_res = InterpolatedUnivariateSpline(*q_res, k=1, ext=3)

        self.q = e_to_q(self.energy, two_theta)
        self.f_flux = interp1d(self.q, self.flux)
        self.q_max = np.max(self.q)
        self.q_range = np.linspace(0, self.q_max, bins)
        self.q0, self.q_res, self.hkl = {}, {}, {}
        self.a, self.sigma = {}, {}
        self.intensity = {'total': np.zeros_like(self.q_range)}

    def add_peaks(self, material, elements=None, B=1, weight=1):

        q0, M, hkl = q0_multiplicity(material, self.q_max)
        self.q0[material] = q0
        self.q_res[material] = self.f_q_res(q0)
        self.hkl[material] = hkl

        if elements is not None:
            i_sf = scattering_factor_complex(elements, self.q0)
        else:
            i_sf = scattering_factor(material, q0)
        i_tf = temperature_factor(q0, B)
        flux = self.f_flux(q0)
        integrated_intensity = M * i_sf * i_tf * flux * weight

        sigma = self.f_q_res(q0)
        self.sigma[material] = sigma
        peak_height = integrated_intensity / (sigma * (2 * np.pi) ** 0.5)
        self.a[material] = peak_height

        i = np.zeros_like(self.q_range)
        for idx, q in enumerate(q0):
            i += peak_height[idx] * np.exp(-(self.q_range - q) ** 2 /
                                            (2 * sigma[idx] ** 2))
        self.intensity['total'] += i
        self.intensity[material] = i

    def plot_intensity(self, x_axis='q', plot_type='both'):

        e_range = q_to_e(self.q_range, self.two_theta)
        x = self.q_range if x_axis == 'q' else e_range
        i_max = np.max(self.intensity['total'])
        unit = 'keV' if x_axis == 'energy' else r'A$^{-1}$'

        if plot_type != 'total':
            for material in self.q0:
                # Load data
                q0 = self.q0[material]
                hkl = self.hkl[material]
                a = self.a[material]
                intensity = self.intensity[material]

                if plot_type == 'individual':
                    plt.figure()
                plt.plot(x, intensity/i_max, '-+', label=material)
                legend = plt.legend()
                legend.get_frame().set_color('white')
                plt.xlabel('{} ({})'.format(x_axis, unit))
                plt.ylabel('Intensity')

                for idx, q in enumerate(q0):

                    q_ = q if x_axis == 'q' else q_to_e(q, self.two_theta)
                    plt.annotate(hkl[idx], xy=(q_, a[idx]/i_max),
                                 xytext=(0, 0), textcoords='offset points',
                                 ha='center', va='bottom')

            plot_total = any([plot_type == 'total', plot_type =='all'])
            if len(self.q0) > 1 and plot_total:
                intensity = self.intensity['total']
                plt.plot(x, intensity / i_max, 'k:', label='total')
                legend = plt.legend()
                legend.get_frame().set_color('white')
                plt.xlabel('{} ({})'.format(x_axis, unit))
                plt.ylabel('Intensity')

        plt.show()

if __name__ == '__main__':
    test = EnergyDispersiveIntensity()#energy=100, delta_energy=2, sample_to_detector=2000)
    test.add_peaks('Fe', weight=1)
    test.add_peaks('Cu')
    test.plot_intensity(x_axis='q', plot_type='all')
