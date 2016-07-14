from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from multiplicity import peak_details
from beamline_details import edxd_info
from conversions import e_to_q, q_to_e, q_to_tth, tth_to_q
from intensity_factors import scattering_factor, temperature_factor, lp_factor
from detectors import Detector

plt.style.use('ggplot')


def strained_gaussians(x, *p):
    """
    Guassian curve fit for diffraction data.

    #   Peak height above background : p[0]
    #   Central value                : p[1]
    #   Standard deviation           : p[2]
    #   Strain                       : p[3]
    """
    img = np.zeros_like(x)
    for i in range(p[0].size):
        q0 = p[1][i] + p[1][i] * p[3]
        img = img + p[0][i] * np.exp(- (x - q0)**2 / (2. * p[2][i]**2))
    return img


def strain_trans(e_xx, e_yy, e_xy, phi):
    e_xx_1 = ((((e_xx + e_yy) / 2) + ((e_xx - e_yy) * np.cos(2 * phi) / 2)) +
              (e_xy * np.sin(2 * phi)))
    return e_xx_1


def relative_peak_heights(q0, a, sigma):
    # Flatten all dicts - extracting all Gaussian parameters
    # print(q0)
    q0_conc = np.concatenate([q0[i] for i in q0])
    a_conc = np.concatenate([a[i] for i in a])
    sigma_conc = np.concatenate([sigma[i] for i in sigma])
    a_max = np.max(strained_gaussians(q0_conc, a_conc, q0_conc, sigma_conc, 0))
    # print(q0)
    a_rel = {}
    for material in a:
        print(q0)
        q0_ = q0[material]
        a_ = a[material]
        sigma_ = sigma[material]
        a_rel[material] = strained_gaussians(q0_, a_, q0_, sigma_, 0) / a_max

    return a_rel

class Intensity(object):

    label_dict = {'q': r'q (A$^{-1}$)',
                  '2theta': r'$2\theta$',
                  'energy': 'Energy (keV)'}

    def __init__(self):
        self.energy, self.two_theta = None
        self.q_range = None
        self.q0, self.hkl = None, None
        self.a, self.sigma = None, None
        self.sigma_q, self.flux_q = None, None
        self.method = None

    def add_peaks(self, material, b=1, weight=1):
        """
        Find peaks and relative intensities based on material and Debye-Waller
        or B factor (generally between 0.5 and 1.5 in inorganic materials).
        Deduce associated fit parameters for gaussian intensity distribution.
        """
        # Store material, b factor and weight for recalculation
        self.materials[material] = {'b': b, 'weight': weight}

        # Find peaks and multiplicity
        q0, m, hkl = peak_details(np.max(self.q_range), material)
        # print([(i, j) for i, j in zip(q0, hkl)])
        self.q0[material], self.hkl[material] = q0, hkl

        # Intensity factors
        method = self.method
        i_lp = lp_factor(q_to_tth(q0, self.energy)) if method == 'mono' else 1
        i_sf = scattering_factor(material, q0)  # consider adding complex
        i_tf = temperature_factor(q0, b)
        flux = self.flux_q(q0) if method == 'energy' else 1
        integrated_intensity = m * i_sf * i_tf * i_lp * flux * weight

        # Gaussian fit parameters
        sigma = self.sigma_q(q0)
        self.sigma[material] = sigma
        peak_height = integrated_intensity / (sigma * (2 * np.pi) ** 0.5)
        self.a[material] = peak_height

    def intensity(self, x_axis='q', background=0.01,
                  e_xx=0, e_yy=0, e_xy=0, phi=0):
        """
        Returns normalised intensity against 'q' or 'energy' / '2theta'.

        It there are multiple materials then 'separate', 'total'
        or 'all' intensity profiles can be returned.
        """
        # Select label and convert x if selected
        valid = ['q'] + ['2theta' if self.method == 'mono' else 'energy']
        error = "Can't plot wrt. {} in {} mode".format(x_axis, self.method)
        assert x_axis in valid, error

        # Calculate the normal strain wrt. phi for each pixel
        strain = strain_trans(e_xx, e_yy, e_xy, phi)
        i = {}
        for mat in self.q0:
            q0, a, sigma = self.q0[mat], self.a[mat],  self.sigma[mat]
            i[mat] = strained_gaussians(self.q_range, a, q0, sigma, strain)

        i_total = np.sum([i[mat] for mat in i], axis=0)
        background *= np.random.rand(*i_total.shape) * np.max(i_total)
        i['total'] = i_total + background

        for material in i:
            i[material] /= np.max(i_total + background)

        x = self.q_range if x_axis == 'q' else self.convert(self.q_range)
        return x, i

    def plot_intensity(self, x_axis='q', plot_type='all', exclude_labels=0.02,
                       background=0.02, e_xx=0, e_yy=0, e_xy=0, phi=0):



        x, i = self.intensity(x_axis, background, e_xx, e_yy, e_xy, phi)
        i_total = i.pop('total')
        i_total_noiseless = np.sum([i[mat] for mat in i], axis=0)
        noise_factor =  np.max(i_total) / np.max(i_total_noiseless)
        print(noise_factor)


        a_rel = relative_peak_heights(self.q0, self.a, self.sigma)
        strain = strain_trans(e_xx, e_yy, e_xy, phi)
        if plot_type == 'all' or plot_type == 'separate':
            for material in i:
                q0, hkl = self.q0[material], self.hkl[material]
                x0 = q0 if x_axis == 'q' else self.convert(q0)
                plt.plot(x, i[material], '-', label=material)
                for idx, a_ in enumerate(a_rel[material]):

                    if a_ > exclude_labels:
                        x_ = x0[idx] * (1 + strain)
                        a_h = 0.005 + a_ / noise_factor# - a_ * background / 2
                        plt.annotate(hkl[idx], xy=(x_, a_h),
                                     xytext=(0, 0), textcoords='offset points',
                                     ha='center', va='bottom')

        if plot_type in ['all', 'total'] and len(i) > 1:
            plt.plot(x, i_total, 'k:', label='total')

        legend = plt.legend()
        legend.get_frame().set_color('white')
        plt.xlabel(self.label_dict[x_axis])
        plt.ylabel('Relative Intensity')
        plt.ylim([0, 1.05])
        plt.show()


class MonochromaticIntensity(Intensity):
    def __init__(self, detector_shape=(2000, 2000), pixel_size=0.2,
                 sample_to_detector=1000, energy=100, delta_energy=0.5):

        self.detector = Detector(detector_shape, pixel_size)
        self.detector.setup(energy, sample_to_detector)
        r_max, q_max = np.max(self.detector.r), np.max(self.detector.q)

        self.energy = energy
        self.sigma_q = lambda q: e_to_q(delta_energy, q_to_tth(q, self.energy))
        self.q_range = np.linspace(0, q_max, int(r_max))
        self.two_theta = q_to_tth(self.q_range, self.energy)
        self.q0, self.hkl = {}, {}
        self.a, self.sigma = {}, {}
        self.materials = {}
        self.convert = lambda x: q_to_tth(x, self.energy) * 180 / np.pi
        self.method = 'mono'

    def new_setup(self, energy, sample_detector):
        self.detector.setup(energy, sample_detector)
        r_max, q_max = np.max(self.detector.r), np.max(self.detector.q)
        self.q_range = np.linspace(0, q_max, int(r_max))

        for material in self.materials:
            b = self.materials[material]['b']
            weight = self.materials[material]['weight']
            self.add_peaks(material, b, weight)

    def rings(self, e_xx=0, e_yy=0, e_xy=0, exclude_criteria=0.01, crop=0,
              background=0.02):
        """
        Returns a 2D numpy array (image) containing the Debye-Scherrer rings
        for the previously specified detector/sample setup. Specify the
        strain tensor (e_xx, e_yy, e_xy) for strained diffraction rings.

        Can be computationally expensive for large detectors sizes
        (i.e. > 1000 x 1000) - there are therefore options to remove weak
        reflections (recommended at 0.01) and to crop the detector.

        Returned array is scaled according to the maximum peak height.
        """
        # Extract data from detector object
        crop = [int(crop * i / 2) for i in self.detector.shape]
        crop = [None, None] if crop[0] == 0 else crop
        phi = self.detector.phi[crop[0]:-crop[0], crop[1]:-crop[1]]
        q = self.detector.q[crop[0]:-crop[0], crop[1]:-crop[1]]

       # Calculate the normal strain wrt. phi for each pixel
        strain = strain_trans(e_xx, e_yy, e_xy, phi)

        # Flatten all dicts - extracting all Gaussian parameters
        q0 = np.concatenate([self.q0[i] for i in self.q0])
        a = np.concatenate([self.a[i] for i in self.a])
        sigma = np.concatenate([self.sigma[i] for i in self.sigma])

        # Exclude peaks based on rel. height and add strained Gaussian curves
        a_max = np.max(strained_gaussians(q0, a, q0, sigma, 0))
        ex = [a > exclude_criteria * np.max(a_max)]
        img = strained_gaussians(q, a[ex], q0[ex], sigma[ex], strain)

        # Normalise peak intensity and add background noise
        img /= np.max(a_max)
        img += np.random.rand(*phi.shape) * background

        return img / np.nanmax(img) # rescale


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
        self.materials = {}
        self.intensity = {'total': np.zeros_like(self.q_range)}
        self.convert = lambda x: q_to_e(x, self.two_theta)
        self.method = 'edxd'


if __name__ == '__main__':
    test = MonochromaticIntensity(detector_shape=(2000,2000), pixel_size=.2,
                                  energy=100, delta_energy=1,
                                  sample_to_detector=1000)
    test.add_peaks('Al', weight=2)
    test.add_peaks('Fe')
    # print(test.q0)
    test.plot_intensity(x_axis='2theta', exclude_labels=0.05, background=0.01)
    # plt.imshow(test.rings(0.2, 0.1, 0.05, exclude_criteria=0, crop=0.3))
    # plt.show()
    # test.new_setup(125, 500)
    # plt.imshow(test.rings(0.2, 0.1, 0.05, exclude_criteria=0.01, crop=0.3))
    # print(test.q0)
    #test.plot_intensity(x_axis='2theta')
    #cProfile.run('test.rings(0.2, 0.1, 0.05, exclude_criteria=0.01, crop=0.3)')
    #cProfile.run('test.rings(0.2, 0.1, 0.05, exclude_criteria=0, crop=0)')
    print(np.nanmax(test.rings(0.2, 0.1, 0.05, exclude_criteria=0.01, crop=0.3)))
    plt.show()