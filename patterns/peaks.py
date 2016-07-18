from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np

from .multiplicity import peak_details
from .conversions import q_to_tth, q_to_e
from .intensity_factors import scattering_factor, temperature_factor, lp_factor

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
        q0 = p[1][i] * (1 + p[3]) # + p[1][i] * p[3]
        img = img + p[0][i] * np.exp(- (x - q0)**2 / (2. * p[2][i]**2))
    return img


def strain_trans(e_xx, e_yy, e_xy, phi):
    e_xx_1 = ((((e_xx + e_yy) / 2) + ((e_xx - e_yy) * np.cos(2 * phi) / 2)) +
              (e_xy * np.sin(2 * phi)))
    return e_xx_1


class Peaks(object):

    label_dict = {'q': r'q (A$^{-1}$)',
                  '2theta': r'$2\theta$',
                  'energy': 'Energy (keV)'}

    def __init__(self, r, q, energy, two_theta, phi, sigma_q, flux_q, method):

        self.method = method
        self.two_theta = two_theta
        self.phi = phi
        self.energy = energy
        self.q = q
        self.sigma_q = sigma_q
        self.flux_q = flux_q
        self.r = r

        # Empty dicts for storing peaks / materials
        self.a, self.sigma, self.q0 = {}, {}, {}
        self.materials, self.hkl = {}, {}

    def convert(self, q):
        if self.method == 'mono':
            return q_to_tth(self.q, self.energy) * 180 / np.pi
        else:
            return q_to_e(self.q, self.two_theta)

    def intensity_factors(self, material, b=1, q=None, plot=True, x_axis='q'):
        """ Calculates normalised intensity factors (with option for plotting).

        Finds intensity factors wrt. q based on material and diffraction setup.
        Either returns values or plots for visualisation.

        Args:
            material (str): Element symbol (compound formula)
            b (float): B factor
            q (np.ndarray): Values of q to calculate intensty factors at.
            plot (bool): Plot selector
            x_axis (str): Plot relative to 'q' or 'energy' / '2theta'

        Returns:
            tuple: Intensity factor components(i_lp, i_sf, i_tf, flux)
        """
        # Make decorator from this?
        valid = ['q'] + ['2theta' if self.method == 'mono' else 'energy']
        error = "Can't plot wrt. {} in {} mode".format(x_axis, self.method)
        assert x_axis in valid, error

        method = self.method

        if q is None and method == 'mono':
            q = np.linspace(0, self.q.max(), self.r.max())
        else:
            q = self.q if q is None else q
        # Intensity factors

        i_lp = lp_factor(q_to_tth(q, self.energy)) if method == 'mono' else 1
        i_sf = scattering_factor(material, q)  # consider adding complex
        i_tf = temperature_factor(q, b)
        flux = self.flux_q(q) if method == 'edxd' else 1

        if plot:
            ind = np.argsort(q)[q > 2]
            q = q if x_axis == 'q' else self.convert(q)
            labels = ['flux', 'lorentz', 'scatter', 'temp']
            for i_f, label in zip([flux, i_lp, i_sf, i_tf], labels):
                if isinstance(i_f, int) or len(i_f) == 1:
                    i_f *= np.ones_like(q)
                plt.plot(q[ind], i_f[ind] / i_f[ind].max(), '-', label=label)
            total = i_sf ** 2 * i_lp * i_tf * flux
            total = total[ind]
            plt.plot(q[ind], total / total.max(), 'k-.', label='total')
            plt.ylim([0, 1.05])
            legend = plt.legend()
            legend.get_frame().set_color('white')
            plt.ylabel('Relative Intensity Factor')
            plt.xlabel(self.label_dict[x_axis])
            plt.show()
        else:
            return i_lp, i_sf, i_tf, flux

    def add_peaks(self, material, b=1., weight=1.):
        """ Add peak locations and intensities for a given material.

        Find peaks and relative intensities based on material and Debye-Waller
        or B factor (generally between 0.5 and 1.5 in inorganic materials).
        Deduce associated fit parameters for gaussian intensity distribution.

        Args:
            material (str): Element symbol (compound formula)
            b (float): B factor
            weight (float): Relative peak weight (useful for mixtures/phases)
        """
        # Store material, b factor and weight for recalculation
        self.materials[material] = {'b': b, 'weight': weight}

        # Find peaks and multiplicity
        q0, m, hkl = peak_details(np.max(self.q), material)
        self.q0[material], self.hkl[material] = q0, hkl

        # Intensity factors
        i_lp, i_sf, i_tf, flux = self.intensity_factors(material, b, q0, False)
        integrated_intensity = i_sf * m * i_tf * i_lp * flux * weight

        # Gaussian fit parameters
        sigma = self.sigma_q(q0)
        self.sigma[material] = sigma
        peak_height = integrated_intensity / (sigma * (2 * np.pi) ** 0.5)
        self.a[material] = peak_height

    def relative_heights(self):
        """ Computes relative peak heights wrt. total intensity profile.

        Returns:
            dict: Relative peak heights for each material
        """
        q0 = np.concatenate([self.q0[i] for i in self.q0])
        a = np.concatenate([self.a[i] for i in self.a])
        sigma = np.concatenate([self.sigma[i] for i in self.sigma])
        a_max = np.max(strained_gaussians(q0, a, q0, sigma, 0))
        a_rel = {}
        for mat in self.a:
            q0_, a_, sigma_ = self.q0[mat], self.a[mat], self.sigma[mat]
            a_rel[mat] = strained_gaussians(q0_, a_, q0_, sigma_, 0) / a_max
        return a_rel

    def intensity(self, phi=None, x_axis='q', background=0.01,
                  strain_tensor=(0, 0, 0)):
        """ Computes normalised intensity against 'q' or 'energy' / '2theta'.

        Specify the strain tensor (e_xx, e_yy, e_xy) for strained diffraction
        peaks at a given angle (phi).

        Args:
            phi (float, list, np.ndarray): Azimuthal angle (rad)
            x_axis (str): Plot relative to 'q' or 'energy' / '2theta'
            background (float): Relative background noise
            strain_tensor (tuple): Strain tensor components (e_xx, e_yy, e_xy)

        Returns:
            x (np.ndarray): 1D positional array ('q' or 'energy' / '2theta')
            intensity (dict): Material specific + total rel. intensity profiles
        """
        # Select label and convert x if selected
        valid = ['q'] + ['2theta' if self.method == 'mono' else 'energy']
        error = "Can't plot wrt. {} in {} mode".format(x_axis, self.method)
        assert x_axis in valid, error

        # If no phi is defined, phi is set to self.phi for edxd, else error
        if phi is None:
            error = 'Must define azimuthal angle(s), phi, for mono detectors.'
            assert self.method != 'mono', error
            phi = self.phi

        # To allow for list of phi values, must be column vector
        if np.array(phi).ndim < 2:
            phi = np.expand_dims(phi, 1)

        # Calculate the normal strain wrt. phi for each pixel
        e_xx, e_yy, e_xy = strain_tensor
        strain = strain_trans(e_xx, e_yy, e_xy, phi)
        i = {}
        for mat in self.q0:
            q0, a, sigma = self.q0[mat], self.a[mat],  self.sigma[mat]
            # q = np.repeat(self.q, np.array(strain).size, axis=0) ##!!!??
            i[mat] = strained_gaussians(self.q, a, q0, sigma, strain)

        i_total = np.sum([i[mat] for mat in i], axis=0)  # OK?
        background *= np.random.rand(*i_total.shape) * np.max(i_total)
        i['total'] = i_total + background

        for material in i:
            i[material] /= np.max(i_total + background)

        x = self.q if x_axis == 'q' else self.convert(self.q)
        return x, i

    def plot_intensity(self, phi=0, x_axis='q', background=0.02,
                       strain_tensor=(0., 0., 0.), plot_type='all',
                       exclude_labels=0.02):
        """ Plot normalised intensities against 'q' or 'energy' / '2theta'.

        Specify the strain tensor (e_xx, e_yy, e_xy) for strained diffraction
        peaks at a given angle (phi). If there are multiple materials there
        is the option to plot each material separately, or the total intensity
        or both of these. Background noise can be relative to max intensity.

        Note: Background noise not used for 'separate' plots.

        Args:
            phi (float): Azimuthal angle (rad)
            x_axis (str): Plot relative to 'q' or 'energy' / '2theta'
            background (float): Relative background noise
            strain_tensor (tuple): Strain tensor components (e_xx, e_yy, e_xy)
            plot_type (str): Plot 'separate' intensities, 'total' or 'both'.
            exclude_labels (float): Relative maxima for peak label exclusion
        """
        x, i = self.intensity(phi, x_axis, background, strain_tensor)
        i_total = i.pop('total')
        i_total_noiseless = np.sum([i[mat] for mat in i], axis=0)
        noise_factor = np.max(i_total) / np.max(i_total_noiseless)
        if plot_type == 'separate':
            i_max = np.max([i[mat] for mat in i])
            noise_factor = i_max / np.max(i_total_noiseless)
            for material in i:
                i[material] /= i_max

        a_rel = self.relative_heights()
        e_xx, e_yy, e_xy = strain_tensor
        strain = strain_trans(e_xx, e_yy, e_xy, phi)
        if plot_type == 'all' or plot_type == 'separate':
            for material in i:
                q0, hkl = self.q0[material], self.hkl[material]
                x0 = q0 if x_axis == 'q' else self.convert(q0)
                plt.plot(x, i[material], '-', label=material)
                for idx, a_ in enumerate(a_rel[material]):

                    if a_ > exclude_labels:
                        x_ = x0[idx] * (1 + strain)
                        a_h = 0.005 + a_ / noise_factor
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


class Rings(Peaks):

    def __init__(self, phi, q, a, q0, sigma):
        self.phi = phi
        self.q = q
        self.a = a
        self.q0 = q0
        self.sigma = sigma

    def rings(self, exclude_criteria=0.01, crop=0.0, background=0.02,
              strain_tensor=(0., 0., 0.)):
        """ Computes detector sized array containing Debye-Scherrer rings.

        Returns a 2D numpy array (image) containing the Debye-Scherrer rings
        for the previously specified detector/sample setup. Specify the
        strain tensor (e_xx, e_yy, e_xy) for strained diffraction rings.

        Can be computationally expensive for large detectors sizes
        (i.e. > 1000 x 1000) - there are therefore options to remove weak
        reflections (recommended at 0.01) and to crop the detector.

        Args:
            exclude_criteria (float): Relative peak maxima for peak exclusion
            crop (float): Crop fraction of detector dimensions
            background (float): Relative background noise
            strain_tensor (tuple): Strain tensor components (e_xx, e_yy, e_xy)

        Returns:
            np.ndarray: 2D array scaled according to the maximum peak height.
        """

        # Extract data from detector object
        crop = [int(crop * i / 2) for i in self.q.shape]
        crop = [None, None] if crop[0] == 0 else crop
        phi = self.phi[crop[0]:-crop[0], crop[1]:-crop[1]]
        q = self.q[crop[0]:-crop[0], crop[1]:-crop[1]]

        # Calculate the normal strain wrt. phi for each pixel
        e_xx, e_yy, e_xy = strain_tensor
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

        return img / np.nanmax(img)  # Rescale

    def plot_rings(self, exclude_criteria=0.01, crop=0., background=0.02,
                   strain_tensor=(0, 0, 0)):
        """ Plots an image array containing Debye-Scherrer rings.

        Plots the 2D numpy array (image) containing the Debye-Scherrer rings
        for the previously specified detector/sample setup. Specify the
        strain tensor (e_xx, e_yy, e_xy) for strained diffraction rings.

        Can be computationally expensive for large detectors sizes
        (i.e. > 1000 x 1000) - there are therefore options to remove weak
        reflections (recommended at 0.01) and to crop the detector.

        Args:
            exclude_criteria (float): Relative peak maxima for peak exclusion
            crop (float): Crop fraction of detector dimensions
            background (float): Relative background noise
            strain_tensor (tuple): Strain tensor components (e_xx, e_yy, e_xy)
        """
        img = self.rings(exclude_criteria, crop, background, strain_tensor)
        plt.imshow(img)
        plt.show()