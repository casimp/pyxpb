from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

folder = os.path.expanduser('~/Dropbox/Projects/diffraction/')
form_factor = os.path.join(folder, 'form_factor.csv')

df_form = pd.read_csv(form_factor, index_col=0)


def scattering_factor(element, q=None, plot=False):
    if q is None:
        q = np.linspace(0, 25, 1000)
    a = [df_form.loc[element][i] for i in ['a0', 'a1', 'a2', 'a3']]
    b = [df_form.loc[element][i] for i in ['b0', 'b1', 'b2', 'b3']]
    c = df_form.loc[element]['c']

    i_sf = np.sum([a[i] * np.exp(-b[i] * (q/(4*np.pi))**2)
                   for i in range(4)], axis=0) + c
    if plot:
        plt.plot(q, i_sf)
    return i_sf


def scattering_factor_complex(elements, q=None, plot=False):
    return None


def temperature_factor(q, B=1, plot=False):
    if q is None:
        q = np.linspace(0, 25, 1000)
    i_tf = np.exp(-B * (q / (4 * np.pi)) ** 2)
    if plot:
        plt.plot(q, i_tf)
    return i_tf


def lorentz_polarization_factor(two_theta=None, plot=False):
    if two_theta is None:
        two_theta = np.linspace(0, np.pi, 1000)
    theta = two_theta
    lorentz = 1 / (4 * np.cos(theta) * np.sin(theta) ** 2)
    polarization = (1 + np.cos(2 * theta)**2) / 2
    i_lp = lorentz * polarization
    if plot:
        plt.plot(two_theta, i_lp)
    return i_lp

