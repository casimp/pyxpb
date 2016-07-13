from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd

df = pd.read_csv(r'data/form_factor.csv', index_col=0)


def scattering_factor(element, q):
    """
    Atomic scattering factor as a function of q for a monatomic material.
    The values are calculated according to the 9-parameter equation produced
    by Cromer and Mann.
    """
    try:
        a = [df.loc[element][i] for i in ['a0', 'a1', 'a2', 'a3']]
        b = [df.loc[element][i] for i in ['b0', 'b1', 'b2', 'b3']]
        c = df.loc[element]['c']
    except KeyError:
        print('Invalid element selection - '
              'valid options are as follows:{}'.format(df.index))
        raise

    i_sf = np.sum([a[i] * np.exp(-b[i] * (q/(4*np.pi))**2)
                   for i in range(4)], axis=0) + c
    return i_sf


def scattering_factor_complex(elements, q):
    """
    Atomic scattering for polyvalent materials.
    * More difficult to implement *
    """
    pass


def temperature_factor(q, b=1):
    """
    Thermal scattering as related to the Debye-Waller or B factor and q (A-1).
    B is typically in the range 0.5-1.5 for inorganic materials.
    """
    i_tf = np.exp(-b * (q / (4 * np.pi)) ** 2)
    return i_tf


def lp_factor(two_theta):
    """
    The combined lorentz and polarization factors, which depend on on the
    diffracted angle (2theta).
    """
    theta = two_theta
    lorentz = 1 / (4 * np.cos(theta) * np.sin(theta) ** 2)
    polarization = (1 + np.cos(2 * theta)**2) / 2
    i_lp = lorentz * polarization
    return i_lp


if __name__ == '__main__':
    scattering_factor('ds', 2)