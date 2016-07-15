from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np


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
