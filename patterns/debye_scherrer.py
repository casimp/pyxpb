from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d


def strained_rings(detector='default', rings=range(50, 250, 50),
                   intensity=None, e_xx=0, e_yy=0, e_xy=0):


    if detector == 'default':
        shape = (250, 250)
        xc, yc = shape[0] // 2, shape[1] // 2

    y, x = np.ogrid[:shape[0], :shape[1]]  # meshgrid without linspace
    y -= yc
    x -= xc
    r = np.sqrt(x ** 2 + y ** 2)  # simple pythagoras - radius of each pixel
    theta = np.cos(x / r)  # what angle are these pixels at

    e_xx_1 = strain_trans(e_xx, e_yy, e_xy, theta)

    img = np.zeros(shape)
    for idx, radius in enumerate(rings):
        rel_intensity = 1 if intensity is None else intensity[idx]
        radius += e_xx_1 * radius
        img += np.exp(-(
                       r - radius) ** 2) * rel_intensity  # recentre radius and apply gaussian

    return img