from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from .conversions import tth_to_q


class BaseDetector(object):
    def __init__(self, shape, pixel_size=0.2):
        y, x = np.ogrid[:float(shape[0]), :float(shape[1])]
        x, y = x - shape[0] / 2, y - shape[1] / 2
        self.r = (x ** 2 + y ** 2) ** .5
        self.shape = shape
        self.pixel_size = pixel_size
        self.phi = np.arctan(y / x)
        self.energy = None
        self.sample_detector = None

    def setup(self, energy, sample_detector):
        self.sample_detector = sample_detector
        self.energy = energy
        self.two_theta = np.arctan(self.r * self.pixel_size / sample_detector)
        self.q = tth_to_q(self.two_theta, energy)




