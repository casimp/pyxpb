from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from conversions import e_to_w


def extract_q(energy, detector_shape=(2000, 2000), pixel_size=0.2,
              sample_to_detector=1000):

    n_steps = ((detector_shape[0] / 2) ** 2 +
               (detector_shape[1] / 2) ** 2) ** 0.5

    r_max = n_steps * pixel_size

    theta_max = np.arctan(r_max / sample_to_detector)
    q_max = (4 * np.pi * np.sin(theta_max)) / e_to_w(energy)
    return np.linspace(0, q_max, n_steps) / (10 ** 10)


