from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mock import patch
import numpy as np

from pyxpb.detectors import i12_energy, MonoDetector
from pyxpb.array_create import ring_array, intensity_array, gauss2d_tensor

i12 = i12_energy()
mono = MonoDetector(shape=(2000, 2000), pixel_size=0.2,
                    sample_detector=700, energy=100,
                    energy_sigma=0.5)

i12.add_peaks('Fe')
mono.add_peaks('Fe')


def test_ring():
    r_array = ring_array(mono, pnts=(3, 3), xlim=(-1, 1), ylim=(-1, 1),
                         background=0, exclude=0.05, strain_func=gauss2d_tensor)

def test_intensity():
    i_array = intensity_array(i12, pnts=(3, 3), xlim=(-1, 1), ylim=(-1, 1),
                              background=0, strain_func=gauss2d_tensor)