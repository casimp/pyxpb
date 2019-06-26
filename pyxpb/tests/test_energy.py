from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mock import patch
import numpy as np
import matplotlib.pyplot as plt

from pyxpb.detectors import i12_energy

i12 = i12_energy()
i12.add_material('Fe')


def test_intensity_profiles():
    phi = np.pi / 3.2
    profile_0 = i12.intensity(phi, background=0, x_axis='q',
                             strain_tensor=(0.2, 0.2, 0.3))
    profile_180 = i12.intensity(phi-np.pi, background=0, x_axis='q',
                               strain_tensor=(0.2, 0.2, 0.3))
    assert np.allclose(profile_0[1]['Fe'], profile_180[1]['Fe'])

    # Test a few combinations of parameters
    i12.intensity(0, background=0, x_axis='energy',
                   strain_tensor=(0.2, 0.2, 0.3))

    i12.intensity([0, np.pi/7], background=0, x_axis='energy',
                   strain_tensor=(0., 0., 0.))


def test_all_intensity():
    # Should return data for all detectors in array
    error = i12.intensity()[1]['Fe'].shape
    assert i12.intensity()[1]['Fe'].shape == (23, 4096), error


#@patch("matplotlib.pyplot.show")
def test_plot_intensity():#mock_show):
    #mock_show.return_value = None
    i12.define_background([0, 1, 2, 3, 4, 5, 6, 7],
                          [0, 2, 3, 3.5, 3.75, 3.5, 3, 2], k=2)
    i12.plot_intensity()
    plt.close()
    i12.intensity_factors('Fe', plot=True)
    plt.close()


if __name__ == '__main__':
    test_plot_intensity()
