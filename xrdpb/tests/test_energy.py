from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mock import patch
import numpy as np

from xrdpb.detectors import i12_energy

i12_energy.add_peaks('Fe')


def test_intensity_profiles():
    phi = np.pi / 3.2
    profile_0 = i12_energy.intensity(phi, background=0, x_axis='q',
                             strain_tensor=(0.2, 0.2, 0.3))
    profile_180 = i12_energy.intensity(phi-np.pi, background=0, x_axis='q',
                               strain_tensor=(0.2, 0.2, 0.3))
    assert np.allclose(profile_0[1]['Fe'], profile_180[1]['Fe'])

    # Test a few combinations of parameters
    i12_energy.intensity(0, background=0, x_axis='energy',
                   strain_tensor=(0.2, 0.2, 0.3))

    i12_energy.intensity([0, np.pi/7], background=0, x_axis='energy',
                   strain_tensor=(0., 0., 0.))


def test_all_intensity():
    # Should return data for all detectors in array
    error = i12_energy.intensity()[1]['Fe'].shape
    assert i12_energy.intensity()[1]['Fe'].shape == (23, 4000), error


@patch("matplotlib.pyplot.show")
def test_plot_intensity(mock_show):
    mock_show.return_value = None
    i12_energy.plot_intensity()
    i12_energy.intensity_factors('Fe', plot=True)


if __name__ == '__main__':
    test_plot_intensity()
