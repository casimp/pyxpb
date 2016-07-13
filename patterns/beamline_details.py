from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

i12_e_flux = np.loadtxt('data/i12_energy_distribution.csv', delimiter=',')

i12 = {'energy': i12_e_flux[:, 0],
       'flux': i12_e_flux[:, 1],
       'res_e': {'energy': [50, 150],
                 'delta': [0.5 * 0.007, 0.5 * 0.004]},
       'bins': 4000}

