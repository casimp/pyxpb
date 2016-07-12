from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

folder = os.path.expanduser('~/Dropbox/Projects/diffraction/')
lattice_params = os.path.join(folder, 'lattice_params.csv')
miller_info = os.path.join(folder, 'miller_info.csv')

df_latt = pd.read_csv(lattice_params, index_col=0)
df_miller = pd.read_csv(miller_info)


def q0_multiplicity(compound, q_max=10):

    a = df_latt.loc[compound]['a']
    structure = df_latt.loc[compound]['structure']
    structure = 'simple' if structure == 'Simple Cubic' else structure

    if structure.lower() not in ['fcc', 'bcc', 'simple']:
        print('Does not work on lattices other than fcc/bcc/simple cubic')
        return None

    N_max = (a * q_max / (2 * np.pi))**2
    hkl = df_miller[(df_miller[structure.lower()]) & (df_miller['N'] < N_max)]

    q0 = 2 * np.pi * hkl['N'].values**0.5 / a
    miller_name = hkl['h'].map(str) + hkl['k'].map(str) + hkl['l'].map(str)
    return q0, hkl['M'].values, miller_name.values