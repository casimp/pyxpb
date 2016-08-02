from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from itertools import product
import os

import numpy as np
import pandas as pd

fname = os.path.join(os.path.dirname(__file__), 'data/lattice_params.csv')
df_latt = pd.read_csv(fname, index_col=0)


def cubic_permutations(N):
    """ Returns all hkl permutations up to given N (h**2 + k**2 + l**2) """
    max_h = np.ceil(N ** 0.5).astype(int)
    hkl_perm = list(product(*[range(max_h + 1) for _ in range(3)]))
    hkl = [i for i in hkl_perm if (i[0] ** 2 + i[1] ** 2 + i[2] ** 2) <= N]
    return hkl


def hkl_type(h, k, l):
    """
    Returns the general hkl type (e.g. 'hh0') for a given set of miller
    indices. Valid for all Bravais lattices.
    """
    zero_count = sum(1 if i == 0 else 0 for i in [h, k, l])
    diff_count = len(set([h, k, l]))

    if diff_count == 3:
        if zero_count == 0:
            return 'hkl'
        else:
            return 'hk0' if not l else 'h0l' if not k else '0kl'
    elif diff_count == 2:
        if zero_count == 0:
            return 'hhl'
        elif zero_count == 1:
            return 'hh0' if h == k else '0kk'
        else:
            return 'h00' if h else '0k0' if k else '00l'
    else:
        return 'hhh'


def cubic_multiplicity(structure, N):
    """
    For a given cubic crystal structure (i.e. 'fcc', 'bcc', 'simple') and
    maximum value for N (h**2 + k**2 + l**2) returns list of valid hkl
    families and there associated multiplicity.

    Note that for equivalent reflections (e.g. (2,2,1) and (3,0,0)) the
    multiplicites are summed and returned according to lowest h
    (e.g. hkl = (2,2,1), M = 24 + 6)
    """
    # print(structure, N)
    factor = {'hkl': 48, 'hhl': 24, 'hh0': 12, '0kk': 12, 'hhh': 8, 'hk0': 24,
              'h0l': 24, '0kl': 24, 'h00': 6, '0k0': 6, '00l': 6}

    hkl_perm = cubic_permutations(N)

    # In cubic systems a=b=c, so to filter out reflections:
    hkl = list(set([tuple(sorted(n, reverse=True)) for n in hkl_perm]))

    # The hkl indices are correct for simple cubic but if bcc/fcc:
    if structure.lower() == 'bcc':
        # print(structure, N)
        hkl = [i for i in hkl if (i[0] + i[1] + i[2]) % 2 == 0]
    elif structure.lower() == 'fcc':
        hkl = [i for i in hkl if i[0] % 2 == i[1] % 2 == i[2] % 2]

    # N = h**2 + k**2 + l**2
    N = [i[0] ** 2 + i[1] ** 2 + i[2] ** 2 for i in hkl]

    # Dict of N against hkl - captures equiv reflections i.e. diff hkl, same N
    d = defaultdict(list)
    for k, v, in zip(N, hkl):
        d[k].append(v)

    # Remove the invalid (0, 0, 0) indices
    d.pop(0, None)

    # If there are equivalent reflections then sum the multiplicites and
    # keep hkl with lowest h
    hkl_unique = []
    M = []
    for v in d.values():
        n = [i[0] for i in v]
        hkl_unique.append(v[n.index(min(n))])
        M.append(sum([factor[hkl_type(*i)] for i in v]))

    return hkl_unique, M


def h2_l2_k2(a, q):
    """ Returns N (h**2 + k**2 + l**2) for given lattice parameter, a and q """
    return (a * q / (2 * np.pi))**2


def reciprocal_spacings(h, k, l, a):
    """ Returns reciprocal lattice spacings (i.e. q0 rather than d0) """
    return 2 * np.pi * (h**2 + k**2 + l**2)**0.5 / a


def peak_details(q_max, material=None, a=None, structure=None):
    """
    Returns three arrays containing peak position, multiplicity and miller
    indices (strings) based on the maximum sampled reciprocal spacing (q_max)
    and material properties (lattice parameter & crystal structure).

    The material properties can either be specified separately of deduced
    from the material.
    """
    error = 'Must define either material or lattice param & crystal structure'
    assert not all([i is None for i in [material, a, structure]]), error
    if material is not None:
        a = df_latt.loc[material]['a']
        structure = df_latt.loc[material]['structure']
        structure = 'simple' if structure == 'Simple Cubic' else structure
    error = 'Must select a cubic crystal structure [fcc/bcc/simple]'
    assert structure.lower() in ['fcc', 'bcc', 'simple'], error

    N_max = h2_l2_k2(a, q_max)
    hkl, M = cubic_multiplicity(structure, N_max)
    q0 = np.array([reciprocal_spacings(i[0], i[1], i[2], a) for i in hkl])
    miller_names = np.array(['{}{}{}'.format(i[0], i[1], i[2]) for i in hkl])
    return a, q0, np.array(M), miller_names
