from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np

def gaussian(x, *p):
    """
    Guassian curve fit for diffraction data.
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    """
    return p[0] + p[1] * np.exp(- (x - p[2])**2 / (2. * p[3]**2))


def gaussian_2d(pnts=(8, 8), xlim=(-1, 1), ylim=(-1, 1), height=1,
                centre=(0., -0.), sigma=(0.5, 0.5)):

    x = np.linspace(xlim[0], xlim[1], pnts[0])
    y = np.linspace(ylim[0], ylim[1], pnts[1])
    X, Y = np.meshgrid(x, y)

    Z = (gaussian(X, *(0, np.sqrt(height), centre[0], sigma[0])) *
         gaussian(Y, *(0, np.sqrt(height), centre[1], sigma[1])))

    return X, Y, Z


def gauss2d_tensor(pnts=(8, 8), xlim=(-1, 1), ylim=(-1, 1), max_strain=1e-2):
    """ Creates a sensible, non-symmetric strain distribution for testing"""

    xs, ys = (xlim[1] - xlim[0]) / 2, (ylim[1] - ylim[0]) / 2
    X, Y, e_xx = gaussian_2d(pnts, xlim, ylim, height=max_strain,
                             centre=(xs * 0.25, ys * -0.25), sigma=(0.5, 0.5))

    e_yy = gaussian_2d(pnts, xlim, ylim, height=max_strain,
                       centre=(xs * -0.25, ys * 0.25), sigma=(0.5, 0.5))[2]

    e_xy = gaussian_2d(pnts, xlim, ylim, height=max_strain/10,
                       centre=(xs * 0., ys * 0.), sigma=(0.5, 0.5))[2]

    return X, Y, e_xx, e_yy, e_xy


def crack_tensor(pnts=(8, 8), xlim=(-1, 1), ylim=(-1, 1), K=20):
    pass


def ring_array(detector, pnts=(8, 8), xlim=(-1, 1), ylim=(-1, 1), background=0,
               exclude=0.05, crop=0, strain_func=gauss2d_tensor, **kwargs):
    X, Y, e_xx, e_yy, e_xy = strain_func(pnts, xlim, ylim, **kwargs)
    shape = detector.phi.shape
    cropped_shape = (int((1 - crop) * shape[0]), int((1 - crop) * shape[1]))
    print(cropped_shape)
    images = np.zeros((X.shape + cropped_shape))
    for idx in np.ndindex(e_xx.shape):
        # Representative detector but the s_to_d is small so we crop
        tensor = (e_xx[idx], e_yy[idx], e_xy[idx])
        img = detector.rings(exclude, crop, background, strain_tensor=tensor)
        images[idx] = img
    print(images.shape, e_xx.shape)
    return X, Y, images, (e_xx, e_yy, e_xy)


def intensity_array(detector, pnts=(8, 8), xlim=(-1, 1), ylim=(-1, 1),
                    background=0, strain_func=gauss2d_tensor, **kwargs):
    X, Y, e_xx, e_yy, e_xy = strain_func(pnts, xlim, ylim, **kwargs)
    intensity = np.zeros(X.shape + (detector.phi.size, detector.q.size))
    for idx in np.ndindex(e_xx.shape):
        # Representative detector but the s_to_d is small so we crop
        tensor = (e_xx[idx], e_yy[idx], e_xy[idx])

        data = detector.intensity(background=background,
                                  strain_tensor=tensor)
        intensity[idx] = data[1]['total']
    q = np.repeat(data[0][None, :], 23, axis=0)
    return X, Y, q, intensity, (e_xx, e_yy, e_xy)
