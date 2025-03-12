import numpy as np
from sotodlib.core import AxisManager

def dct_i(dat, inv: bool = False, axis: int = -1):
    r"""
    Compute the 1d discrete cosine transform (DCT) of the first kind.

    Note that since this uses `rfft` to compute the DCT this suffers from
    slightly lower accuracy than a proper DCT such at the `fftw` or `scipy` ones.
    Passing white noise back and forth through this function results in errors
    at the $1e-7$ level.

    Parameters
    ----------
    dat : Array
        The data to DCT.
        If this is multidimensional the DCT is computed along only `axis`.
    inv : bool, defualt: False
        If True apply the $\frac{1}{2*(n-1)}$ normalization that turns this into
        the inverse DCT.
    axis : int, defualt: -1
        The axis to compute the DCT along.

    Returns
    -------
    dat_dct : Array
        The DCT of the first kind of `dat`.
        Has the same shape and dtype as `dat`.
    """
    s = dat.shape
    dat = np.atleast_2d(dat)
    dat_tmp = np.hstack([dat, np.fliplr(dat[:, 1:-1])])
    dat_dct = np.real(np.fft.rfft(dat_tmp))
    if inv:
        dat_dct = dat_dct * (1.0 / (2 * (dat.shape[axis] - 1)))
    return dat_dct.reshape(s)

def gauss_smooth_1d(dat, fwhm):
    """
    Smooth an array along its last axis with a gaussian.
    If you want to smooth along another axis consider using `roll` or `moveaxis`.

    Parameters
    ----------
    dat : Array
        The data to smooth.
    fwmh : float
        The full width half max of the gaussian used to smooth.

    Returns
    -------
    dat_smooth : Array
        The smoothed data.
        Has the same shape and dtype as `dat`.
    """
    smooth_kern = np.exp(
        -0.5 * (np.arange(dat.shape[-1]) * np.sqrt(8 * np.log(2)) / fwhm) ** 2
    )
    tot = smooth_kern[0] + smooth_kern[-1] + 2 * np.sum(smooth_kern[1:-1])
    smooth_kern /= tot
    smooth_kern = dct_i(smooth_kern)
    dat_smooth = dct_i(dat)
    dat_smooth = dat_smooth*smooth_kern
    dat_smooth = dct_i(dat_smooth, True)

    return dat_smooth

def compute_noise(aman, dat, fwhm=20):
    if dat is None:
        dat = aman.signal
    if aman.signal.shape != dat.shape:
        raise ValueError("data shape does not match aman.signal")

    u, *_ = np.linalg.svd(dat, False)
    v = u.T
    dat_rot = np.dot(v, dat)
    dat_ft = dct_i(dat_rot)
    dat_ft = gauss_smooth_1d(dat_ft**2, fwhm)
    dat_ft[:, 1:] = 1.0 / dat_ft[:, 1:]
    dat_ft[:, 0] = 0.

    if "noise" in aman:
        aman.move("noise", None)
    noise = AxisManager(aman.dets, aman.samps)
    noise.wrap("v", v.astype(dat.dtype), [(0, "dets"), (1, "dets")])
    noise.wrap("filt_spectrum", dat_ft.astype(dat.dtype), [(0, "dets")])
    aman.wrap("noise", noise)

    return aman


def apply_noise(aman, dat):
    if dat is None:
        dat = aman.signal
    if aman.signal.shape != dat.shape:
        raise ValueError("data shape does not match aman.signal")
    if "noise" not in aman:
        raise ValueError("no noise model in aman!")

    dat_rot = np.dot(aman.noise.v, dat)
    dat_rft = dct_i(dat_rot)
    dat_filt = dct_i(
        dat_rft*aman.noise.filt_spectrum[:, : dat_rft.shape[1]], False
    )
    dat_filt = np.dot(aman.noise.v.T, dat_filt)
    dat_filt[:, 0] *= 0.5
    dat_filt[:, -1] *= 0.5
    return dat_filt.astype(dat.dtype)
