import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy.interpolate import interp1d
from scipy.ndimage import sobel


def solid_angle(az, el, beam, cent, r1, norm):
    """Compute the integrated solid angle of a beam map.

    return value is in steradians  (sr)
    """
    r2 = np.sqrt(2) * r1

    _az, _el = np.meshgrid(az, el)
    r = np.sqrt((_az - _az[cent]) ** 2 + (_el - _el[cent]) ** 2)
    # convert from arcsec to rad
    az = np.deg2rad(az / 3600)
    el = np.deg2rad(el / 3600)

    integrand = beam / norm

    # perform the solid angle integral
    _integrand = integrand.copy()
    _integrand[r > r1] = 0
    integral_inner = np.trapz(np.trapz(_integrand, el, axis=0), az, axis=0)

    _integrand = integrand.copy()
    _integrand[(r < r1) + (r > r2)] = 0
    integral_outer = np.trapz(np.trapz(_integrand, el, axis=0), az, axis=0)
    return integral_inner - integral_outer


def estimate_solid_angle(imap, model, res, data_fwhm, c, off, min_sigma):
    fwhm_pix = int(data_fwhm / res)
    # Get solid angles
    y = np.linspace(-imap.shape[0] * res / 2, imap.shape[0] * res / 2, imap.shape[0])
    x = np.linspace(-imap.shape[1] * res / 2, imap.shape[1] * res / 2, imap.shape[1])
    kern = Gaussian2DKernel((data_fwhm / 2.3548) / res, (data_fwhm / 2.3548) / res)
    imap_smooth = convolve_fft(imap - off, kern)
    model_smooth = convolve_fft(model - off, kern)
    norm = np.max(
        imap_smooth[
            max(0, c[0] - fwhm_pix) : min(imap_smooth.shape[0], c[0] + fwhm_pix),
            max(0, c[1] - fwhm_pix) : min(imap_smooth.shape[1], c[1] + fwhm_pix),
        ]
    )
    data_solid_angle_meas = solid_angle(
        x, y, imap_smooth, c, min_sigma * (data_fwhm / 2.355), norm
    )
    model_solid_angle_meas = solid_angle(
        x, y, model_smooth, c, min_sigma * (data_fwhm / 2.355), norm
    )
    model_solid_angle_true = solid_angle(x, y, model, c, np.inf, np.max(model))
    data_solid_angle_corr = (
        data_solid_angle_meas * model_solid_angle_true / model_solid_angle_meas
    )
    return (
        data_solid_angle_meas,
        model_solid_angle_meas,
        model_solid_angle_true,
        data_solid_angle_corr,
    )


def radial_profile(data, center):
    msk = np.isfinite(data.ravel())
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel()[msk], data.ravel()[msk])
    nr = np.bincount(r.ravel()[msk])
    radialprofile = tbin / nr
    return radialprofile


def get_fwhm_radial_bins(r, y, interpolate=False):
    half_point = np.max(y) * 0.5

    if interpolate:
        r_diff = r[1] - r[0]
        interp_func = interp1d(r, y)
        r_interp = np.arange(np.min(r), np.max(r) - r_diff, r_diff / 100)
        y_interp = interp_func(r_interp)
        r, y = (r_interp, y_interp)
    d = y - half_point
    inds = np.where(d > 0)[0]
    fwhm = 2 * (r[inds[-1]])
    return fwhm


def get_cent(imap, buf=30, sigma=5):
    smoothed = imap.copy()
    smoothed[smoothed == 0] = np.nan
    kern = Gaussian2DKernel(sigma, sigma)
    smoothed = convolve_fft(smoothed, kern)
    smoothed[:buf] = 0
    smoothed[-1 * buf :] = 0
    smoothed[:, :buf] = 0
    smoothed[:, -1 * buf :] = 0
    cent = np.unravel_index(np.argmax(smoothed, axis=None), smoothed.shape)

    return cent


def crop_maps(maps, cent, extent):
    xmin = max(0, cent[0] - extent)
    xmax = min(maps[0].shape[0], cent[0] + extent)
    ymin = max(0, cent[1] - extent)
    ymax = min(maps[0].shape[1], cent[1] + extent)
    maps = [m[xmin:xmax, ymin:ymax] for m in maps]
    return maps
