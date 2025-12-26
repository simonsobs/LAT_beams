import datetime as dt
import os

import astropy.units as u
import h5py
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy.interpolate import interp1d
from sotodlib.core import AxisManager


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


def crop_maps(maps, cent, extent):
    xmin = max(0, cent[0] - extent)
    xmax = min(maps[0].shape[-2], cent[0] + extent)
    ymin = max(0, cent[1] - extent)
    ymax = min(maps[0].shape[-1], cent[1] + extent)
    slx = slice(int(xmin), int(xmax))
    sly = slice(int(ymin), int(ymax))
    maps = [m[..., slx, sly] for m in maps]
    return maps


def load_beam_fits(fpath):
    f = h5py.File(fpath, mode="r")
    obs_ids = []
    times = []
    stream_ids = []
    bands = []
    for o in f.keys():
        for s in f[o].keys():
            for b in f[o][s].keys():
                obs_ids += [o]
                times += [float(o.split("_")[1])]
                stream_ids += [s]
                bands += [b]
    obs_ids = np.array(obs_ids)
    times = np.array(times)
    stream_ids = np.array(stream_ids)
    bands = np.array(bands)

    dates = [dt.date.fromtimestamp(ct) for ct in times]
    tdelt = (
        np.array(
            [
                ct - dt.datetime(year=d.year, month=d.month, day=d.day).timestamp()
                for ct, d in zip(times, dates)
            ]
        )
        / 3600
    )

    amans = np.array(
        [
            AxisManager.load(f[os.path.join(o, s, b)])
            for o, s, b in zip(obs_ids, stream_ids, bands)
        ]
    )
    # check that all fits have the same pars
    par_list = np.sort(list(amans[0]._fields.keys()))
    for aman in amans:
        pars = np.sort(list(aman._fields.keys()))
        if not np.array_equal(par_list, pars):
            raise ValueError("Not all fits have the same pars!")
    f.close()
    dtype = [
        ("obs_id", obs_ids.dtype),
        ("stream_id", stream_ids.dtype),
        ("band", bands.dtype),
        ("time", float),
        ("hour", float),
        ("aman", "O"),
    ]
    all_fits = np.fromiter(
        zip(obs_ids, stream_ids, bands, times, tdelt, amans), dtype, count=len(amans)
    )
    return all_fits


def load_beam_fits_from_jobs(fpath, jobs):
    f = h5py.File(fpath, mode="r")
    obs_ids = np.array([job.tags["obs_id"] for job in joblist])
    times = np.array([float(o.split("_")[1]) for o in obs_ids])
    wafer_slots = np.array([job.tags["wafer_slots"] for job in joblist])
    stream_ids = np.array([job.tags["stream_id"] for job in joblist])
    bands = np.array([job.tags["band"] for job in joblist])
    sources = np.array([job.tags["source"] for job in joblist])
    dates = np.array([dt.date.fromtimestamp(ct) for ct in times])
    tdelt = (
        np.array(
            [
                ct - dt.datetime(year=d.year, month=d.month, day=d.day).timestamp()
                for ct, d in zip(times, dates)
            ]
        )
        / 3600
    )

    amans = np.array(
        [
            AxisManager.load(f[os.path.join(o, s, b)])
            for o, s, b in zip(obs_ids, stream_ids, bands)
        ]
    )
    # check that all fits have the same pars
    par_list = np.sort(list(amans[0]._fields.keys()))
    for aman in amans:
        pars = np.sort(list(aman._fields.keys()))
        if not np.array_equal(par_list, pars):
            raise ValueError("Not all fits have the same pars!")
    f.close()
    dtype = [
        ("obs_id", obs_ids.dtype),
        ("wafer_slot", wafer_slots.dtype),
        ("stream_id", stream_ids.dtype),
        ("band", bands.dtype),
        ("source", sources.dtype),
        ("time", float),
        ("hour", float),
        ("aman", "O"),
    ]
    all_fits = np.fromiter(
        zip(obs_ids, wafer_slot, stream_ids, bands, sources, times, tdelt, amans),
        dtype,
        count=len(amans),
    )
    return all_fits


def get_fit_vec(all_fits, name, fall_back=None):
    if fall_back is not None:
        dat = u.Quantity(
            [
                aman[name] if name in aman else aman[fall_back]
                for aman in all_fits["aman"]
            ]
        )
    else:
        dat = u.Quantity([aman[name] for aman in all_fits["aman"]])

    if dat.unit == u.Unit(3):
        dat = dat.value * u.pW
    return dat


def get_split_vec(fits, split, ctx, round_to=2):
    split_vecs = []
    for spl in split.split("+"):
        if spl in fits.dtype.names:
            split_vecs += [fits[spl].astype(str)]
            continue
        split_vec = []
        for fit in fits:
            obs = ctx.obsdb.get(fit["obs_id"])
            split_vec += [obs[spl]]
        split_vec = np.array(split_vec)
        if np.issubdtype(split_vec.dtype, np.number):
            split_vec = np.round(split_vec, round_to)
        split_vecs += [split_vec.astype(str)]
    split_vecs = np.column_stack(split_vecs)

    return np.array(["+".join(v) for v in split_vecs])


def estimate_cent(imap, sigma=5, buf=30):
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


def subpix_shift(imap, ishape, iwcs):
    crdelt = iwcs.wcs.crval - imap.wcs.wcs.crval
    cpdelt = iwcs.wcs.crpix - imap.wcs.wcs.crpix
    subpix = (crdelt / iwcs.wcs.cdelt - cpdelt + 0.5) % 1 - 0.5
    imap2 = enmap.fractional_shift(imap, -1 * subpix[::-1], nofft=False)

    return imap2


def coadd(imaps, iweights, medsub=True):
    # Get a joint geometry and init output maps
    ishape, iwcs = imaps[0].shape, imaps[0].wcs
    imaps = [subpix_shift(imap, ishape, iwcs) for imap in imaps]
    iweights = [subpix_shift(imap, ishape, iwcs) for imap in iweights]
    oshape, owcs = enmap.union_geometry([im.geometry for im in imaps])
    omap = enmap.zeros((len(imaps[0].shape),) + oshape, owcs)
    oweight = enmap.zeros((len(imaps[0].shape),) + oshape, owcs)

    # Coadd
    op = np.ndarray.__iadd__
    for im, iw in zip(imaps, iweights):
        oweight.insert(iw, op=op)
        off = 0
        # if medsub:
        #     off = utils.weighted_median(im, iw, axis=0)
        omap.insert(iw * (im - off), op=op)
    with np.errstate(divide="ignore", invalid="ignore"):
        omap /= oweight
    np.nan_to_num(omap, copy=False, nan=0, posinf=0, neginf=0)

    return omap, oweight


def recenter(imap, obs_id, stream_id, band, fit_file, norm=True, extent=None):
    aman_path = os.path.join(obs_id, stream_id, band)
    aman = AxisManager.load(fit_file, aman_path)
    cent = enmap.sky2pix(
        imap.shape,
        imap.wcs,
        np.array((aman.eta0.to(u.rad).value, aman.xi0.to(u.rad).value)),
    )
    zero = enmap.sky2pix(imap.shape, imap.wcs, np.array((0, 0)))
    imap = enmap.shift(imap, cent - zero)

    if norm:
        imap = (imap) / aman.amp.value

    if extent is not None:
        imap = crop_maps(
            [imap],
            enmap.sky2pix(imap.shape, imap.wcs, np.array((0, 0))),
            extent / np.abs(3600 * imap.wcs.wcs.cdelt[1]),
        )[0]

    return imap
