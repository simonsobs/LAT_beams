"""
Utility functions for working with beam maps.

TODO: Make everything radians
"""

import datetime as dt
import os
from logging import Logger
from typing import Optional, cast

import astropy.units as u
import h5py
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from jaxtyping import Float, Shaped
from scipy.interpolate import interp1d
from sotodlib.core import AxisManager, Context
from sotodlib.site_pipeline import jobdb

from .utils.jobs import set_tag


def solid_angle(
    az: Float[np.ndarray, "nx"],
    el: Float[np.ndarray, "ny"],
    beam: Float[np.ndarray, "nx ny"],
    cent: tuple[int, int],
    r1: float,
    norm: float,
) -> float:
    """
    Compute the integrated solid angle of a beam map.
    This uses aperture photometry to handle bias from the background of the map.

    Parameters
    ----------
    az : Float[np.ndarray, "nx"]
        The x coordinates of the map in arcseconds.
    el : Float[np.ndarray, "nx"]
        The y coordinates of the map in arcseconds.
    beam : Float[np.ndarray, "nx ny"]
        The beam map to compute the solid angle of.
    cent : tuple[int, int]
        The index of the center pixel.
    r1 : float
        The radius of the inner ring in aperture photometry in arcseconds.
    norm : float
        The value to normalize the map by.

    Returns
    -------
    solid_angle : float
        the solid angle in stradians.
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


def estimate_solid_angle(
    imap: Float[np.ndarray, "nx ny"],
    model: Float[np.ndarray, "nx ny"],
    res: float,
    data_fwhm: float,
    cent: tuple[int, int],
    min_sigma: float,
) -> tuple[float, float, float, float]:
    r"""
    Estimate the solid angle of a map given a fit model.
    Here we correct for the bias in our solid angle integration by computing:

    $$
    \Omega = \tilde{\Omega_{imap}} \frac{\Omega_{model}}{\tilde{\Omega_{imap}}}
    $$

    Where $\tilde{\Omega}$ implies a solid angle estimated with the `solid_angle` function from this module.

    Parameters
    ----------
    imap : Float[np.ndarray, "nx ny"]
        The input map to estimate the solid angle of.
    model : Float[np.ndarray, "nx ny"]
        Model of map to estimate the solid angle of.
    res : float
        The resolution of the map in arcseconds.
    data_fwhm : float
        The FWHM of the map in arcseconds.
    cent : tuple[int, int]
        The index of the center pixel.
    min_sigma : float
        The number of sigma to use as the radius for aperture photometry.

    Returns
    -------
    data_solid_angle_meas : float
        $\tilde{\Omega_{imap}}$ in stradians.
    model_solid_angle_meas : float
        $\tilde{\Omega_{model}}$ in stradians.
    model_solid_angle_true : float
        $Omega_{model}$ in stradians.
    data_solid_angle_corr : float
        $Omega$ in stradians.
    """
    fwhm_pix = int(data_fwhm / res)
    # Get solid angles
    y = np.linspace(-imap.shape[0] * res / 2, imap.shape[0] * res / 2, imap.shape[0])
    x = np.linspace(-imap.shape[1] * res / 2, imap.shape[1] * res / 2, imap.shape[1])
    kern = Gaussian2DKernel((data_fwhm / 2.3548) / res, (data_fwhm / 2.3548) / res)
    imap_smooth = convolve_fft(imap, kern)
    model_smooth = convolve_fft(model, kern)
    norm = np.max(
        imap_smooth[
            max(0, cent[0] - fwhm_pix) : min(imap_smooth.shape[0], cent[0] + fwhm_pix),
            max(0, cent[1] - fwhm_pix) : min(imap_smooth.shape[1], cent[1] + fwhm_pix),
        ]
    )
    data_solid_angle_meas = solid_angle(
        x, y, imap_smooth, cent, min_sigma * (data_fwhm / 2.355), norm
    )
    model_solid_angle_meas = solid_angle(
        x, y, model_smooth, cent, min_sigma * (data_fwhm / 2.355), norm
    )
    model_solid_angle_true = solid_angle(x, y, model, cent, np.inf, np.max(model))
    data_solid_angle_corr = (
        data_solid_angle_meas * model_solid_angle_true / model_solid_angle_meas
    )
    return (
        data_solid_angle_meas,
        model_solid_angle_meas,
        model_solid_angle_true,
        data_solid_angle_corr,
    )


def radial_profile(
    data: Float[np.ndarray, "nx ny"], center: tuple[int, int]
) -> Float[np.ndarray, "nr"]:
    """
    Compute the radial profile of a beam.

    Parameters
    ----------
    data : Float[np.ndarray, "nx ny"]
        The input beam.
    center : tuple[int, int]
        The index of the center pixel.

    Returns
    -------
    radialprofile : Float[np.ndarray, "nr"]
        Radial profile of the input map.
    """
    msk = np.isfinite(data.ravel())
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel()[msk], data.ravel()[msk])
    nr = np.bincount(r.ravel()[msk])
    radialprofile = tbin / nr
    return radialprofile


def get_fwhm_radial_bins(
    r: Float[np.ndarray, "nr"], y: Float[np.ndarray, "nr"], interpolate: bool = False
) -> float:
    """
    Estimate FWHM from a radial profile.

    Parameters
    ----------
    r : Float[np.ndarray, "nr"]
        The radial position at each point.
    y : Float[np.ndarray, "nr"]
        The value of the profile at each point.
    interpolate : bool, default: False
        If True then interpolate the input profile on an evenly spaced
        grid of 100 points before estimating the FWHM.

    Returns
    -------
    fwhm : float
        The estimated FWHM in the same units as `r`.
    """
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
    return cast(float, fwhm)


def crop_maps(
    maps: list[Float[np.ndarray, "nx ny"]], cent: tuple[int, int], extent: int
) -> list[Float[np.ndarray, "2extent 2extent"]]:
    """
    Crop a list of maps to be smaller.
    Note that all input maps will be cropped relative to the same pixel.

    Parameters
    ----------
    maps : list[Float[np.ndarray, "nx ny"]]
        List of maps to crop.
        These should all have the same center.
    cent : tuple[int, int]
        The index of the center pixel.
    extent : int
        The extent of the output map.

    Returns
    -------
    cropped : list[Float[np.ndarray "2extent 2extent"]]
        The cropped maps.
        Each one will have size (2*extent, 2*extent)
        unless that goes outside of the input map's bounding box,
        in which case the cropped map stops at that box.
    """
    xmin = max(0, cent[0] - extent)
    xmax = min(maps[0].shape[-2], cent[0] + extent)
    ymin = max(0, cent[1] - extent)
    ymax = min(maps[0].shape[-1], cent[1] + extent)
    slx = slice(int(xmin), int(xmax))
    sly = slice(int(ymin), int(ymax))
    maps = [m[..., slx, sly] for m in maps]
    return maps


def estimate_cent(
    imap: Float[np.ndarray, "nx ny"], sigma: float = 5, buf: int = 30
) -> tuple[int, int]:
    """
    Estimate the location of the central pixel of a beam map.
    To do this we first smooth the map with a gaussian of size `sigma`,
    then we take the location of the maximum that is farther than `buf` from the edge of the map.

    Parameters
    ----------
    imap : Float[np.ndarray, "nx, ny"]
        The beam map to look for the center of.
    sigma : float
        The sigma of the gaussian in pixels to smooth
        the map by when searching for the max.
    buf : int
        Pixels within `buf` of the edge of the map
        will not be searched. Meant to avoid low hits
        pixels near the edge of the map.

    Returns
    -------
    cent : tuple[int, int]
        The index of the estimated center pixel.
    """
    smoothed = imap.copy()
    smoothed[smoothed == 0] = np.nan
    kern = Gaussian2DKernel(sigma, sigma)
    smoothed = convolve_fft(smoothed, kern)
    smoothed[:buf] = 0
    smoothed[-1 * buf :] = 0
    smoothed[:, :buf] = 0
    smoothed[:, -1 * buf :] = 0
    cent = np.unravel_index(np.argmax(smoothed, axis=None), smoothed.shape)

    return (int(cent[0]), int(cent[1]))


def process_model(
    aman: AxisManager,
    solved: Float[np.ndarray, "nx ny"],
    model: Float[np.ndarray, "nx ny"],
    noise: float,
    min_snr: float,
    c: tuple[int, int],
    map_units: u.Unit,
    pixsize: float,
    data_fwhm: float,
    min_sigma: float,
    job: Optional[jobdb.Job],
    logger: Optional[Logger],
) -> Optional[AxisManager]:
    """
    Convenience function to postproccess a map and it's fit model.
    This fundtion checks the SNR of the model,
    computes its radial profile,
    and computes the solid angle.

    Parameters
    ----------
    aman : AxisManager
        AxisManager with the fit parameters.
        No values are read off this, but it wil be modified in place.
    solved : Float[np.ndarray, "nx ny"]
        The map that was fit.
    model : Float[np.ndarray, "nx ny"]
        The model computed on the same grid as the map.
    noise : float
        The noise level of the map.
    min_snr : float
        The minimum SNR of the model.
        If the SNR is less than this then `None` is returned.
    c : tuple[int, int]
        The index of the center pixel.
    map_units : u.Unit
        The units of the map.
    pixsize : float
        The pixel size in arcseconds.
    data_fwhm : float
        The data fwhm in arcseconds.
    min_sigma : float
        See `estimate_solid_angle`.
    job : Optional[jobdb.Job]
        The job associated with the fit.
        Pass `None` if running without a job.
    logger : Optional[Logger]
        The logger to log with.
        Pass `None` if running without a logger.

    Returns
    -------
    aman : Optional[AxisManager]
        The input `aman` modified to add the radial profile
        of the model and all the solid angles output by
        `estimate_solid_angle`.
        If the SNR chech is not passed then `None` is returned.
    """
    # Check snr
    if np.nanmax(model) / noise < min_snr:
        msg = "Model SNR too low"
        if logger is None:
            print(f"\t{msg}")
        else:
            logger.error("\t%s", msg)
        if job is not None:
            set_tag(job, "message", msg)
            job.jstate = "failed"
        return None

    # Get model profile
    mprof = radial_profile(model, c[::-1])
    aman.wrap("mprof", mprof * map_units)

    # Get solid angle
    (
        data_solid_angle_meas,
        model_solid_angle_meas,
        model_solid_angle_true,
        data_solid_angle_corr,
    ) = estimate_solid_angle(solved, model, pixsize, data_fwhm.value, c, min_sigma)
    aman.wrap("data_solid_angle_meas", data_solid_angle_meas * u.sr)
    aman.wrap("data_solid_angle_corr", data_solid_angle_corr * u.sr)
    aman.wrap("model_solid_angle_meas", model_solid_angle_meas * u.sr)
    aman.wrap("model_solid_angle_true", model_solid_angle_true * u.sr)

    return aman


def load_beam_fits_from_jobs(
    fpath: str, joblist: list[jobdb.Job]
) -> Shaped[np.ndarray, "nfits"]:
    """
    Load beam fits from a list of jobs.

    Parameters
    ----------
    fpath : str
        The path to the HDF5 file containing the fits.
    job : list[jobdb.Job]
        List of jobs to load fits for.
        Jobs should be of jclass `fit_map`.

    Returns
    -------
    all_fits : Shaped[np.ndarray, "nfits"]
        Loaded fits.
        This is a numpy structured array with the following collumns:

        * obs_id : str, the obs_id of the fit data
        * wafer_slot : str, the wafer slot of the fit data
        * stream_id : str, the stream id of the fit data
        * band : str, the band (ie. f090) of the fit data
        * source : str, the source that was fit
        * time : float, the time of the observation
        * hour : float, what hour of the day the observation was at
        * aman : AxisManager, the loaded fit

    Raises
    ------
    ValueError
        If loaded fits do not all contain the same structure.
    """
    f = h5py.File(fpath, mode="r")
    obs_ids = np.array([job.tags["obs_id"] for job in joblist])
    times = np.array([float(o.split("_")[1]) for o in obs_ids])
    wafer_slots = np.array([job.tags["wafer_slot"] for job in joblist])
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
        zip(obs_ids, wafer_slots, stream_ids, bands, sources, times, tdelt, amans),
        dtype,
        count=len(amans),
    )
    return all_fits


def get_fit_vec(
    all_fits: Shaped[np.ndarray, "nfits"], name: str, fall_back: Optional[str] = None
) -> u.Quantity:
    """
    Get a fit value from all fits in a structured array.

    Parameters
    ----------
    all_fits : Shaped[np.ndarray, "nfits"]
        The fits to get values from.
        See `load_beam_fits_from_jobs` for details on the structure.
    name : str
        The name of the field to load from the AxisManagers in
        `all_fits["aman"]`.
    fall_back : str
        Field to load in `name` is not found.

    Returns
    -------
    fit_vec : u.Quantity
        The loaded values.
        Will have length `nfits`.
    """
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


def get_split_vec(
    fits: Shaped[np.ndarray, "nfits"], split: str, ctx: Context, round_to: int = 2
) -> Shaped[np.ndarray, "nfits"]:
    """
    Get an array of metadata to split fits by.

    Parameters
    ----------
    fits : Shaped[np.ndarray, "nfits"]
        The fits to get values from.
        See `load_beam_fits_from_jobs` for details on the structure.
    split : str
        List of collumns in `fits` or the obsdb to split by,
        should be one string with `+` to seperate collumn names.
    ctx : Context
        Context used to lookup values from the obsdb.
    round_to : int
        How many decimal places to round numeric collumns to.

    Returns
    -------
    split_vec : Shaped[np.ndarray, "nfits"]
        Array of strings constaining the values from the stlip collumns.
        Values are in the same order as `split` and are seperated by `+`s.
    """
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
