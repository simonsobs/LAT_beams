# TODO: UPDATE DOCSTRING
"""
Just fitting a single TOD for now, can genralize later (but maybe just do a WITCH interface for that)
Maybe should add priors
"""

import logging
import sys
import warnings
from copy import deepcopy

import numpy as np
import so3g
import sotodlib.coords.planets as planets
from astropy import units as u
from astropy.convolution import (
    Gaussian1DKernel,
    Gaussian2DKernel,
    convolve,
    convolve_fft,
)
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.signal import detrend
from scipy.stats import binned_statistic, binned_statistic_2d
from so3g.proj import Ranges, quat
from sotodlib import core
from sotodlib.core.context import AxisManager
from sotodlib.tod_ops.fft_ops import (
    RFFTObj,
    find_inferior_integer,
    find_superior_integer,
)
from sotodlib.tod_ops.filters import fourier_filter, high_pass_sine2, identity_filter
from sotodlib.tod_ops.filters import logger as flog
from sotodlib.tod_ops.filters import low_pass_sine2
from tqdm.auto import tqdm
from typing_extensions import Optional, cast

from .models import gaussian2d, multipole_decomp, multipole_expansion

flog.setLevel(logging.ERROR)


def get_xieta_src_centered(
    ctime: NDArray[np.floating],
    az: NDArray[np.floating],
    el: NDArray[np.floating],
    roll: NDArray[np.floating],
    sso_name: str,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Get xieta in source centered coordinates.
    Here we get the source trajectory at the center of the ctime array,
    for a slow source this is a fairly good approximation but you may get bad results
    if you are spanning a very large ctime or a ctime that spans both sides of a transit.

    Arguments:

        ctime: Array of times to get the source at. See docstring for caveat on this.

        az: Array of azimuth angles in radians to get the source at. Should be the same size as ctime.

        el: Array of elevation angles in radians to get the source at. Should be the same size as ctime.

        roll: Array of roll angles in radians to get the source at. Should be the same size as ctime.

    Returns:

        xi: The xi angles centered on the source in radians.

        eta: The eta angles centered on the source in radians.
    """
    csl = so3g.proj.CelestialSightLine.az_el(
        ctime, az, el, roll=roll, weather="typical"
    )
    q_bore = csl.Q  # type: ignore

    # planet position
    planet = planets.SlowSource.for_named_source(sso_name, ctime[int(len(ctime) / 2)])
    ra0, dec0 = planet.pos(ctime)
    q_obj = so3g.proj.quat.rotation_lonlat(ra0, dec0)

    q_total = ~q_bore * q_obj
    xi, eta, _ = quat.decompose_xieta(q_total)

    return xi, eta


def _empty_fp(aman: AxisManager) -> AxisManager:
    focal_plane = core.AxisManager(aman.dets)
    focal_plane.wrap("xi", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("eta", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("gamma", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("fwhm", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("amp", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("dist", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("hits", np.zeros(len(aman.dets.vals), dtype=int), [(0, "dets")])
    focal_plane.wrap("az", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("el", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap(
        "roll",
        np.zeros(len(aman.dets.vals), dtype=float) + np.mean(aman.boresight.roll),
        [(0, "dets")],
    )
    focal_plane.wrap(
        "reduced_chisq", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")]
    )
    focal_plane.wrap("R2", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])

    return focal_plane


# TODO: NEED DOCSTRING
# Smoothening stuff
def _bin_priors_1d(
    fit_am: AxisManager, xi0: float, eta0: float, fwhm: float
) -> tuple[float, float]:
    xi = np.array(fit_am.xi)
    eta = np.array(fit_am.eta)
    xi_binned, edges, _ = binned_statistic(
        xi, fit_am.resid_filt[0], bins=int(np.ptp(xi) / (1 * fwhm))
    )
    xi_cents = 0.5 * (edges[:-1] + edges[1:])
    xi_binned = convolve(
        xi_binned, Gaussian1DKernel((fwhm / 2.3548) / np.mean(np.diff(edges)))
    )
    if not np.all(np.isnan(xi_binned)):
        xi0 = xi_cents[np.nanargmax(xi_binned)]
    eta_binned, edges, _ = binned_statistic(
        eta, fit_am.resid_filt[0], bins=int(np.ptp(eta) / (1 * fwhm))
    )
    eta_cents = 0.5 * (edges[:-1] + edges[1:])
    eta_binned = convolve(
        eta_binned, Gaussian1DKernel((fwhm / 2.3548) / np.mean(np.diff(edges)))
    )
    if not np.all(np.isnan(eta_binned)):
        eta0 = eta_cents[np.nanargmax(eta_binned)]

    return xi0, eta0


# TODO: NEED DOCSTRING
def _bin_priors_2d(
    fit_am: AxisManager, xi0: float, eta0: float, fwhm: float
) -> tuple[float, float]:
    xi = np.array(fit_am.xi)
    eta = np.array(fit_am.eta)
    binned, x_edges, y_edges, _ = binned_statistic_2d(
        xi,
        eta,
        fit_am.resid_filt[0],
        bins=(int(np.ptp(xi) / (1 * fwhm)), int(np.ptp(eta) / (1 * fwhm))),  # type: ignore
    )
    warnings.filterwarnings("ignore", category=UserWarning, append=True)
    binned = convolve_fft(
        binned,
        Gaussian2DKernel(
            (fwhm / 2.3548) / np.mean(np.diff(x_edges)),
            (fwhm / 2.3548) / np.mean(np.diff(y_edges)),
        ),
    )
    xi_cents = 0.5 * (x_edges[:-1] + x_edges[1:])
    eta_cents = 0.5 * (y_edges[:-1] + y_edges[1:])
    if not np.all(np.isnan(binned)):
        max_idx = np.unravel_index(np.nanargmax(binned), binned.shape)
        xi0 = xi_cents[max_idx[0]]
        eta0 = eta_cents[max_idx[1]]
    return xi0, eta0


def filter_tod(am, filt, signal_name="resid", rfft=None):
    sig_filt_name = f"{signal_name}_filt"
    am[sig_filt_name] = am[signal_name].copy()
    filt_kw = dict(
        detrend="linear",
        resize=None,
        axis_name="samps",
        signal_name=sig_filt_name,
        time_name="timestamps",
        rfft=rfft,
    )
    am[sig_filt_name] = fourier_filter(am, filt, **filt_kw)
    return am


def fit_tod_pointing(
    aman: AxisManager,
    bandpass_range: tuple[Optional[float], Optional[float]] = (None, None),
    fwhm: float = np.deg2rad(0.5),
    max_rad: Optional[float] = None,
    source: str = "mars",
    bin_priors: bool = True,
    bin_2d: bool = True,
    pos_priors: Optional[NDArray[np.floating]] = None,
    show_tqdm: bool = False,
    min_snr: float = 5.0,
) -> AxisManager:
    """
    Fit detector offsets from a TOD of a source observation. Assumes that TOD has been trimmed to just the time source is in TOD.

    Arguments:

        aman: AxisManager containing at minimum a source TOD and boresight TOD.
              For best performance please fft trim ahead of time.

        bandpass_range: A tuple specifying the (low, high) cuttoffs for a bandpass filter in Hz.
                        Set a cutoff to None to not use it.

        max_rad: The maximum radius around the initial guess of the detector offset to use.
                 This is in radians, if None then 20  times the input fwhm is used.

        source: The name of the source to fit.

        bin_priors: If True try to guess the detector offsets by binning and smoothing the TODs.
                    If a positional prior is provided for a given detector it is used instead of this.

        bin_2d: If True the binned prior is done by binning the TOD into a naive map.
                If False xi and eta are binned seperately.

        pos_priors: Positional priors. Pass None to not use.
                    If using pass in a (ndets, 2) array where each row is the (xi, eta) to
                    start the fit for that detector at. To disable for only some detectors set
                    the row to (nan, nan).

        show_tqdm: If True show a progress bar.

        min_snr: Calculates hits out to min_snr compared to white noise level.

    Returns:

        focal_plane: An AxisManager with the same detectors as the input aman.
                     This contains the following fields:

                     * xi (in radians)
                     * eta (in radians)
                     * gamma (all 0 placeholder)
                     * fwhm (in radians)
                     * amp (in units of aman.signal)
                     * dist (in radians), distance between fit offset and the initial guess
                     * az (in radians), the azimuth at the detector crossing
                     * el (in radians), the elevation at the detector crossing
                     * roll (in radians), the roll at the detector crossing
                     * reduced_chisq
    """
    # TODO: Can use full detector map to fit the array at once. This hasnt been written.
    # Right now det maps for LAT not good; SAT will be ok to test with though. Should provide both options here,
    # since doing full det maps fit is faster than per det.
    # Either way, want to do per TOD because per TOD would be a refinement to the full det maps fit.
    if pos_priors is not None and len(pos_priors) != aman.dets.count:
        raise ValueError(
            f"{len(pos_priors)} positional priors given for {aman.dets.count} detectors"
        )
    if pos_priors is None:
        pos_priors = np.ones((cast(int, aman.dets.count), 2)) * np.nan
    sigma = fwhm / 2.3548
    if max_rad is None:
        max_rad = 20 * fwhm

    focal_plane = _empty_fp(aman)
    mean_el = np.mean(np.array(aman.boresight.el))

    # getting xi eta in a coordinate system where (0, 0) is the planet youre fitting. Expecting trimmed TOD for source.
    # Cannot include both rising and setting (ie a sign change). Note that a transit is flat -- so is ok.
    xi, eta = get_xieta_src_centered(
        np.array(aman.timestamps),
        np.array(aman.boresight.az),
        np.array(aman.boresight.el),
        np.array(aman.boresight.roll),
        source,
    )
    aman.wrap("xi", xi, [(0, "samps")])
    aman.wrap("eta", eta, [(0, "samps")])

    az_d = detrend(aman.boresight.az)
    d_az = np.sign(np.diff(az_d, prepend=az_d[0]))
    turnarounds = np.diff(d_az, prepend=d_az[0]) != 0
    turnarounds = ~Ranges.from_mask(turnarounds)  # Invert for convenience

    # 0 is the highpass part, 1 lowpass part.
    filt = identity_filter()
    if bandpass_range[0] is not None:
        filt *= high_pass_sine2(cutoff=bandpass_range[0])
    if bandpass_range[1] is not None:
        filt *= low_pass_sine2(cutoff=bandpass_range[1])

    def fit_func(x, fit_am, filt, rfft):
        xi0, eta0, amp, fwhm, offset = x
        model = gaussian2d(
            (fit_am.xi, fit_am.eta), amp, xi0, eta0, fwhm, fwhm, 0, offset
        )
        fit_am.resid = (fit_am.signal.ravel() - model).reshape(fit_am.resid.shape)
        fit_am = filter_tod(fit_am, filt, signal_name="resid", rfft=rfft)
        return np.sum(fit_am.resid * fit_am.resid_filt) * fit_am.wn

    # Loop through all detectors and fit them one at a time.
    it = np.array(aman.dets.vals)
    if show_tqdm:
        it = tqdm(np.array(aman.dets.vals))
    aman.signal = aman.signal.astype(np.float32)
    for i, det in enumerate(it):
        if show_tqdm:
            sys.stderr.flush()
        # Make a temporary restricted axis manager with just the one desired detector.
        fit_am = aman.restrict("dets", [det], in_place=False)
        # Containers for residual of fit and the filtered residual.
        fit_am.wrap("resid", fit_am.signal.copy(), [(0, "dets"), (1, "samps")])
        fit_am.wrap(
            "resid_filt", np.zeros_like(fit_am.signal), [(0, "dets"), (1, "samps")]
        )
        # Estimateing white noise by taking the standard deviation of the filtered TOD.
        fit_am = filter_tod(fit_am, filt)
        std = np.std(np.array(fit_am.resid_filt))
        if std == 0:
            focal_plane.amp[i] = -np.inf
            continue
        fit_am.wrap("wn", 1.0 / std.item() ** 2)

        # Get starting guess for where the detector is looking. Bad guess = bad pointing final fit. Descends into a local instead of global minima.
        # Get xi, eta at maximum sample, but this is not a very good guess.
        max_idx = np.argmax(np.array(fit_am.resid_filt[0]))
        xi_max = xi[max_idx]
        eta_max = eta[max_idx]
        xi0, eta0 = float(xi_max), float(eta_max)
        _bin_priors = bin_priors
        # If it has a det map fit then can use given positional prior
        if np.all(np.isfinite(pos_priors[i])):
            xi0, eta0 = pos_priors[i]
            _bin_priors = False

        # Determine if 1d or 2d binning used.
        # Bin in xi and eta
        if _bin_priors and not bin_2d:
            xi0, eta0 = _bin_priors_1d(fit_am, xi0, eta0, fwhm)
        elif _bin_priors and bin_2d:
            xi0, eta0 = _bin_priors_2d(fit_am, xi0, eta0, fwhm)
        msk_samps = np.where((xi - xi0) ** 2 + (eta - eta0) ** 2 < max_rad**2)[
            0
        ].astype(float)
        if len(msk_samps) < 10 and _bin_priors:
            xi0, eta0 = xi_max, eta_max
            msk_samps = np.where((xi - xi0) ** 2 + (eta - eta0) ** 2 < max_rad**2)[
                0
            ].astype(float)
        if len(msk_samps) < 10:
            print(f"Not enouth samples flagged for {det}")
            msk_samps = np.arange(cast(int, aman.samps.count))

        start = np.percentile(msk_samps, 5)
        stop = np.percentile(msk_samps, 95)
        cent = int(0.5 * (start + stop))
        nsamps = find_superior_integer(stop - start)
        if nsamps > cast(int, fit_am.samps.count):
            nsamps = find_inferior_integer(fit_am.samps.count)
        start = cent - nsamps // 2
        stop = cent + nsamps // 2 + nsamps % 2
        if start < 0:
            stop += start
            start = 0
        if stop > fit_am.samps.count:
            start -= fit_am.samps.count - stop
            stop = fit_am.samps.count
        sl = slice(
            start + cast(int, fit_am.samps.offset),
            stop + cast(int, fit_am.samps.offset),
        )
        fit_am.restrict("samps", sl)
        rfft = RFFTObj.for_shape(1, cast(int, fit_am.samps.count), "BOTH")

        ptp = np.ptp(np.array(fit_am.signal))
        amp = ptp * 3
        init_pars = [xi0, eta0, amp, fwhm, 0]
        bounds = [
            (xi0 - max_rad, xi0 + max_rad),
            (eta0 - max_rad, eta0 + max_rad),
            (-ptp, np.inf),
            (fwhm * 0.9, 1.1 * fwhm),
            (-ptp, ptp),
        ]

        # Nelder-Mead is only non-gradient method. The gradient ones get stuck on edges and find local minima. This find central global minima.
        res = minimize(
            fit_func,
            init_pars,
            bounds=bounds,
            args=(fit_am, filt, rfft),
            method="Nelder-Mead",
        )

        focal_plane.xi[i] = res.x[0]
        focal_plane.eta[i] = res.x[1]
        focal_plane.amp[i] = res.x[2]
        focal_plane.fwhm[i] = res.x[3]

        if not res.success:
            focal_plane.R2[i] = 0.0
        else:
            fit_am.resid = (fit_am.signal.ravel() - np.mean(fit_am.signal)).reshape(
                fit_am.resid.shape
            )
            fit_am = filter_tod(fit_am, filt, signal_name="resid", rfft=rfft)
            ss_tot = np.sum(fit_am.resid * fit_am.resid_filt) * fit_am.wn
            focal_plane.R2[i] = 1 - (res.fun / ss_tot)

        focal_plane.dist[i] = np.sqrt(
            (np.array(focal_plane.xi[i]) - xi0) ** 2
            + (np.array(focal_plane.eta[i]) - eta0) ** 2
        )
        delta_xi = xi - np.array(focal_plane.xi[i])
        delta_eta = eta - np.array(focal_plane.eta[i])

        # Lets calculate hits
        # solved for delta(x) in gaussian eqn ie
        # f(x)/wnl = A/wnl * e^(-.5 * delta(x) ^ 2 / sigma ^ 2)
        sigma = focal_plane.fwhm[i] / 2.3548
        snr_peak = np.abs(focal_plane.amp[i]) / (min_snr * std)
        if snr_peak >= 1:
            snr_rad = sigma * np.sqrt(2) * np.sqrt(np.log(snr_peak))
        else:
            snr_rad = -1
            focal_plane.R2[i] = 0
        radius = np.sqrt(delta_xi**2 + delta_eta**2)

        # We null out the mask where we turnaround so they count as seperate hits
        hits = Ranges.from_mask((radius <= snr_rad)) * turnarounds
        focal_plane.hits[i] = len(hits.ranges())

        # Azel crossings
        xi_weights = np.exp(-0.5 * ((delta_xi / sigma) ** 2)) / (
            sigma * np.sqrt(2 * np.pi)
        )
        eta_weights = np.exp(-0.5 * ((delta_eta / sigma) ** 2)) / (
            sigma * np.sqrt(2 * np.pi)
        )
        weights = xi_weights * eta_weights
        tot_weight = np.sum(weights)
        if tot_weight == 0:
            focal_plane.R2[i] = 0
        else:
            focal_plane.az[i] = np.sum(aman.boresight.az * weights) / tot_weight
            focal_plane.el[i] = np.sum(aman.boresight.el * weights) / tot_weight
            focal_plane.roll[i] = (
                cast(float, focal_plane.roll[i])
                + cast(float, focal_plane.el[i])
                - mean_el
            )

        # Chisq
        focal_plane.reduced_chisq[i] = res.fun / (
            cast(int, fit_am.samps.count) - len(res.x)
        )

    return focal_plane


def fit_gauss_beam(imap, ivar, pixmap, cent, multipoles=(0,), force_sym=False, map_units="pW"):
    """
    Fit 2d Gaussian to input map.
    This fit gaussian can include multipoles to capture extra structure (ie a cross).

    Arguments
    ---------
    imap : (ny, nx) enmap
        Input map.
    ivar : (ny, nx) enmap
        Inverse-variance map.
    pixmap : (2, ny, nx) array
        X and Y pixel indices for each pixel.
    cent : tuple
        The index of the map center
    multipoles : tuple, default: (0,)
        The multipoles to include in the fit.
        0 is the monopole, 1 is the dipole, 2 the quadropole, 3 the octopole, etc.
    force_sym : bool, default: False
        If True don't allow ellipticity in the fit.
    map_units : str, default: pW
        The units of the map.

    Returns
    -------
    fit_params : dict
        The fit params.
        Base params are: xi0, eta0, off, amp, fwhm_xi, fwhm_eta, phi.
        Multipoles will have '_m{multipole_index}_{0 or 1} appended,
        where the 0 or 1 is 0 for the sin term and 1 for the cos term.
    """
    res = imap.wcs.wcs.cdelt[1] * (60 * 60)
    pixmap = (pixmap[0] * res, pixmap[1] * res)  # convert to arcsec
    nx, ny = imap.shape[-2:]

    # Make weights and zero things out
    sigma = np.sqrt(ivar)
    sigma[~np.isfinite(sigma)] = 0
    sigma[~np.isfinite(imap)] = 0
    imap[~np.isfinite(imap)] = 0

    guess = [
        pixmap[0][cent[0], cent[1]],
        pixmap[1][cent[0], cent[1]],
        0,
        imap[cent[0], cent[1]],
        60,
        60,
        0,
    ]
    bounds = [
        [0, 0, -5 * np.max(np.abs(imap)), 0, 20, 20, 0],
        [nx * res, ny * res, 5 * np.max(imap), 5 * np.max(imap), 300, 300, 2 * np.pi],
    ]
    map_units = u.Unit(map_units)
    par_names = ["xi0", "eta0", "off", "amp", "fwhm_xi", "fwhm_eta", "phi"]
    par_units = [u.arcsec, u.arcsec, map_units, map_units, u.arcsec, u.arcsec, u.radian]  # type: ignore
    if force_sym:
        guess = guess[:-2]
        bounds[0] = bounds[0][:-2]
        bounds[1] = bounds[1][:-2]
    bounds = [(lb, ub) for lb, ub in zip(*bounds)]

    pixmap = (pixmap[0].astype(float), pixmap[1].astype(float))

    def _to_pars(coeffs):
        dx, dy, off, amp = coeffs[:4]

        if force_sym:
            fwhm_xi = fwhm_eta = coeffs[4]
            phi = 0
        else:
            fwhm_xi, fwhm_eta, phi = coeffs[4:]

        return dx, dy, off, amp, fwhm_xi, fwhm_eta, phi

    def _get_base_theta(dx, dy, off, fwhm_xi, fwhm_eta, phi):
        x, y = pixmap[0], pixmap[1]
        theta = np.arctan2(y - dy, x - dx)
        xieta = (x, y)
        base_beam = gaussian2d(xieta, 1, dx, dy, fwhm_xi, fwhm_eta, phi, 0)

        return base_beam, theta

    def _objective(coeffs,):
        dx, dy, off, amp, fwhm_xi, fwhm_eta, phi = _to_pars(coeffs)
        beam = gaussian2d(pixmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, off)

        diff = imap - beam
        chisq = np.nansum((diff * sigma) ** 2)
        return chisq

    res = minimize(_objective, guess, bounds=bounds)
    if not res.success:
        return None, None

    # Compute model
    dx, dy, off, amp, fwhm_xi, fwhm_eta, phi = pars = _to_pars(res.x)
    base_beam, theta = _get_base_theta(dx, dy, off, fwhm_xi, fwhm_eta, phi)
    if len(multipoles) == 0:
        amps = []
        model = amp*(base_beam - off)
    amps = multipole_decomp(base_beam - off, imap - off, sigma, multipoles, theta, True)
    base_beam = gaussian2d(pixmap, 1, dx, dy, fwhm_xi, fwhm_eta, phi, 0)
    model = multipole_expansion(base_beam, amps, multipoles, theta)

    # Convert to a dict
    params = {n: v * u for n, u, v in zip(par_names, par_units, pars)}
    for m, n in enumerate(multipoles):
        for i in (0, 1):
            j = 2 * m + i
            params[f"amp_m{m}_{i}"] = amps[j] * map_units

    return params, model
