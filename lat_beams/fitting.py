# TODO: UPDATE DOCSTRING
# TODO: Cleanup and standardize interfaces
"""
Just fitting a single TOD for now, can genralize later (but maybe just do a WITCH interface for that)
Maybe should add priors
"""

import logging
import sys
import warnings

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
from scipy.special import jv
from scipy.stats import binned_statistic, binned_statistic_2d
from so3g.proj import Ranges, quat
from sotodlib import core
from sotodlib.core import AxisManager, IndexAxis, LabelAxis
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

from .models import (
    bessel_term,
    dr4_beam,
    gaussian2d,
    multipole_decomp,
    multipole_expansion,
    scatter_beam,
)

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
            (fit_am.eta, fit_am.xi), amp, xi0, eta0, fwhm, fwhm, 0, offset
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


def fit_gauss_beam(
    imap,
    ivar,
    posmap,
    cent,
    force_sym=False,
    map_units="pW",
    fwhm_start=np.deg2rad(1 / 60.0),
    mask_size=-1,
):
    """
    Fit 2d Gaussian to input map.
    This fit gaussian can include multipoles to capture extra structure (ie a cross).

    Arguments
    ---------
    imap : (ny, nx) enmap
        Input map.
    ivar : (ny, nx) enmap
        Inverse-variance map.
    posmap: (2, ny, nx) array
        X and Y coordinates for each pixel in radians.
    cent : tuple
        The index of the map center
    force_sym : bool, default: False
        If True don't allow ellipticity in the fit.
    map_units : str, default: pW
        The units of the map.
    fwhm_start : float, default: np.deg2rad(1/60)
        The starting guess of fwhm in radians.
    mask_size : float, default: -1
        If this is >0 then a mask will be applies to ivar
        such that only data withing mask_size*fwhm_start of cent
        is used in the fit.

    Returns
    -------
    fit_params : dict
        The fit params.
        Base params are: xi0, eta0, off, amp, fwhm_xi, fwhm_eta, phi.
    """
    y, x = posmap
    guess = [
        x[cent[0], cent[1]],
        y[cent[0], cent[1]],
        0,
        imap[cent[0], cent[1]],
        fwhm_start,
        fwhm_start,
        0,
    ]
    bounds = [
        [
            np.min(x) - fwhm_start,
            np.min(y) - fwhm_start,
            -5 * np.max(np.abs(imap)),
            0,
            fwhm_start / 3,
            fwhm_start / 3,
            0,
        ],
        [
            np.max(x) + fwhm_start,
            np.max(y) + fwhm_start,
            5 * np.max(imap),
            5 * np.max(imap),
            fwhm_start * 3,
            fwhm_start * 3,
            2 * np.pi,
        ],
    ]
    map_units = u.Unit(map_units)
    par_names = ["xi0", "eta0", "off", "amp", "fwhm_xi", "fwhm_eta", "phi"]
    par_units = [u.radian, u.radian, map_units, map_units, u.radian, u.radian, u.radian]  # type: ignore
    if force_sym:
        guess = guess[:-2]
        bounds[0] = bounds[0][:-2]
        bounds[1] = bounds[1][:-2]
    bounds = [(lb, ub) for lb, ub in zip(*bounds)]

    # Mask out things too far from the starting center
    if mask_size > 0:
        r = np.sqrt((x - guess[0]) ** 2 + (y - guess[1]) ** 2)
        ivar = ivar.copy()
        ivar[r > mask_size * fwhm_start] = 0

    def _to_pars(coeffs):
        dx, dy, off, amp = coeffs[:4]

        if force_sym:
            fwhm_xi = fwhm_eta = coeffs[4]
            phi = 0
        else:
            fwhm_xi, fwhm_eta, phi = coeffs[4:]

        return dx, dy, off, amp, fwhm_xi, fwhm_eta, phi

    def _objective(
        coeffs,
    ):
        dx, dy, off, amp, fwhm_xi, fwhm_eta, phi = _to_pars(coeffs)
        beam = gaussian2d(posmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, off)

        diff = imap - beam
        chisq = np.nansum((diff**2) * ivar)
        return chisq

    res = minimize(_objective, guess, bounds=bounds)
    if not res.success:
        return None, None

    # Convert to aman
    aman = AxisManager()
    dx, dy, off, amp, fwhm_xi, fwhm_eta, phi = pars = _to_pars(res.x)
    for n, un, v in zip(par_names, par_units, pars):
        aman.wrap(n, v * un)
    model = gaussian2d(posmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, off)

    return aman, model


def fit_multipole_model(imap, ivar, posmap, base_beam, gauss_fit, n_multipoles):
    y, x = posmap
    theta = np.arctan2(
        y - gauss_fit.eta0.to(u.radian).value, x - gauss_fit.eta0.to(u.radian).value
    )

    # Compute model
    if n_multipoles == 0:
        amps = np.array([[gauss_fit.amp.values, 0]])
    amps = multipole_decomp(base_beam, imap, ivar, n_multipoles, theta, True)
    model = multipole_expansion(base_beam, amps, theta)

    # Convert to aman
    map_units = gauss_fit.amp.unit
    aman = AxisManager()
    mp_ax = IndexAxis("multipoles", n_multipoles)
    sc_ax = LabelAxis("term", ["cos", "sin"])
    aman.wrap("amps", amps * map_units, [(0, mp_ax), (1, sc_ax)])

    return aman, model


def fit_bessel_model(
    imap,
    ivar,
    posmap,
    gauss_fit,
    n_bessel,
    n_multipoles,
    d,
    lmd,
    force_cent=False,
    fit_wing=False,
    mask_size=np.inf,
    data_fwhm=np.inf,
):
    ell_max = (np.pi * d / lmd).decompose().value
    eta, xi = posmap
    eta0 = gauss_fit.eta0.to(u.radian).value
    xi0 = gauss_fit.xi0.to(u.radian).value
    xi = xi - xi0
    eta = eta - eta0
    theta = np.arctan2(eta, xi)
    r = np.sqrt(xi**2 + eta**2)

    # Compute model
    amps = np.zeros((n_bessel, n_bessel, n_multipoles, 2))
    beam_model = np.zeros_like(xi)
    for n0 in range(len(amps)):
        b0 = bessel_term(r, ell_max, n0)
        for n1 in range(len(amps)):
            b1 = bessel_term(r, ell_max, n1)
            base_beam = b0 * b1
            amps[n0, n1] = multipole_decomp(
                base_beam, imap - beam_model, ivar, n_multipoles, theta, True, True
            )
            beam_model += multipole_expansion(base_beam, amps[n0, n1], theta)

    # Deal with numerical errors near the center from having r in the denom
    if force_cent:
        cent_pix = r < np.deg2rad(posmap.wcs.wcs.cdelt[1]) / 2
        beam_model[cent_pix] = gauss_fit.amp.value + gauss_fit.off.value
        cent_ring = (
            (r < 2 * np.deg2rad(posmap.wcs.wcs.cdelt[1]))
            * (~cent_pix)
            # * (beam_model >= gauss_fit.amp.value)
        )
        # Radial interp
        ci, cj = np.where(cent_pix)
        for i, j in zip(*np.where(cent_ring)):
            if i > beam_model.shape[0] or j > beam_model.shape[1]:
                beam_model[i, j] = gauss_fit.amp.value
            beam_model[i, j] = (
                2 * (gauss_fit.amp.value + gauss_fit.off.value)
                + beam_model[2 * i - ci[0], 2 * j - cj[0]]
            ) / 3

    # Convert to aman
    map_units = gauss_fit.amp.unit
    aman = AxisManager()
    b_ax = IndexAxis("bessel", n_bessel)
    mp_ax = IndexAxis("multipoles", n_multipoles)
    sc_ax = LabelAxis("term", ["cos", "sin"])
    aman.wrap("amps", amps * map_units, [(0, b_ax), (1, b_ax), (2, mp_ax), (3, sc_ax)])
    aman.wrap("ell_max", ell_max)
    aman.wrap("force_cent", force_cent)
    aman.wrap("fit_wing", fit_wing)
    aman.wrap("r0_wing", np.inf * u.radian)
    aman.wrap("amp_wing", 0 * map_units)
    aman.wrap("off_wing", 0 * map_units)
    aman.wrap("off_core", 0 * map_units)

    if not fit_wing:
        return aman, beam_model

    # Fit for a symmetric r^-3 wing
    wing_model = beam_model.copy()
    ivar = ivar.copy()
    ivar[r > mask_size] = 0

    def _wing_obj(coeffs):
        r0, a, off_wing, off_core = coeffs
        wing_model[r <= r0] = beam_model[r <= r0] + off_core
        wing_model[r > r0] = off_wing + a * (r0**3) / np.power(r[r > r0], 3)

        return np.nansum(ivar * (imap - wing_model) ** 2)

    # Initial guess of r0
    avg_sig = 2.355 * (data_fwhm).to(u.radian).value
    r0 = min(2 * avg_sig, 0.9 * mask_size)
    guess = [r0, 0, 0, 0]
    bounds = [(r0 * 0.5, mask_size), (0, np.inf), (0, 0), (-np.inf, np.inf)]
    res = minimize(_wing_obj, guess, bounds=bounds)
    if not res.success:
        aman.fit_wing = None
        return aman, beam_model
    aman.r0_wing, aman.amp_wing, aman.off_wing, aman.off_core = res.x
    beam_model[r <= aman.r0_wing] += aman.off_core
    beam_model[r > aman.r0_wing] = aman.off_wing + aman.amp_wing * (
        aman.r0_wing**3
    ) / np.power(r[r > aman.r0_wing], 3)

    return aman, beam_model


def _dr4_model(pars, r_use, ell_0, r_0, scatter_pars):
    ell_max, r_c, alpha, off = pars[:4]
    amps = pars[4:]
    model = dr4_beam(r_use, ell_max * ell_0, r_c * r_0, alpha, off, amps, scatter_pars)
    return model


def _dr4_objective(pars, prof, r_use, ell_0, r_0, scatter_pars):
    model = _dr4_model(pars, r_use, ell_0, r_0, scatter_pars)
    return np.nansum((prof - model) ** 2) * 1e10


def fit_dr4_profile(r, rprof, fwhm, d, lmd, sang, corr, eps, r_calc, max_modes=30):
    # Assuming unitful profiles here...
    # TODO: Covariance and scattering
    r_rad = r.to(u.radian).value
    r_calc_rad = r_calc.to(u.radian).value
    prof = rprof.value
    ell_0 = (2 * np.pi * d / lmd).decompose().value
    r_0 = fwhm.to(u.radian).value / 2.355
    par_names = ["ell_max", "r_c", "alpha", "off"]
    par_units = [u.dimensionless_unscaled, r.unit, rprof.unit, rprof.unit]
    par_units_fit = [u.dimensionless_unscaled, u.radian, rprof.unit, rprof.unit]
    guess = [1, 5000, 1, 0]
    bounds = [(0.7, 1.3), (5000, 5000), (0, 0), (0, 0)]

    # Fix the number of modes to half the data points within 5 sigma and setup amps
    n_modes = min(max_modes, int(np.sum(r_rad < 5 * r_0) / 2))
    par_names += [f"amp_{i}" for i in range(n_modes)]
    par_units += [rprof.unit] * n_modes
    par_units_fit += [rprof.unit] * n_modes
    guess += [0] * n_modes
    bounds += [(-np.inf, np.inf)] * n_modes

    # Setup scatter beam
    scatter_pars = {
        "n_terms": 5,
        "lmd": lmd.to(u.m).value,
        "sang": sang.to(u.sr).value,
        "corr": corr.to(u.m).value,
        "eps": np.sqrt(2) * eps.to(u.m).value,
    }

    res = minimize(
        _dr4_objective,
        guess,
        bounds=bounds,
        args=(prof, r_rad, ell_0, r_0, scatter_pars),
        options={"maxcor": 10, "maxfun": 100000},
    )
    if res.success is False:
        return None, None, None
    model = _dr4_model(res.x, r_rad, ell_0, r_0, scatter_pars)
    frac_res = abs(rprof - model) / model
    bad_fit = (frac_res >= 1) * (r_rad > 5 * r_0)
    guess = res.x
    guess[1] = min(5, 0.9 * np.max(r_rad) / r_0)
    if np.sum(bad_fit) > 0:
        r_c = r_rad[int(np.percentile(np.where(bad_fit)[0], 5))]
        guess[1] = r_c / r_0
    bounds[0] = (0.1, 10.0)
    bounds[1] = (guess[1] * 0.7, np.max(r_rad) / r_0)
    bounds[2] = (0, np.inf)
    bounds[3] = (-np.inf, np.inf)
    res = minimize(
        _dr4_objective,
        guess,
        bounds=bounds,
        args=(prof, r_rad, ell_0, r_0, scatter_pars),
        options={"maxcor": 10, "maxfun": 100000},
    )

    model = _dr4_model(res.x, r_calc_rad, ell_0, r_0, scatter_pars)
    model_oto = _dr4_model(res.x, r_rad, ell_0, r_0, scatter_pars)
    pars = res.x
    pars[0] *= ell_0
    pars[1] *= r_0
    params = {
        n: (v * uf).to(un)
        for n, un, uf, v in zip(par_names, par_units, par_units_fit, pars)
    }
    # TODO: include uncertainties here
    mprofile = np.column_stack((r_calc.value, model))
    mprofile_oto = np.column_stack((r.value, model_oto))

    return mprofile, params, mprofile_oto


def fit_bessel_profile(r, rprof, fwhm, d, lmd, sang, corr, eps, r_calc, max_modes=100):
    # TODO: Move model to models.py
    # Assuming unitful profiles here...
    r_rad = r.to(u.radian).value
    r_calc_rad = r_calc.to(u.radian).value
    prof = rprof.value
    ell_0 = (2 * np.pi * d / lmd).decompose().value
    r_0 = fwhm.to(u.radian).value / 2.355
    par_names = []  # ["ell_max", "r_c", "alpha", "off"]
    par_units = []  # [u.dimensionless_unscaled, r.unit, rprof.unit, rprof.unit]
    par_units_fit = []  # [u.dimensionless_unscaled, u.radian, rprof.unit, rprof.unit]

    # Fix the number of modes to half the data points within 5 sigma and setup amps
    n_modes = max_modes  # min(max_modes, int(np.sum(r_rad < 5 * r_0)))
    par_names += [f"amp_{i}" for i in range(n_modes)]
    par_units += [rprof.unit] * n_modes
    par_units_fit += [rprof.unit] * n_modes
    r_msk = r > 0
    amps = np.zeros(n_modes)
    chisq = np.nansum(rprof**2)
    r_use = r_rad[r_msk] * ell_0
    r_calc_use = r_calc_rad * ell_0
    model_oto = np.zeros_like(r_use)
    model = np.zeros_like(r_calc_use)
    for i in range(n_modes):
        term = (jv(i, r_use) / r_use) ** 2
        amp = np.nansum(term * (rprof[r_msk] - model_oto)) / np.nansum(term**2)
        if not np.isfinite(amp):
            continue
        new_chisq = np.nansum((rprof[r_msk] - model_oto - amp * term) ** 2)
        if new_chisq > chisq:
            continue
        amps[i] = amp
        model_oto += amp * term
        with np.errstate(divide="ignore", invalid="ignore"):
            model += amp * (jv(i, r_calc_use) / r_calc_use) ** 2
        chisq = new_chisq
    model_oto = np.insert(model_oto, 0, 1)
    model[r_calc_use == 0] = 1
    pars = amps

    # Setup scatter beam
    scatter_pars = {
        "n_terms": 5,
        "lmd": lmd.to(u.m).value,
        "sang": sang.to(u.sr).value,
        "corr": corr.to(u.m).value,
        "eps": np.sqrt(2) * eps.to(u.m).value,
    }

    # Fit for a symmetric r^-3 wing
    wing_model_oto = model_oto.copy()

    def _wing(coeffs, wmodel, beam_model, r_use):
        r0_msk = r_use == 0
        rc, alpha, off_wing, off_core = coeffs
        wmodel[(r_use <= rc) * ~r0_msk] = beam_model[(r_use <= rc) * ~r0_msk] + off_core
        wmodel[r_use > rc] = off_wing + alpha * (rc**3) / np.power(r_use[r_use > rc], 3)
        wmodel[r_use > rc] += scatter_beam(r_use[r_use > rc], **scatter_pars)
        return wmodel

    def _wing_obj(coeffs):
        wing_model = _wing(coeffs, wing_model_oto, model_oto, r_rad)

        return np.nansum((rprof - wing_model) ** 2)

    # Initial guess of rc
    mask_size = np.max(r_rad)
    rc = min(7 * r_0, 0.9 * mask_size)
    guess = [rc, 0, 0, 0]
    bounds = [(rc * 0.5, mask_size), (0, np.inf), (0, 0), (0, 0)]
    res = minimize(_wing_obj, guess, bounds=bounds)
    if res.success:
        r_c, alpha, off_wing, off_core = res.x
    else:
        r_c, alpha, off_wing, off_core = np.inf, 0, 0, 0

    model_oto = _wing((r_c, alpha, off_wing, off_core), model_oto, model_oto, r_rad)
    model = _wing((r_c, alpha, off_wing, off_core), model, model, r_calc_rad)
    par_names += ["r_c", "alpha", "off_wing", "off_core"]
    par_units += [r.unit, u.dimensionless_unscaled, rprof.unit, rprof.unit]
    par_units_fit += [u.radian, u.dimensionless_unscaled, rprof.unit, rprof.unit]
    pars = np.hstack((amps, (r_c, alpha, off_wing, off_core)))

    params = {
        n: (v * uf).to(un)
        for n, un, uf, v in zip(par_names, par_units, par_units_fit, pars)
    }
    mprofile = np.column_stack((r_calc.value, model))
    mprofile_oto = np.column_stack((r.value, model_oto))

    return mprofile, params, mprofile_oto
