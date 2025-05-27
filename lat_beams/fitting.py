"""
Just fitting a single TOD for now, can genralize later (but maybe just do a WITCH interface for that)
Maybe should add priors
"""

import sys
import warnings
from copy import deepcopy

import numpy as np
import so3g
import sotodlib.coords.planets as planets
from astropy.convolution import (
    Gaussian1DKernel,
    Gaussian2DKernel,
    convolve,
    convolve_fft,
)
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, minimize
from scipy.stats import binned_statistic, binned_statistic_2d
from so3g.proj import Ranges, quat
from sotodlib import core
from sotodlib.core.context import AxisManager
from sotodlib.tod_ops.fft_ops import (
    RFFTObj,
    find_inferior_integer,
    find_superior_integer,
)
from sotodlib.tod_ops.filters import (
    fourier_filter,
    high_pass_sine2,
    identity_filter,
    low_pass_sine2,
)
from tqdm.auto import tqdm
from typing_extensions import Optional, cast


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


# TOOO: Add gradient function
def gaussian2d(xieta, a, xi0, eta0, fwhm_xi, fwhm_eta, phi, off):
    """
    Stolen from analyze_bright_ptsrc

    Simulate a time stream with an Gaussian beam model
    Args
    ------
    xi, eta: cordinates in the detector's system
    a: float
        amplitude of the Gaussian beam model
    xi0, eta0: float, float
        center position of the Gaussian beam model
    fwhm_xi, fwhm_eta, phi: float, float, float
        fwhm along the xi, eta axis (rotated)
        and the rotation angle (in radians)

    Ouput:
    ------
    sim_data: 1d array of float
        Time stream at sampling points given by xieta
    """
    xi, eta = xieta
    xi_sft = xi - xi0
    eta_sft = eta - eta0
    xi_rot = xi_sft * np.cos(phi) - eta_sft * np.sin(phi)
    eta_rot = xi_sft * np.sin(phi) + eta_sft * np.cos(phi)
    factor = 2 * np.sqrt(2 * np.log(2))
    xi_coef = -0.5 * (xi_rot) ** 2 / (fwhm_xi / factor) ** 2
    eta_coef = -0.5 * (eta_rot) ** 2 / (fwhm_eta / factor) ** 2
    sim_data = a * np.exp(xi_coef + eta_coef)
    return sim_data + off


def gaussian2d_deriv(xieta, a, xi0, eta0, fwhm_xi, fwhm_eta, phi, off):
    factor = 2 * np.sqrt(2 * np.log(2))
    xi, eta = xieta
    gauss = gaussian2d(xieta, a, xi0, eta0, fwhm_xi, fwhm_eta, phi, off)
    xi_sft = xi - xi0
    eta_sft = eta - eta0

    da = gauss - off
    dxi0 = (
        -1
        * (factor**2)
        * da
        * (
            (-1 * eta_sft) * (fwhm_xi**2 - fwhm_eta**2) * np.sin(phi) * np.cos(phi)
            + (-1 * xi_sft)
            * ((fwhm_xi * np.sin(phi)) ** 2 + (fwhm_eta * np.cos(phi)) ** 2)
        )
        / ((fwhm_xi * fwhm_eta) ** 2)
    )
    deta0 = (
        -1
        * (factor**2)
        * da
        * (
            (-1 * xi_sft) * (fwhm_xi**2 - fwhm_eta**2) * np.sin(phi) * np.cos(phi)
            + (-1 * eta_sft)
            * ((fwhm_xi * np.cos(phi)) ** 2 + (fwhm_eta * np.sin(phi)) ** 2)
        )
        / ((fwhm_xi * fwhm_eta) ** 2)
    )
    dfwhm_xi = (
        (factor**2)
        * da
        * (((-1 * xi_sft) * np.cos(phi) - (-1 * eta_sft) * np.sin(phi)) ** 2)
        / (fwhm_xi**3)
    )
    dfwhm_eta = (
        (factor**2)
        * da
        * (((-1 * xi_sft) * np.sin(phi) + (-1 * eta_sft) * np.cos(phi)) ** 2)
        / (fwhm_eta**3)
    )
    dphi = (
        -1
        * (factor**2)
        * da
        * (fwhm_xi**2 + fwhm_eta**2)
        * ((xi_sft) * np.sin(phi) - ((-1 * eta_sft) * np.cos(phi)))
        * ((-1 * xi_sft) * np.cos(phi) + (eta_sft) * np.sin(phi))
        / ((fwhm_xi * fwhm_eta) ** 2)
    )
    doff = np.ones_like(xi)

    dgauss = np.vstack([da, dxi0, deta0, dfwhm_xi, dfwhm_eta, dphi, doff])

    return gauss, dgauss


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

    return focal_plane


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


def fit_tod_pointing(
    aman: AxisManager,
    bandpass_range: tuple[Optional[float], Optional[float]] = (None, None),
    fwhm: float = np.deg2rad(0.5),
    max_rad: Optional[float] = None,
    source: str = "mars",
    bin_priors: bool = True,
    bin_2d: bool = True,
    multistep: bool = False,
    n_err: float = 5,
    pos_priors: Optional[NDArray[np.floating]] = None,
    show_tqdm: bool = False,
) -> AxisManager:
    """
    Fit detector offsets from a TOD of a source observation.

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

        multistep: If True fit using a multiple rounds of the L-BFGS-B optimizer.
                   In the first round we hold the fwhm fixed.
                   In the second the detector offsets are held fixed based on the first round and fwhm is fit.
                   In the third all parameters are fit and the bounds are set based on the errors from the previous rounds.
                   If False fit with a single round of Nelder-Mead.

        n_err: How many times to errors to use for bounds in multistep fitting.

        pos_priors: Positional priors. Pass None to not use.
                    If using pass in a (ndets, 2) array where each row is the (xi, eta) to
                    start the fit for that detector at. To disable for only some detectors set
                    the row to (nan, nan).

        show_tqdm: If True show a progress bar.

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

    xi, eta = get_xieta_src_centered(
        np.array(aman.timestamps),
        np.array(aman.boresight.az),
        np.array(aman.boresight.el),
        np.array(aman.boresight.roll),
        source,
    )
    aman.wrap("xi", xi, [(0, "samps")])
    aman.wrap("eta", eta, [(0, "samps")])

    filt = identity_filter()
    if bandpass_range[0] is not None:
        filt *= high_pass_sine2(cutoff=bandpass_range[0])
    if bandpass_range[1] is not None:
        filt *= low_pass_sine2(cutoff=bandpass_range[1])

    def filter_tod(am, signal_name="resid", rfft=None):
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

    def fit_func(x, fit_am, rfft):
        xi0, eta0, amp, fwhm, offset = x
        model = gaussian2d(
            (fit_am.xi, fit_am.eta), amp, xi0, eta0, fwhm, fwhm, 0, offset
        )
        fit_am.resid = (fit_am.signal.ravel() - model).reshape(fit_am.resid.shape)
        fit_am = filter_tod(fit_am, signal_name="resid", rfft=rfft)
        return np.sum(fit_am.resid * fit_am.resid_filt) * fit_am.wn

    it = np.array(aman.dets.vals)
    if show_tqdm:
        it = tqdm(np.array(aman.dets.vals))
    for i, det in enumerate(it):
        if show_tqdm:
            sys.stderr.flush()
        fit_am = aman.restrict("dets", [det], in_place=False)
        fit_am.wrap("resid", fit_am.signal.copy(), [(0, "dets"), (1, "samps")])
        fit_am.wrap(
            "resid_filt", np.zeros_like(fit_am.signal), [(0, "dets"), (1, "samps")]
        )
        fit_am = filter_tod(fit_am)
        std = np.std(np.array(fit_am.resid_filt))
        if std == 0:
            focal_plane.amp[i] = -np.inf
            continue
        fit_am.wrap("wn", 1.0 / std.item() ** 2)

        max_idx = np.argmax(np.array(fit_am.resid_filt[0]))
        xi_max = xi[max_idx]
        eta_max = eta[max_idx]
        xi0, eta0 = float(xi_max), float(eta_max)
        _bin_priors = bin_priors
        if np.all(np.isfinite(pos_priors[i])):
            xi0, eta0 = pos_priors[i]
            _bin_priors = False

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

        if multistep:
            ftol = 2.220446049250313e-09  # default
            _bounds = deepcopy(bounds)
            _bounds[3] = (fwhm, fwhm)
            _bounds[0] = (np.min(xi), np.max(xi))
            _bounds[1] = (np.min(eta), np.max(eta))
            res = minimize(
                fit_func,
                init_pars,
                bounds=_bounds,
                args=(fit_am, rfft),
                method="L-BFGS-B",
            )
            init_pars = res.x
            bounds[0] = (res.x[0] - max_rad, res.x[0] + max_rad)
            bounds[1] = (res.x[1] - max_rad, res.x[1] + max_rad)
            if res.success:
                errs = [
                    np.sqrt(max(1, abs(res.fun)) * ftol * res.hess_inv(eye)[i])
                    for i, eye in enumerate(np.eye(len(res.x)))
                ]
                _bounds = deepcopy(bounds)
                _bounds[0] = (res.x[0] - 1e-9, res.x[0] + 1e-9)
                _bounds[1] = (res.x[1] - 1e-9, res.x[1] + 1e-9)
                res = minimize(
                    fit_func,
                    res.x,
                    bounds=_bounds,
                    args=(fit_am, rfft),
                    method="L-BFGS-B",
                )
                if res.success:
                    errs_new = [
                        np.sqrt(max(1, abs(res.fun)) * ftol * res.hess_inv(eye)[i])
                        for i, eye in enumerate(np.eye(len(res.x)))
                    ]
                    errs[2:] = errs_new[2:]
                    _bounds = [
                        (p - n_err * e, p + n_err * e) for p, e in zip(res.x, errs)
                    ]
                    res = minimize(
                        fit_func,
                        res.x,
                        bounds=_bounds,
                        args=(fit_am, rfft),
                        method="L-BFGS-B",
                    )
        else:
            res = minimize(
                fit_func,
                init_pars,
                bounds=bounds,
                args=(fit_am, rfft),
                method="Nelder-Mead",
            )

        focal_plane.xi[i] = res.x[0]
        focal_plane.eta[i] = res.x[1]
        focal_plane.amp[i] = res.x[2]
        focal_plane.fwhm[i] = res.x[3]

        if not res.success:
            focal_plane.amp[i] = -np.inf

        focal_plane.dist[i] = np.sqrt(
            (np.array(focal_plane.xi[i]) - xi0) ** 2
            + (np.array(focal_plane.eta[i]) - eta0) ** 2
        )
        delta_xi = xi - np.array(focal_plane.xi[i])
        delta_eta = eta - np.array(focal_plane.eta[i])

        # Lets calculate hits
        xi_msk = np.abs(delta_xi) <= 3 * np.array(focal_plane.fwhm[i]) / 2.3548
        eta_msk = np.abs(delta_eta) <= 3 * np.array(focal_plane.fwhm[i]) / 2.3548
        hits = Ranges.from_mask(xi_msk * eta_msk)
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
            focal_plane.amp[i] = -np.inf
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


def solid_angle(az, el, beam, cent, r1, r2, smooth=0):
    """Compute the integrated solid angle of a beam map.

    return value is in steradians  (sr)
    """
    r = np.sqrt(az**2 + el**2)
    # convert from arcsec to rad
    az = np.deg2rad(az / 3600)
    el = np.deg2rad(el / 3600)
    norm = beam[cent]
    if smooth > 0:
        smoothed = gaussian_filter(beam, sigma=smooth)
        norm = smoothed[cent]
    integrand = beam / norm
    # integrand[integrand < np.exp(-0.5 * (min_sigma**2))] = 0
    # perform the solid angle integral
    _integrand = integrand.copy()
    _integrand[r > r1] = 0
    integral_inner = np.trapz(np.trapz(_integrand, el, axis=0), az, axis=0)
    
    _integrand = integrand.copy()
    _integrand[(r < r1) + (r > r2)] = 0
    integral_outer = np.trapz(np.trapz(_integrand, el, axis=0), az, axis=0)
    return integral_inner - integral_outer


def fit_gauss_beam(imap, ivar, pixmap, cent, min_sigma=3):
    """
    Fit 2d Gaussian to input map.
    Based on Tommy's code from 2024 F2F tutorial.

    Arguments
    ---------
    imap : (ny, nx) enmap
        Input map.
    ivar : (ny, nx) enmap
        Inverse-variance map.
    pixmap : (2, ny, nx) array
        X and Y pixel indices for each pixel.

    Returns
    -------
    amp : float
        Amplitude of Gaussian.
    shift_x : float
        X shift needed to center the Gaussian in middle of map
    shift_y : float
        Y shift needed to center the Gaussian in middle of map
    fwhm
        Fitted FWHM of Gaussian
    """
    res = imap.wcs.wcs.cdelt[1] * (60 * 60)
    ny, nx = imap.shape[-2:]

    sigma = np.sqrt(ivar)
    sigma = np.divide(1, sigma, where=sigma != 0)

    # Set to numerically large value.
    sigma[ivar == 0] = sigma[~(ivar == 0)].max() * 1e5

    guess = [
        imap[cent[0], cent[1]],
        pixmap[0][cent[0], cent[1]],
        pixmap[1][cent[0], cent[1]],
        60 / res,
        60 / res,
        0,
        0,
    ]

    #  amp, x0, y0, fwhm_x, fwhm_y, phi
    bounds = (
        [0, 0, 20 / res, 20 / res, 0, 0, -np.inf],
        [np.inf, nx, ny, 300 / res, 300 / res, 2 * np.pi, np.inf],
    )

    pixmap = (pixmap[0].ravel().astype(float), pixmap[1].ravel().astype(float))

    try:
        popt, pcov = curve_fit(
            gaussian2d,
            pixmap,
            imap.ravel(),
            p0=guess,
            sigma=sigma.ravel(),
            bounds=bounds,
        )
    except RuntimeError:
        return None
    perr = np.sqrt(np.diag(pcov))

    model = gaussian2d(pixmap, *popt).reshape(imap.shape)
    c = np.unravel_index(np.argmax(model, axis=None), model.shape)
    if popt[0] < min_sigma * perr[0] or model[c] == 0:
        return None

    # convert units of pixels to arcsecs
    popt[1:5] *= res
    perr[1:5] *= res

    # Get FWHM from data
    rprof = radial_profile(imap, c[::-1])
    mprof = radial_profile(model, c[::-1])
    r = np.linspace(0, len(rprof), len(rprof)) * res
    data_fwhm = get_fwhm_radial_bins(r, rprof, interpolate=True)

    # Get solid angles
    y = np.linspace(-imap.shape[0] * res / 2, imap.shape[0] * res / 2, imap.shape[0])
    x = np.linspace(-imap.shape[1] * res / 2, imap.shape[1] * res / 2, imap.shape[1])
    data_solid_angle_meas = solid_angle(x, y, imap, c, min_sigma * (data_fwhm / 2.355), 2*min_sigma * (data_fwhm / 2.355), (data_fwhm / 2.355) / res)
    model_solid_angle_meas = solid_angle(x, y, model, c, min_sigma * (data_fwhm / 2.355), 2*min_sigma * (data_fwhm / 2.355))
    model_solid_angle_true = 2*np.pi*(np.deg2rad(popt[3]/3600) * np.deg2rad(popt[4]/3600))
    data_solid_angle_corr = data_solid_angle_meas*model_solid_angle_true/model_solid_angle_meas

    return (
        popt,
        perr,
        model,
        data_fwhm,
        data_solid_angle_meas,
        data_solid_angle_corr,
        model_solid_angle_meas,
        model_solid_angle_true,
        r,
        rprof,
        mprof,
    )
