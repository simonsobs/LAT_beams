"""
Just fitting a single TOD for now, can genralize later (but maybe just do a WITCH interface for that)
Maybe should add priors
"""

import sys
from pixell import enmap

import numpy as np
import so3g
import sotodlib.coords.planets as planets
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from so3g.proj import quat
from sotodlib import core
from sotodlib.tod_ops.filters import (
    fourier_filter,
    high_pass_sine2,
    low_pass_sine2,
    identity_filter,
)
from tqdm.auto import tqdm
from so3g.proj import Ranges

# from . import noise as nn


def get_xieta_src_centered_new(
    ctime,
    az,
    el,
    roll,
    sso_name,
):
    """
    Modified from analyze_bright_ptsrc
    """
    csl = so3g.proj.CelestialSightLine.az_el(
        ctime, az, el, roll=roll, weather="typical"
    )
    q_bore = csl.Q

    # planet position
    planet = planets.SlowSource.for_named_source(sso_name, ctime[int(len(ctime) / 2)])
    ra0, dec0 = planet.pos(ctime)
    q_obj = so3g.proj.quat.rotation_lonlat(ra0, dec0)

    q_total = ~q_bore * q_obj
    xi, eta, _ = quat.decompose_xieta(q_total)

    return xi, eta


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


def pointing_quickfit(
    aman,
    bandpass_range=(None, None),
    fwhm=np.deg2rad(0.5),
    max_rad=None,
    source="mars",
    bin_priors=False,
    show_tqdm=False,
    min_sigma=5,
):
    """
    Modified from analyze_bright_ptsrc
    """
    sigma = fwhm / 2.3548
    if max_rad is None:
        max_rad = 5 * fwhm

    ts = aman.timestamps
    az = aman.boresight.az
    el = aman.boresight.el
    roll = aman.boresight.roll

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
        np.zeros(len(aman.dets.vals), dtype=float) + np.mean(roll),
        [(0, "dets")],
    )
    focal_plane.wrap(
        "reduced_chisq", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")]
    )

    xi, eta = get_xieta_src_centered_new(ts, az, el, roll, source)
    aman.wrap("xi", xi, [(0, "samps")])
    aman.wrap("eta", eta, [(0, "samps")])

    filt = identity_filter()
    if bandpass_range[0] is not None:
        filt *= high_pass_sine2(cutoff=bandpass_range[0])
    if bandpass_range[1] is not None:
        filt *= low_pass_sine2(cutoff=bandpass_range[1])

    def filter_tod(am, signal_name="resid"):
        sig_filt_name = f"{signal_name}_filt"
        am[sig_filt_name] = am[signal_name].copy()
        filt_kw = dict(
            detrend="linear",
            resize="zero_pad",
            axis_name="samps",
            signal_name=sig_filt_name,
            time_name="timestamps",
        )
        am[sig_filt_name] = fourier_filter(am, filt, **filt_kw)
        return am

    def fit_func(x, fit_am):
        xi0, eta0, amp, fwhm, offset = x
        model = (
            gaussian2d((fit_am.xi, fit_am.eta), amp, xi0, eta0, fwhm, fwhm, 0) + offset
        )
        fit_am.resid = (fit_am.signal.ravel() - model).reshape(fit_am.resid.shape)
        fit_am = filter_tod(fit_am, signal_name="resid")
        return np.sum(fit_am.resid * fit_am.resid_filt) * fit_am.wn
        # return (fit_am.resid.ravel().T@fit_am.resid_filt.ravel())#*fit_am.wn

    it = aman.dets.vals
    if show_tqdm:
        it = tqdm(aman.dets.vals)
    for i, det in enumerate(it):
        if show_tqdm:
            it.refresh()
            sys.stderr.flush()
        fit_am = aman.restrict("dets", [det], in_place=False)
        fit_am.wrap("resid", fit_am.signal.copy(), [(0, "dets"), (1, "samps")])
        fit_am.wrap(
            "resid_filt", np.zeros_like(fit_am.signal), [(0, "dets"), (1, "samps")]
        )
        fit_am = filter_tod(fit_am)
        std = np.std(fit_am.resid_filt)
        if std == 0:
            focal_plane.amp[i] = -np.inf
            continue
        fit_am.wrap("wn", 1.0 / np.std(fit_am.resid_filt).item() ** 2)

        max_idx = np.argmax(fit_am.resid_filt[0])
        xi_max = xi[max_idx]
        eta_max = eta[max_idx]
        xi0, eta0 = xi_max, eta_max
        # Bin in xi and eta
        # Should I do this in 2d as a crappy map?
        if bin_priors:
            xi_binned, edges, _ = binned_statistic(
                xi, fit_am.resid_filt[0], bins=int(np.ptp(xi) / (0.1 * fwhm))
            )
            xi_cents = 0.5 * (edges[:-1] + edges[1:])
            xi_binned = gaussian_filter1d(
                xi_binned, (fwhm / 2.3548) / np.mean(np.diff(edges))
            )
            if not np.all(np.isnan(xi_binned)):
                xi0 = xi_cents[np.nanargmax(xi_binned)]
            eta_binned, edges, _ = binned_statistic(
                eta, fit_am.resid_filt[0], bins=int(np.ptp(eta) / (0.1 * fwhm))
            )
            eta_cents = 0.5 * (edges[:-1] + edges[1:])
            eta_binned = gaussian_filter1d(
                eta_binned, (fwhm / 2.3548) / np.mean(np.diff(edges))
            )
            if not np.all(np.isnan(eta_binned)):
                eta0 = eta_cents[np.nanargmax(eta_binned)]
        ptp = np.ptp(fit_am.signal)
        amp = ptp * 3
        msk_samps = np.where((xi - xi0) ** 2 + (eta - eta0) ** 2 < max_rad**2)[
            0
        ].astype(float)
        if len(msk_samps) < 10 and bin_priors:
            xi0, eta0 = xi_max, eta_max
            msk_samps = np.where((xi - xi0) ** 2 + (eta - eta0) ** 2 < max_rad**2)[
                0
            ].astype(float)
        if len(msk_samps) < 10:
            print(f"Not enouth samples flagged for {det}")
            msk_samps = np.arange(aman.samps.count)
        sl = slice(
            int(np.percentile(msk_samps, 5)) + aman.samps.offset,
            int(np.percentile(msk_samps, 95)) + aman.samps.offset,
        )
        fit_am.restrict("samps", sl)

        init_pars = [xi0, eta0, amp, fwhm, 0]
        bounds = [
            (xi0 - max_rad, xi0 + max_rad),
            (eta0 - max_rad, eta0 + max_rad),
            (-1, 10 * amp),
            (fwhm / 4, 4 * fwhm),
            (-ptp, ptp),
        ]

        res = minimize(
            fit_func, init_pars, bounds=bounds, args=(fit_am,), method="Nelder-Mead"
        )

        focal_plane.xi[i] = res.x[0]
        focal_plane.eta[i] = res.x[1]
        focal_plane.amp[i] = res.x[2]
        focal_plane.fwhm[i] = res.x[3]

        if not res.success:
            focal_plane.amp[i] = -np.inf

        # if focal_plane.amp[i] > 0 and (focal_plane.amp[i] < min_sigma*np.std(fit_am.resid_filt)):
        #     focal_plane.amp[i] *= -1

        focal_plane.dist[i] = np.sqrt(
            (focal_plane.xi[i] - xi0) ** 2 + (focal_plane.eta[i] - eta0) ** 2
        )

        delta_xi = xi - focal_plane.xi[i]
        delta_eta = eta - focal_plane.eta[i]

        # Lets calculate hits
        xi_msk = np.abs(delta_xi) <= 3 * focal_plane.fwhm[i] / 2.3548
        eta_msk = np.abs(delta_eta) <= 3 * focal_plane.fwhm[i] / 2.3548
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
        focal_plane.az[i] = np.sum(aman.boresight.az * weights) / tot_weight
        focal_plane.el[i] = np.sum(aman.boresight.el * weights) / tot_weight

        # Chisq
        focal_plane.reduced_chisq[i] = res.fun / len(res.x)

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


def solid_angle(az, el, beam, cent, min_sigma, smooth=0):
    """Compute the integrated solid angle of a beam map.

    return value is in steradians  (sr)
    """
    # convert from arcsec to rad
    az = np.deg2rad(az / 3600)
    el = np.deg2rad(el / 3600)
    norm = beam[cent]
    if smooth > 0:
        smoothed = gaussian_filter(beam, sigma=smooth)
        norm = smoothed[cent]
    integrand = beam / norm
    integrand[integrand < np.exp(-0.5 * (min_sigma**2))] = 0
    # perform the solid angle integral
    integral = np.trapz(np.trapz(integrand, el, axis=0), az, axis=0)
    return integral


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
    data_solid_angle = solid_angle(x, y, imap, c, min_sigma, (data_fwhm / 2.355) / res)
    model_solid_angle = solid_angle(x, y, model, c, min_sigma)

    return (
        popt,
        perr,
        model,
        data_fwhm,
        data_solid_angle,
        model_solid_angle,
        r,
        rprof,
        mprof,
    )
