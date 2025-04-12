"""
Just fitting a single TOD for now, can genralize later (but maybe just do a WITCH interface for that)
Maybe should add priors
"""

import sys

import numpy as np
import so3g
import sotodlib.coords.planets as planets
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.stats import binned_statistic
from so3g.proj import quat
from sotodlib import core
from sotodlib.tod_ops.filters import fourier_filter, high_pass_sine2, low_pass_sine2
from tqdm.auto import tqdm
from so3g.proj import Ranges

# from . import noise as nn


def invsafe(matrix, thresh: float = 1e-14):
    """
    Safe SVD based psuedo-inversion of the matrix.
    This zeros out modes that are too small when inverting.
    Use with caution in cases where you really care about what the inverse is.
    """
    u, s, v = np.linalg.svd(matrix, False)
    s_inv = np.array(np.where(np.abs(s) < thresh * np.max(s), 0, 1 / s))

    return np.dot(np.transpose(v), np.dot(np.diag(s_inv), np.transpose(u)))


def invscale(matrix, thresh: float = 1e-14):
    """
    Invert and rescale a matrix by the diagonal.
    This uses `invsafe` for the inversion.

    """
    diag = np.diag(matrix)
    vec = np.array(np.where(diag != 0, 1.0 / np.sqrt(np.abs(diag)), 1e-10))
    mm = np.outer(vec, vec)

    return mm * invsafe(mm * matrix, thresh)


# TODO: ellipticity
def gauss_grad(x, y, x0, y0, amp, fwhm):
    dx = x - x0
    dy = y - y0
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    var = sigma**2
    gauss = amp * np.exp(-0.5 * (dx**2 + dy**2) / (var))

    dfdx = -(dx / var) * gauss
    dfdy = -(dy / var) * gauss
    dady = gauss / amp
    dsdy = ((dx**2 + dy**2) / sigma**3) * gauss
    grad = np.array([dfdx, dfdy, dady, dsdy])

    return gauss, grad


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


def gaussian2d(xi, eta, a, xi0, eta0, fwhm_xi, fwhm_eta, phi):
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
    xi_rot = xi * np.cos(phi) - eta * np.sin(phi)
    eta_rot = xi * np.sin(phi) + eta * np.cos(phi)
    factor = 2 * np.sqrt(2 * np.log(2))
    xi_coef = -0.5 * (xi_rot - xi0) ** 2 / (fwhm_xi / factor) ** 2
    eta_coef = -0.5 * (eta_rot - eta0) ** 2 / (fwhm_eta / factor) ** 2
    sim_data = a * np.exp(xi_coef + eta_coef)
    return sim_data


def objective(aman, pars, filter_tod):
    npar = len(pars)
    chisq = np.array(0)
    grad = np.zeros(npar)
    curve = np.zeros((npar, npar))

    pred_dat, grad_dat = gauss_grad(aman.xi, aman.eta, *pars)

    aman.resid = aman.signal - pred_dat
    aman = filter_tod(aman, signal_name="resid")
    chisq = np.sum(aman.resid * aman.resid_filt)
    grad_filt = np.zeros_like(grad_dat)
    for i in range(npar):
        aman.grad_buf[0] = grad_dat[i]
        aman = filter_tod(aman, signal_name="grad_buf")
        grad_filt[i] = aman.grad_buf_filt.copy().ravel()
    grad_filt = np.reshape(grad_filt, (npar, -1))
    grad_dat = np.reshape(grad_dat, (npar, -1))
    resid = aman.resid.ravel()
    grad = np.dot(grad_filt, np.transpose(resid))
    curve = np.dot(grad_filt, np.transpose(grad_dat))

    return chisq, grad, curve


def prior_pars(pars, priors):
    prior_l, prior_u = priors
    at_edge_l = pars <= prior_l
    at_edge_u = pars >= prior_u
    pars = np.where(at_edge_l, prior_l, pars)
    pars = np.where(at_edge_u, prior_u, pars)

    return pars


def lm_fitter(aman, filter_tod, init_pars, bounds, max_iters=20, chitol=1e-5):
    priors = np.array(
        [[bound[i] for bound in bounds] for i in range(2)]
    )  # Convert from scipy opt bounds to flat priors
    pars = prior_pars(np.array(init_pars), priors)
    chisq, grad, curve = objective(aman, pars, filter_tod)
    errs = np.inf + np.zeros_like(pars)
    delta_chisq = np.inf
    lmd = 0
    i = 0

    for i in range(max_iters):
        if delta_chisq < chitol:
            break
        curve_use = curve + (lmd * np.diag(np.diag(curve)))
        # Get the step
        step = np.dot(invscale(curve_use), grad)
        new_pars = prior_pars(pars + step, priors)
        # Get errs
        errs = np.sqrt(np.diag(invscale(curve_use)))
        # Now lets get an updated model
        new_chisq, new_grad, new_curve = objective(aman, new_pars, filter_tod)
        new_delta_chisq = chisq - new_chisq

        if new_delta_chisq > 0:
            pars, chisq, grad, curve, delta_chisq = (
                new_pars,
                new_chisq,
                new_grad,
                new_curve,
                new_delta_chisq,
            )
            if lmd < 0.2:
                lmd = 0
            else:
                lmd /= np.sqrt(2)
        else:
            if lmd == 0:
                lmd = 1
            else:
                lmd *= 2

    return pars, errs, i, delta_chisq


def pointing_quickfit(
    aman,
    bandpass_range=(None, None),
    fwhm=np.deg2rad(0.5),
    max_rad=None,
    source="mars",
    bin_priors=False,
    show_tqdm=False,
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
    focal_plane.wrap("roll", np.zeros(len(aman.dets.vals) + roll, dtype=float), [(0, "dets")])

    xi, eta = get_xieta_src_centered_new(ts, az, el, roll, source)
    aman.wrap("xi", xi, [(0, "samps")])
    aman.wrap("eta", eta, [(0, "samps")])

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
        if bandpass_range[0] is not None:
            highpass = high_pass_sine2(cutoff=bandpass_range[0])
            am[sig_filt_name] = fourier_filter(am, highpass, **filt_kw)
        if bandpass_range[1] is not None:
            lowpass = low_pass_sine2(cutoff=bandpass_range[1])
            am[sig_filt_name] = fourier_filter(am, lowpass, **filt_kw)
        return am

    def fit_func(x, fit_am):
        xi0, eta0, amp, fwhm = x
        model = gaussian2d(fit_am.xi, fit_am.eta, amp, xi0, eta0, fwhm, fwhm, 0)
        fit_am.resid = (fit_am.signal.ravel() - model).reshape(fit_am.resid.shape)
        fit_am = filter_tod(fit_am, signal_name="resid")
        return np.sum(fit_am.resid * fit_am.resid_filt)

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
        fit_am.wrap(
            "grad_buf", np.zeros_like(fit_am.signal), [(0, "dets"), (1, "samps")]
        )
        fit_am.wrap(
            "grad_buf_filt", np.zeros_like(fit_am.signal), [(0, "dets"), (1, "samps")]
        )
        fit_am = filter_tod(fit_am)

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
            xi0 = xi_cents[np.nanargmax(xi_binned)]
            eta_binned, edges, _ = binned_statistic(
                eta, fit_am.resid_filt[0], bins=int(np.ptp(eta) / (0.1 * fwhm))
            )
            eta_cents = 0.5 * (edges[:-1] + edges[1:])
            eta_binned = gaussian_filter1d(
                eta_binned, (fwhm / 2.3548) / np.mean(np.diff(edges))
            )
            eta0 = eta_cents[np.nanargmax(eta_binned)]
        amp = np.ptp(fit_am.signal) * 3
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
        sl = slice(int(np.percentile(msk_samps, 5)), int(np.percentile(msk_samps, 95)))
        fit_am.restrict("samps", sl)

        init_pars = [xi0, eta0, amp, fwhm]
        bounds = [
            (xi0 - max_rad, xi0 + max_rad),
            (eta0 - max_rad, eta0 + max_rad),
            (-1, 10 * amp),
            (fwhm / 4, 4 * fwhm),
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
        xi_weights = np.exp(-.5*((delta_xi/sigma)**2))/(sigma*np.sqrt(2*np.pi)) 
        eta_weights = np.exp(-.5*((delta_eta/sigma)**2))/(sigma*np.sqrt(2*np.pi)) 
        weights = xi_weights * eta_weights
        tot_weight = np.sum(weights)
        focal_plane.az[i] = np.sum(aman.boresight.az*weights)/tot_weight
        focal_plane.el[i] = np.sum(aman.boresight.el*weights)/tot_weight


    return focal_plane
