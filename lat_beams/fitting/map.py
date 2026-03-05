import logging

import numpy as np
from astropy import units as u
from scipy.optimize import minimize
from sotodlib.core import AxisManager, IndexAxis, LabelAxis
from sotodlib.tod_ops.filters import logger as flog

from .models import bessel_term, gaussian2d, multipole_decomp, multipole_expansion

flog.setLevel(logging.ERROR)


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
