import logging
from typing import Optional

import numpy as np
from astropy import units as u
from pixell.enmap import ndmap
from scipy.optimize import minimize
from sotodlib.core import AxisManager, IndexAxis, LabelAxis
from sotodlib.tod_ops.filters import logger as flog

from ..core import gaussian2d, multipole_decomp, multipole_expansion

flog.setLevel(logging.ERROR)


def fit_gauss_map(
    imap: ndmap,
    ivar: ndmap,
    posmap: ndmap,
    guess: AxisManager,
    map_units: str = "pW",
    force_sym: bool = False,
    mask_size: float = -1,
):
    """
    Fit 2d Gaussian to input map.
    Note that only keywod arguments are shown below.
    See `FitMap` for the rest.

    Arguments
    ---------
    force_sym: bool, default: False
        It true fit a symmetric beam.
        Both FWHMs will still be in the output,
        but they will have the same value.
    mask_size : float, default: -1
        If this is >0 then a mask will be applies to ivar
        such that only data within `mask_size*(guess.fwhm_xi + guess.fwhm_eta)/2`
        of `(guess.xi0, guess.eta0)` is used in the fit.

    Returns
    -------
    fit_params : Optional[AxisManager]
        Parameters are:

        - `amp`: Amplitude of the beam
        - `fwhm_xi`: FWHM in xi
        - `fwhm_eta`: FWHM in eta
        - `xi0`: Center of beam in xi
        - `eta0`: Center of beam in eta
        - `phi`: Rotation of the beam
        - `off`: DC offset of the beam

        Note that all positional parameters are in radians.
        Returns `None` if the fit failed.
    model : Optional[NDArray]
        The model evaluated with the fit parameters.
        Returns `None` if the fit failed.
    """
    y, x = posmap
    x0 = [
        guess.xi0,
        guess.eta0,
        guess.off,
        guess.amp,
        guess.fwhm_xi,
        guess.fwhm_eta,
        guess.phi,
    ]
    bounds = [
        [
            np.min(x) - guess.fwhm_xi,
            np.min(y) - guess.fwhm_eta,
            -5 * np.max(np.abs(imap)),
            0,
            guess.fwhm_xi / 3,
            guess.fwhm_eta / 3,
            0,
            # r0,
            # 0,
        ],
        [
            np.max(x) + guess.fwhm_xi,
            np.max(y) + guess.fwhm_eta,
            5 * np.max(imap),
            5 * np.max(imap),
            guess.fwhm_xi * 3,
            guess.fwhm_eta * 3,
            2 * np.pi,
        ],
    ]
    map_unit = u.Unit(map_units)
    par_names = [
        "xi0",
        "eta0",
        "off",
        "amp",
        "fwhm_xi",
        "fwhm_eta",
        "phi",
    ]  # , "wing_r0", "wing_amp"]
    par_units = [
        u.radian,
        u.radian,
        map_unit,
        map_unit,
        u.radian,
        u.radian,
        u.radian,
    ]  # , u.radian, map_unit]  # type: ignore
    if force_sym:
        x0 = x0[:-2]
        bounds[0] = bounds[0][:-2]
        bounds[1] = bounds[1][:-2]
    bounds = [(lb, ub) for lb, ub in zip(*bounds)]

    # Mask out things too far from the starting center
    if mask_size > 0:
        r = np.sqrt((x - x0[0]) ** 2 + (y - x0[1]) ** 2)
        ivar = ivar.copy()
        ivar[r > mask_size * 0.5 * (guess.fwhm_xi + guess.fwhm_eta)] = 0

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

    res = minimize(_objective, x0, bounds=bounds)
    if not res.success:
        return None, None

    # Convert to aman
    aman = AxisManager()
    dx, dy, off, amp, fwhm_xi, fwhm_eta, phi = pars = _to_pars(res.x)
    for n, un, v in zip(par_names, par_units, pars):
        aman.wrap(n, v * un)
    model = gaussian2d(posmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, off)

    return aman, model


def fit_multipole_map(
    imap: ndmap,
    ivar: ndmap,
    posmap: ndmap,
    guess: AxisManager,
    map_units: str = "pW",
    base_beam: Optional[ndmap] = None,
    n_multipoles: int = 5,
):
    """
    Fit the multipole expansion of a input model to a map.

    Arguments
    ---------
    base_beam : Optional[ndmap], default: None
        The base beam model to take the multipole expansion of.
        If `None` then this will be computend by passing the `guess` parameters to
        `guassian2d` (but with the amplitude set to 1).
    n_multipoles : int, default: 5
        The number of multipoles to fit.
        0 will just be the monopole, 1 the dipole, and so on.
    Returns
    -------
    fit_params : AxisManager
        The only element is an array called `amps` with shape
        `(n_multipoles, 2)` where each row containes the amplitudes
        for a given multipole, the first collumn is the `cos` terms,
        and the second `collumn` is the `sin` terms.
        The `AxisManager` will contain axes called `multipoles` and `term`
        for this array.
    model : NDArray
        The model evaluated with the fit parameters.
    """
    if base_beam is None:
        base_beam = gaussian2d(
            posmap,
            1,
            guess.xi0,
            guess.eta0,
            guess.fwhm_xi,
            guess.fwhm_eta,
            guess.phi,
            guess.off,
        )
    y, x = posmap
    theta = np.arctan2(
        y - guess.eta0.to(u.radian).value, x - guess.eta0.to(u.radian).value
    )

    # Compute model
    if n_multipoles == 0:
        amps = np.array([[guess.amp.value, 0]])
    else:
        amps = multipole_decomp(base_beam, imap, ivar, n_multipoles, theta, True)
    model = multipole_expansion(base_beam, amps, theta)

    # Convert to aman
    m_units = u.Unit(map_units)
    aman = AxisManager()
    mp_ax = IndexAxis("multipoles", n_multipoles)
    sc_ax = LabelAxis("term", ["cos", "sin"])
    aman.wrap("amps", amps * m_units, [(0, mp_ax), (1, sc_ax)])

    return aman, model


def gaussian2d_from_aman(posmap, aman):
    if "gaussian" in aman._fields:
        aman = aman.gaussian
    return gaussian2d(
        posmap,
        aman.amp.value,
        aman.xi0.to(u.radian).value,
        aman.eta0.to(u.radian).value,
        aman.fwhm_xi.to(u.radian).value,
        aman.fwhm_eta.to(u.radian).value,
        aman.phi.to(u.radian).value,
        aman.off.value,
    )


def gaussian2d_multipoles_from_aman(posmap, aman):
    base_beam = gaussian2d_from_aman(posmap, aman.gaussian)
    base_beam -= aman.gaussian.off.value
    base_beam /= aman.gaussian.amp.value
    y, x = posmap
    theta = np.arctan2(
        y - aman.gaussian.eta0.to(u.radian).value,
        x - aman.gaussian.eta0.to(u.radian).value,
    )
    return multipole_expansion(base_beam, aman.gauss_multipole.amps.value, theta)
