"""
Functions for fitting beam models to a map.

All functions will have the following standardized interface
which is defined by the `FitMap` protocol in this module.
Fitting functions should follow the naming convention `fit_{MODEL}_map`
where `{MODEL}` is a one word description of the model being fit.
"""

import logging
from typing import Optional, Protocol

import numpy as np
from astropy import units as u
from pixell.enmap import ndmap
from scipy.optimize import minimize
from scipy.linalg import lstsq
from sotodlib.core import AxisManager, IndexAxis, LabelAxis
from sotodlib.tod_ops.filters import logger as flog
from astropy.nddata import block_reduce, block_replicate

from .models import (
    bessel_term,
    gaussian2d,
    multipole_decomp,
    multipole_expansion,
    multipole,
    bessel_beam,
)

flog.setLevel(logging.ERROR)


class FitMap(Protocol):
    def __call__(
        self,
        imap: ndmap,
        ivar: ndmap,
        posmap: ndmap,
        guess: AxisManager,
        map_units: str = "pW",
        **kwargs
    ) -> tuple[Optional[AxisManager], Optional[ndmap]]:
        """
        Function to fit a beam model to a map.

        Arguments
        ---------
        imap : ndmap
            Input map to fit with shape `(nx, ny)`.
        ivar : ndmap
            Inverse-variance map for `imap` with shape `(nx, ny)`.
        posmap : ndmap
            Position map in radians for `imap`.
            First element is eta and the second is xi.
            Should have shape `(2, nx, ny)`.
        guess : AxisManager
            `AxisManager` containing parameters that are useful as a starting point.
            See `make_guess` for the expected parameters.
        map_units : str, default: 'pW'
            The units of the map.
            Should be a string that astromy units understands.
        **kwargs
            Additional arguments for the specific fitting function.

        Returns
        -------
        fit_params : Optional[AxisManager]
            The fit parameters.
            See individual function docstrings for detail.
            Returns `None` if the fit failed.
        model : Optional[NDArray]
            The model evaluated with the fit parameters.
            Returns `None` if the fit failed.
        """


def make_guess(
    amp: float = 1,
    fwhm_xi: float = 2 / 60,
    fwhm_eta: float = 2 / 60,
    xi0: float = 0,
    eta0: float = 0,
    phi: float = 0,
    off: float = 0,
) -> AxisManager:
    """
    Helper function to make the initial guess `AxisManager`.
    Note that all arguments will be scalars in the output
    and all positional parameters are in radians.

    Arguments
    ---------
    amp : float, default: 1
        Amplitude of the beam.
    fwhm_xi : float, default: 2/60
        FWHM in xi.
    fwhm_eta : float, default: 2/60
        FWHM in eta.
    xi0 : float, default: 0
        Center of beam in xi.
    eta0 : float, default: 0
        Center of beam in eta.
    phi : float, default: 0
        Rotation of the beam.
    off : float, default: 0
        DC offset of the beam.

    Returns
    -------
    guess : AxisManager
        `AxisManager` with the guess parameters.
    """
    guess_dict = locals()
    guess = AxisManager()
    for n, v in guess_dict.items():
        guess.wrap(n, v)
    return guess


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
    map_units = u.Unit(map_units)
    par_names = ["xi0", "eta0", "off", "amp", "fwhm_xi", "fwhm_eta", "phi"]
    par_units = [u.radian, u.radian, map_units, map_units, u.radian, u.radian, u.radian]  # type: ignore
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
    map_units = u.Unit(map_units)
    aman = AxisManager()
    mp_ax = IndexAxis("multipoles", n_multipoles)
    sc_ax = LabelAxis("term", ["cos", "sin"])
    aman.wrap("amps", amps * map_units, [(0, mp_ax), (1, sc_ax)])

    return aman, model


def fit_bessel_map(
    imap: ndmap,
    ivar: ndmap,
    posmap: ndmap,
    guess: AxisManager,
    map_units: str = "pW",
    n_bessel: int = 10,
    n_multipoles: int = 5,
    d: u.Quantity = 6 * u.m,
    lmd: u.Quantity = 90 * u.GHz,
    mask_size: float = np.inf,
    data_fwhm: float = np.inf,
    fit_wing: bool = True,
):
    r"""
    Fit a model of squared bessel functions with multipole expansions to the map.
    See `models.bessel_beam` for details on the model.


    Arguments
    ---------
    n_bessel : int, default: 10
        The number of squared bessel functions to fit.
    n_multipoles : int, default: 5
        The number of multipoles to fit.
        0 will just be the monopole, 1 the dipole, and so on.
    d : u.Quantity, default: 6 * u.m
        Aperature size of telescope whose beam we are fitting.
    lmd : u.Quantity, default: 90 * u.GHz
        Wavelength of the beam being fit.
    mask_size : float, default: np.inf
        Size (in radians) of the mask used when making the map.
        Used to zero out `ivar` outside the mask,
        also used to set bounds when fitting the wing.
    data_fwhm : float, default: np.inf
        The FWHM of the data profile.
        Used for the initial guess of the wing.
    fit_wing : bool, default: True
        If True fit for a $r^{-3}$ wing.

    Returns
    -------
    fit_params : AxisManager
        The following scalars:

        - ell_max: The $\ell$ to scale $r$ by when fitting
        - r0_wing: The radius to start the wing at
        - amp_wing: The amplitude of the wing
        - off_wing: The offset of the wing, currently fixed to 0
        - fit_wing: A copy of the input `fit_wing`

        Additionally the amplitude of the core beam terms are stored in a
        `(n_bessel, n_multipole, 2)` array. Where the first axis points
        to which bessel function the term is for, the second the multipole
        and the third whether it is the `cos` or the `sin` term.
        These are tracked by the `bessel`, `multipoles`, and `term`
        axes in the AxisManager.
    model : NDArray
        The model evaluated with the fit parameters.
    """

    def _wing_msk(beam_model, amp_wing, r, r0_wing):
        msk = r > r0_wing
        blk_msk = block_replicate(
            (block_reduce((beam_model < amp_wing).astype(int), 2) > 0), 2, False
        ).astype(bool)
        sl = tuple(slice(0, s) for s in blk_msk.shape)
        msk[sl] += blk_msk
        return msk

    def _wing_obj(coeffs, beam_model, r, imap, ivar):
        r0, a, off = coeffs
        wing_model = beam_model.copy()
        wmsk = _wing_msk(beam_model, a + off, r, r0)
        wing_model[wmsk] = off + a * (r0**3) / np.power(r[wmsk], 3)

        return np.nansum(ivar * (imap - wing_model) ** 2)

    ell_max = (np.pi * d / lmd).decompose().value
    eta, xi = posmap
    eta0 = guess.eta0.to(u.radian).value
    xi0 = guess.xi0.to(u.radian).value
    xi = xi - xi0
    eta = eta - eta0
    theta = np.arctan2(eta, xi)
    r = np.sqrt(xi**2 + eta**2)

    # Setup aman
    map_units = u.Unit(map_units)
    aman = AxisManager()
    b_ax = IndexAxis("bessel", n_bessel)
    mp_ax = IndexAxis("multipoles", n_multipoles)
    sc_ax = LabelAxis("term", ["cos", "sin"])
    aman.wrap("ell_max", ell_max * u.dimensionless_unscaled)
    aman.wrap("fit_wing", fit_wing)
    aman.wrap("r0_wing", np.inf * u.radian)
    aman.wrap("amp_wing", -1 * np.inf * map_units)
    aman.wrap("off_wing", 0 * map_units)

    # Compute initial model
    cent_pix = r < np.deg2rad(posmap.wcs.wcs.cdelt[1]) / 2
    amps = np.zeros((n_bessel, n_bessel, n_multipoles, 2))
    beam_model = np.zeros_like(xi)
    ivar_orig = ivar.copy()
    ivar[cent_pix] = 0
    for n0 in range(len(amps)):
        b0 = bessel_term(r, ell_max, n0)
        for n1 in range(n0, len(amps)):
            b1 = bessel_term(r, ell_max, n1)
            base_beam = b0 * b1
            amps[n0, n1] = multipole_decomp(
                base_beam, imap - beam_model, ivar, n_multipoles, theta, True, True
            )
            beam_model += multipole_expansion(base_beam, amps[n0, n1], theta)

    # Fit for a symmetric r^-3 wing with the intial model
    wmsk = np.zeros_like(beam_model, dtype=bool)
    if fit_wing:
        ivar = ivar_orig.copy()
        ivar[r > mask_size] = 0
        ivar[r <= mask_size] = 1
        avg_sig = (data_fwhm).to(u.radian).value / 2.355
        r0 = min(mask_size, np.max(r))
        w_guess = [min(5 * avg_sig, 0.7 * r0), 0, 0]
        bounds = [
            (min(0.5 * r0, w_guess[0] * 0.5), min(0.85 * r0, 2 * w_guess[0])),
            (0, 1),
            (-np.inf, np.inf),
        ]
        res = minimize(
            _wing_obj,
            w_guess,
            bounds=bounds,
            args=(beam_model, r, imap, ivar),
            method="powell",
        )
        if not res.success:
            return None, None
        wmsk = _wing_msk(beam_model, res.x[1] + res.x[2], r, res.x[0])

    # Now use the wing mask to do a lstsq fit in a limited range
    ivar = ivar_orig.copy()
    core_r = max(np.max(r[~wmsk]), 1.5 * mask_size)
    core_msk = r <= core_r
    X = []
    idx = []
    for n0 in range(len(amps)):
        b0 = bessel_term(r, ell_max, n0)
        for n1 in range(n0, len(amps)):
            b1 = bessel_term(r, ell_max, n1)
            base_beam = b0 * b1
            for m in range(n_multipoles):
                for i in range(2):
                    mp = multipole(theta[core_msk], m, i)
                    X += [base_beam[core_msk] * mp]
                    idx += [(n0, n1, m, i)]
    X += [np.ones_like(beam_model[core_msk])]
    X = np.column_stack(X) * np.sqrt(ivar[core_msk])[..., None]
    als, _, _, _ = lstsq(
        X, imap[core_msk] * np.sqrt(ivar[core_msk]), cond=1e-3 * np.min(amps)
    )
    als, off = als[:-1], als[-1]
    amps[:] = 0
    for i, a in zip(idx, als):
        amps[i] = a
    beam_model = bessel_beam(
        posmap, xi0, eta0, ell_max, amps, off, np.max(r), -1 * np.inf, 0
    )
    aman.wrap("amps", amps * map_units, [(0, b_ax), (1, b_ax), (2, mp_ax), (3, sc_ax)])
    aman.wrap("off", off * map_units)

    # Refit the wing
    if fit_wing:
        ivar[r > mask_size] = 0
        ivar[r <= mask_size] = 1
        bounds[2] = (0, 0)
        w_guess[:2] = res.x[:2]
        res = minimize(
            _wing_obj,
            w_guess,
            bounds=bounds,
            args=(beam_model, r, imap, ivar),
            method="powell",
        )
        if not res.success:
            return None, None
        aman.r0_wing = res.x[0] * u.radian
        aman.amp_wing = res.x[1] * map_units
        aman.off_wing = res.x[2] * map_units
        wmsk = _wing_msk(
            beam_model, aman.amp_wing.value + aman.off_wing.value, r, aman.r0_wing.value
        )
        beam_model[wmsk] = (
            aman.amp_wing.value * (aman.r0_wing.value**3) / np.power(r[wmsk], 3)
            + aman.off_wing.value
        )

    return aman, beam_model
