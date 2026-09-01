"""
Gaussian and multipole expansion beam fitting utilities.

The Gaussian beam model is fit directly to a map using weighted least squares.
A fitted Gaussian can then be used as the base beam for a multipole expansion,
whose amplitudes are obtained by a weighted linear decomposition.  See the
individual fitting and model-conversion functions for details.
"""

import logging
from typing import Optional, cast

import numpy as np
from astropy import units as u
from jaxtyping import Float
from pixell.enmap import ndmap
from scipy.optimize import least_squares  # , minimize
from sotodlib.core import AxisManager, IndexAxis, LabelAxis
from sotodlib.tod_ops.filters import logger as flog

from ..core import gaussian2d, multipole_decomp, multipole_expansion

flog.setLevel(logging.ERROR)


def fit_gauss_map(
    imap: Float[ndmap, "ny nx"],
    ivar: Float[ndmap, "ny nx"],
    posmap: Float[ndmap, "2 ny nx"],
    guess: AxisManager,
    map_units: str = "pW",
    force_sym: bool = False,
    mask_size: float = -1,
) -> tuple[Optional[AxisManager], Optional[Float[ndmap, "ny nx"]]]:
    """Fit a 2D Gaussian beam to a map.

    The fit minimizes the inverse-variance-weighted residual between `imap`
    and a Gaussian beam. The Gaussian can optionally be constrained to be
    symmetric, and the fit can optionally be restricted to a circular region
    around the initial beam center.

    Parameters
    ----------
    imap : Float[ndmap, "ny nx"]
        Input beam map.
    ivar : Float[ndmap, "ny nx"]
        Inverse-variance map.
    posmap : Float[ndmap, "2 ny nx"]
        Beam coordinates. The first component is `eta` and the second is
        `xi`.
    guess : AxisManager
        Initial Gaussian parameters. Must contain `xi0`, `eta0`, `off`,
        `amp`, `fwhm_xi`, `fwhm_eta`, and `phi`.
    map_units : str, default: "pW"
        Unit of the map amplitude and offset parameters.
    force_sym : bool, default: False
        If `True`, constrain the fitted Gaussian to have equal FWHM in
        `xi` and `eta` and zero rotation angle.
    mask_size : float, default: -1
        If positive, only pixels within this radius of the initial beam center
        are included in the fit.

    Returns
    -------
    fit_params : Optional[AxisManager]
        Fitted Gaussian parameters. Positional parameters are stored in
        radians. `None` if the fit fails.
    model : Optional[Float[ndmap, "ny nx"]]
        The fitted Gaussian evaluated on `posmap`. `None` if the fit
        fails.
    """

    y, x = posmap
    x0 = [
        cast(float, guess.xi0),
        cast(float, guess.eta0),
        cast(float, guess.off),
        cast(float, guess.amp),
        cast(float, guess.fwhm_xi),
        cast(float, guess.fwhm_eta),
        cast(float, guess.phi),
    ]
    bounds = [
        [
            np.min(x) - cast(float, guess.fwhm_xi),
            np.min(y) - cast(float, guess.fwhm_eta),
            -5 * np.max(np.abs(imap)),
            0,
            cast(float, guess.fwhm_xi) / 3,
            cast(float, guess.fwhm_eta) / 3,
            0,
        ],
        [
            np.max(x) + cast(float, guess.fwhm_xi),
            np.max(y) + cast(float, guess.fwhm_eta),
            5 * np.max(np.abs(imap)),
            5 * np.max(np.abs(imap)),
            cast(float, guess.fwhm_xi) * 3,
            cast(float, guess.fwhm_eta) * 3,
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
        msk = r < mask_size

        fit_imap = np.asarray(imap)[msk]
        fit_ivar = np.asarray(ivar)[msk]
        fit_posmap = np.asarray(posmap)[:, msk]
    else:
        fit_imap = np.asarray(imap)
        fit_ivar = np.asarray(ivar)
        fit_posmap = np.asarray(posmap)

    w = np.sqrt(fit_ivar)

    def _to_pars(coeffs):
        dx, dy, off, amp = coeffs[:4]

        if force_sym:
            fwhm_xi = fwhm_eta = coeffs[4]
            phi = 0
        else:
            fwhm_xi, fwhm_eta, phi = coeffs[4:]

        return dx, dy, off, amp, fwhm_xi, fwhm_eta, phi

    def _resid(coeffs):
        dx, dy, off, amp, fwhm_xi, fwhm_eta, phi = _to_pars(coeffs)
        beam = gaussian2d(fit_posmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, off)

        return (w * (fit_imap - beam)).ravel()

    # def _objective(
    #     coeffs,
    # ):
    #     resid = _resid(coeffs)
    #     chisq = np.nansum(resid**2)
    #     return chisq

    try:
        res = least_squares(
            _resid,
            x0,
            bounds=np.array(bounds).T,
            method="trf",
            x_scale="jac",
        )
        if not res.success:
            print(res)
            return None, None
    except ValueError:
        return None, None

    # Convert to aman
    aman = AxisManager()
    dx, dy, off, amp, fwhm_xi, fwhm_eta, phi = pars = _to_pars(res.x)
    for n, un, v in zip(par_names, par_units, pars):
        aman.wrap(n, v * un)
    model = gaussian2d(posmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, off)

    return aman, model


def fit_multipole_map(
    imap: Float[ndmap, "ny nx"],
    ivar: Float[ndmap, "ny nx"],
    posmap: Float[ndmap, "2 ny nx"],
    guess: AxisManager,
    map_units: str = "pW",
    base_beam: Optional[Float[ndmap, "ny nx"]] = None,
    n_multipoles: int = 5,
) -> tuple[AxisManager, Float[ndmap, "ny nx"]]:
    """
    Fit a multipole expansion of a beam model to a map.

    The input map is modeled as a multipole expansion of `base_beam`.
    If no base beam is provided, a unit-amplitude Gaussian is constructed
    from `guess`. The multipole amplitudes are fit using inverse-variance
    weighting.

    Parameters
    ----------
    imap : Float[ndmap, "ny nx"]
        Input beam map.
    ivar : Float[ndmap, "ny nx"]
        Inverse-variance map.
    posmap : Float[ndmap, "2 ny nx"]
        Beam coordinates. The first component is `eta` and the second is
        `xi`.
    guess : AxisManager
        Initial Gaussian beam parameters used to construct `base_beam` when
        it is not provided.
    map_units : str, default: "pW"
        Unit of the fitted multipole amplitudes.
    base_beam : Optional[Float[ndmap, "ny nx"]], default: None
        Base beam whose angular dependence is expanded. If `None`, a
        unit-amplitude Gaussian is constructed from `guess`.
    n_multipoles : int, default: 5
        Number of multipoles to fit. `0` fits only the monopole, `1` adds
        the dipole, and so on.

    Returns
    -------
    fit_params : AxisManager
        Fitted multipole amplitudes stored in `amps` with shape
        `(n_multipoles, 2)`. The second dimension contains the cosine and
        sine amplitudes. The `AxisManager` also contains `multipoles` and
        `term` axes.
    model : Float[ndmap, "ny nx"]
        Beam model evaluated with the fitted multipole amplitudes.
    """

    if base_beam is None:
        base_beam = gaussian2d(
            posmap,
            1,
            cast(u.Quantity, guess.xi0).value,
            cast(u.Quantity, guess.eta0).value,
            cast(u.Quantity, guess.fwhm_xi).value,
            cast(u.Quantity, guess.fwhm_eta).value,
            cast(u.Quantity, guess.phi).value,
            0,
        )
    y, x = posmap
    theta = np.arctan2(
        y - cast(u.Quantity, guess.eta0).to(u.radian).value,
        x - cast(u.Quantity, guess.xi0).to(u.radian).value,
    )

    # Compute model
    if n_multipoles == 0:
        amps = np.array([[cast(u.Quantity, guess.amp).value, 0]])
    else:
        amps = multipole_decomp(base_beam, imap, ivar, n_multipoles, theta, True)
    model = imap.copy()
    model[...] = multipole_expansion(base_beam, amps, theta)
    model = cast(
        ndmap,
        multipole_expansion(base_beam, amps, theta),
    )

    # Convert to aman
    m_units = u.Unit(map_units)
    aman = AxisManager()
    mp_ax = IndexAxis("multipoles", n_multipoles)
    sc_ax = LabelAxis("term", ["cos", "sin"])
    aman.wrap("amps", amps * m_units, [(0, mp_ax), (1, sc_ax)])

    return aman, model


def gaussian2d_from_aman(
    posmap: Float[ndmap, "2 ny nx"],
    aman: AxisManager,
) -> Float[ndmap, "ny nx"]:
    """Evaluate a Gaussian beam from an AxisManager.

    Parameters
    ----------
    posmap : Float[ndmap, "2 ny nx"]
        Beam coordinates. The first component is `eta` and the second is
        `xi`.
    aman : AxisManager
        AxisManager containing the Gaussian parameters. If it contains a
        `gaussian` field, that field is used. Otherwise the parameters are
        read directly from `aman`.

    Returns
    -------
    beam : Float[ndmap, "ny nx"]
        Gaussian beam evaluated at `posmap`.
    """
    if "gaussian" in aman._fields:
        aman = aman.gaussian

    return gaussian2d(
        posmap,
        cast(u.Quantity, aman.amp).value,
        cast(u.Quantity, aman.xi0).to(u.radian).value,
        cast(u.Quantity, aman.eta0).to(u.radian).value,
        cast(u.Quantity, aman.fwhm_xi).to(u.radian).value,
        cast(u.Quantity, aman.fwhm_eta).to(u.radian).value,
        cast(u.Quantity, aman.phi).to(u.radian).value,
        cast(u.Quantity, aman.off).value,
    )


def gaussian2d_multipoles_from_aman(
    posmap: Float[ndmap, "2 ny nx"],
    aman: AxisManager,
) -> Float[ndmap, "ny nx"]:
    """Evaluate a Gaussian multipole beam from an AxisManager.

    Parameters
    ----------
    posmap : Float[ndmap, "2 ny nx"]
        Beam coordinates. The first component is `eta` and the second is
        `xi`.
    aman : AxisManager
        AxisManager containing `gaussian` and `gauss_multipole` fields.
        `gaussian` contains the base Gaussian parameters and
        `gauss_multipole.amps` contains the multipole amplitudes.

    Returns
    -------
    beam : Float[ndmap, "ny nx"]
        Gaussian multipole beam evaluated at `posmap`.
    """
    base_beam = gaussian2d_from_aman(posmap, aman.gaussian)
    base_beam -= cast(u.Quantity, aman.gaussian.off).value
    base_beam /= cast(u.Quantity, aman.gaussian.amp).value

    y, x = posmap
    theta = np.arctan2(
        y - cast(u.Quantity, aman.gaussian.eta0).to(u.radian).value,
        x - cast(u.Quantity, aman.gaussian.xi0).to(u.radian).value,
    )
    amps = (np.array(aman.gauss_multipole.amps.value),)

    model = posmap[0].copy()
    model[...] = multipole_expansion(base_beam, amps, theta)
    model = cast(ndmap, multipole_expansion(base_beam, amps, theta))

    return model
