"""
Shared building blocks for beam models and fitting routines.
"""

from typing import overload

import numpy as np
from jaxtyping import Float
from joblib import Memory
from numba import njit
from numpy.typing import NDArray
from pixell.enmap import ndmap
from scipy.special import jv

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
except ImportError:
    myrank = 0

location = f"/tmp/lat_beams_{myrank}"
memory = Memory(location, verbose=0)


@overload
def gaussian2d(
    posmap: Float[ndmap, "2 ny nx"],
    a: float,
    xi0: float,
    eta0: float,
    fwhm_xi: float,
    fwhm_eta: float,
    phi: float,
    off: float,
) -> Float[ndmap, "ny nx"]: ...


@overload
def gaussian2d(
    posmap: Float[NDArray, "2 ny nx"],
    a: float,
    xi0: float,
    eta0: float,
    fwhm_xi: float,
    fwhm_eta: float,
    phi: float,
    off: float,
) -> Float[NDArray, "ny nx"]: ...


@njit
def gaussian2d(
    posmap: Float[ndmap, "ny nx"] | Float[NDArray, "ny nx"],
    a: float,
    xi0: float,
    eta0: float,
    fwhm_xi: float,
    fwhm_eta: float,
    phi: float,
    off: float,
) -> Float[ndmap, "ny nx"] | Float[NDArray, "ny nx"]:
    """
    Evaluate a rotated 2D Gaussian beam.

    Parameters
    ----------
    posmap : Float[ndmap, "2 ny nx"] | Float[NDArray, "2 ny nx"]
        Position map in radians. The first element is eta and the second
        is xi.
    a : float
        Beam amplitude.
    xi0 : float
        Beam center along xi, in radians.
    eta0 : float
        Beam center along eta, in radians.
    fwhm_xi : float
        FWHM along the xi axis, in radians.
    fwhm_eta : float
        FWHM along the eta axis, in radians.
    phi : float
        Rotation angle of the beam, in radians.
    off : float
        Constant offset added to the beam.

    Returns
    -------
    gauss : Float[ndmap, "ny nx"] | Float[NDArray, "ny nx"]
        Gaussian beam evaluated at `posmap`, with the same array type as
        `posmap`.
    """
    eta, xi = posmap
    model = np.empty_like(eta)

    sigma_xi = fwhm_xi / np.sqrt(8 * np.log(2))
    sigma_eta = fwhm_eta / np.sqrt(8 * np.log(2))

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos2 = cos_phi**2
    sin2 = sin_phi**2
    sin2phi = np.sin(2 * phi)

    a_coef = cos2 / (2 * sigma_eta**2) + sin2 / (2 * sigma_xi**2)
    b_coef = -sin2phi / (4 * sigma_eta**2) + sin2phi / (4 * sigma_xi**2)
    c_coef = sin2 / (2 * sigma_eta**2) + cos2 / (2 * sigma_xi**2)

    deta = eta - eta0
    dxi = xi - xi0

    model = (
        a
        * np.exp(-(a_coef * deta * deta + 2 * b_coef * deta * dxi + c_coef * dxi * dxi))
        + off
    )
    return model


@overload
def gaussian2d_deriv(
    posmap: Float[ndmap, "2 ny nx"],
    a: float,
    xi0: float,
    eta0: float,
    fwhm_xi: float,
    fwhm_eta: float,
    phi: float,
    off: float,
) -> tuple[Float[ndmap, "ny nx"], Float[ndmap, "7 ny nx"]]: ...


@overload
def gaussian2d_deriv(
    posmap: Float[NDArray, "2 ny nx"],
    a: float,
    xi0: float,
    eta0: float,
    fwhm_xi: float,
    fwhm_eta: float,
    phi: float,
    off: float,
) -> tuple[Float[NDArray, "ny nx"], Float[NDArray, "7 ny nx"]]: ...


def gaussian2d_deriv(
    posmap: Float[ndmap, "ny nx"] | Float[NDArray, "ny nx"],
    a: float,
    xi0: float,
    eta0: float,
    fwhm_xi: float,
    fwhm_eta: float,
    phi: float,
    off: float,
) -> (
    tuple[Float[ndmap, "ny nx"], Float[ndmap, "7 ny nx"]]
    | tuple[Float[NDArray, "ny nx"], Float[NDArray, "7 ny nx"]]
):
    """
    Evaluate a Gaussian beam and its parameter derivatives.

    Parameters
    ----------
    posmap : Float[ndmap, "2 ny nx"] | Float[NDArray, "2 ny nx"]
        Position map in radians. The first element is xi and the second
        is eta.
    a : float
        Beam amplitude.
    xi0 : float
        Beam center along xi, in radians.
    eta0 : float
        Beam center along eta, in radians.
    fwhm_xi : float
        FWHM along xi, in radians.
    fwhm_eta : float
        FWHM along eta, in radians.
    phi : float
        Rotation angle of the beam, in radians.
    off : float
        Constant offset of the beam.

    Returns
    -------
    gauss : Float[ndmap, "ny nx"] | Float[NDArray, "ny nx"]
        Gaussian beam evaluated at `posmap`, with the same array type as
        the input.
    dgauss : Float[ndmap, "7 ny nx"] | Float[NDArray, "7 ny nx"]
        Derivatives with respect to amplitude, xi center, eta center,
        xi FWHM, eta FWHM, rotation angle, and offset.
    """
    factor = 2 * np.sqrt(2 * np.log(2))
    xi, eta = posmap
    gauss = gaussian2d(posmap, a, xi0, eta0, fwhm_xi, fwhm_eta, phi, off)
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


@overload
def gaussian2d_wing(
    posmap: Float[ndmap, "ny nx"],
    amp: float,
    dx: float,
    dy: float,
    fwhm_xi: float,
    fwhm_eta: float,
    phi: float,
    off: float,
    wing_r0: float,
    wing_amp: float,
) -> Float[ndmap, "ny nx"]: ...


@overload
def gaussian2d_wing(
    posmap: Float[NDArray, "2 ny nx"],
    amp: float,
    dx: float,
    dy: float,
    fwhm_xi: float,
    fwhm_eta: float,
    phi: float,
    off: float,
    wing_r0: float,
    wing_amp: float,
) -> Float[NDArray, "ny nx"]: ...


def gaussian2d_wing(
    posmap: Float[ndmap, "ny nx"] | Float[NDArray, "2 ny nx"],
    amp: float,
    dx: float,
    dy: float,
    fwhm_xi: float,
    fwhm_eta: float,
    phi: float,
    off: float,
    wing_r0: float,
    wing_amp: float,
) -> Float[ndmap, "ny nx"] | Float[NDArray, "ny nx"]:
    """
    Evaluate a Gaussian beam with a $r^{-3}$ power-law wing.

    Parameters
    ----------
    posmap : Float[ndmap, "ny nx"] | Float[NDArray, "2 ny nx"]
        Position map in radians. The first component is eta and the
        second is xi.
    amp : float
        Amplitude of the Gaussian core.
    dx : float
        Beam center in xi.
    dy : float
        Beam center in eta.
    fwhm_xi : float
        FWHM of the Gaussian core along xi.
    fwhm_eta : float
        FWHM of the Gaussian core along eta.
    phi : float
        Rotation angle of the Gaussian core in radians.
    off : float
        Constant offset added to the beam.
    wing_r0 : float
        Radius beyond which the Gaussian is replaced by the power-law wing.
    wing_amp : float
        Amplitude of the power-law wing.

    Returns
    -------
    Float[ndmap, "ny nx"] | Float[NDArray, "ny nx"]
        Beam model evaluated at `posmap`. The return type matches the
        type of `posmap`.
    """
    gauss = gaussian2d(posmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, 0)
    r = np.sqrt((posmap[0] - dy) ** 2 + (posmap[1] - dx) ** 2)
    r_msk = r > wing_r0
    gauss[r_msk] = wing_amp * np.power(r[r_msk], -3)
    gauss += off
    return gauss


def multipole(
    theta: Float[NDArray, "ny nx"], mp: int, sin: int
) -> Float[NDArray, "ny nx"]:
    """
    Evaluate a cosine or sine multipole basis function.

    Parameters
    ----------
    theta : Float[NDArray, "ny nx"]
        Polar angle map in radians.
    mp : int
        Multipole order. Must be non-negative.
    sin : int
        Selects the cosine (`0`) or sine (`1`) term.

    Returns
    -------
    Float[NDArray, "ny nx"]
        Multipole basis evaluated at `theta`.

    Raises
    ------
    ValueError
        If `mp` is negative.
    """
    if mp < 0:
        raise ValueError("Negative multipole orders not allowed!")
    return np.cos(theta * mp - sin * np.pi / 2)


def multipole_decomp(
    base_beam: Float[ndmap, "ny nx"],
    imap: Float[ndmap, "ny nx"],
    sigma: Float[ndmap, "ny nx"],
    n_multipoles: int,
    theta: Float[ndmap, "ny nx"],
    gs: bool = False,
    check_chisq: bool = False,
) -> Float[NDArray, "n_multipoles 2"]:
    """
    Decompose a map into multipoles of a base beam.

    Parameters
    ----------
    base_beam : Float[ndmap, "ny nx"]
        Base beam used for every multipole term.
    imap : Float[ndmap, "ny nx"]
        Input map to decompose.
    sigma : Float[ndmap, "ny nx"]
        Inverse-variance weights for `imap`.
    n_multipoles : int
        Number of multipoles to fit.
    theta : Float[ndmap, "ny nx"]
        Polar angle map in radians.
    gs : bool, default=False
        If `True`, iteratively subtract fitted multipoles before fitting
        the next term.
    check_chisq : bool, default=False
        If `True`, reject terms that increase the weighted chi-squared.

    Returns
    -------
    Float[NDArray, "n_multipoles 2"]
        Fitted amplitudes. The second dimension contains the cosine and
        sine amplitudes.
    """
    amps = np.zeros((n_multipoles, 2))
    beam_model = imap
    _multi_mod = beam_model
    _amps = np.zeros(2)
    if gs or check_chisq:
        beam_model = imap.copy()
        beam_model[:] = 0
        _multi_mod = beam_model.copy()
    chisq = np.inf
    if check_chisq:
        chisq = np.nansum(sigma * (imap - beam_model) ** 2)
    for n in range(n_multipoles):
        _amps[:] = 0
        if check_chisq or gs:
            _multi_mod[:] = 0
        for i in (0, 1):
            mp = multipole(theta, n, i)
            model = mp * base_beam
            _sigma = sigma.copy()
            _sigma[~np.isfinite(model)] = 0
            model[~np.isfinite(model)] = 0
            norm = np.nansum(_sigma * model * model)
            if norm == 0:
                continue
            amp = (
                np.nansum(_sigma * (imap - gs * (beam_model + _multi_mod)) * model)
                / norm
            )
            if np.isnan(amp):
                continue
            _amps[i] = amp
            _multi_mod += amp * model
        if check_chisq:
            new_chisq = np.nansum(sigma * (imap - beam_model - _multi_mod) ** 2)
            if new_chisq > chisq:
                continue
            chisq = new_chisq
        if check_chisq or gs:
            beam_model += _multi_mod
        amps[n] = _amps
    return amps


def multipole_expansion(
    base_beam: Float[ndmap, "ny nx"],
    amps: Float[NDArray, "n_multipoles 2"],
    theta: Float[ndmap, "ny nx"],
) -> Float[ndmap, "ny nx"]:
    """
    Evaluate a multipole expansion of a base beam.

    Parameters
    ----------
    base_beam : Float[ndmap, "ny nx"]
        Base beam used for every multipole term.
    amps : Float[NDArray, "n_multipoles 2"]
        Cosine and sine amplitudes for each multipole.
    theta : Float[ndmap, "ny nx"]
        Polar angle map in radians.

    Returns
    -------
    Float[ndmap, "ny nx"]
        Beam model evaluated using the multipole expansion.
    """
    beam = np.zeros_like(base_beam)
    for n in range(len(amps)):
        for i in (0, 1):
            mp = multipole(theta, n, i)
            beam += amps[n, i] * mp * base_beam
    return beam


def bessel_term(
    r: Float[NDArray, "ny nx"],
    ell_max: float,
    i: int,
) -> Float[NDArray, "ny nx"]:
    r"""
    Evaluate the normalized Bessel basis function:

    $$
    \frac{J_i(r * \ell_{max})}{(r * \ell_{max})}
    $$

    Parameters
    ----------
    r : Float[NDArray, "ny nx"]
        Radial coordinate.
    ell_max : float
        Maximum multipole scale.
    i : int
        Bessel function order.

    Returns
    -------
    Float[NDArray, "ny nx"]
        Bessel basis function
        At `r = 0`, the limiting value is used: `1` for `i = 0`
        and `0` otherwise.
    """
    x = r * ell_max
    out = np.empty_like(x, dtype=float)
    zero = x == 0
    nonzero = ~zero
    out[nonzero] = jv(i, x[nonzero]) / x[nonzero]
    out[zero] = 0.0
    if i == 0:
        out[zero] = 1.0
    return out


bessel_term_cached = memory.cache(bessel_term)
bessel_term_cached.__doc__ = """
Cached version of `bessel_term`.

Results are cached on disk using `joblib.Memory`. The cache is stored in
`/tmp/lat_beams_{myrank}`, where `{myrank}` is the MPI rank of the current
process. Calls with the same arguments reuse the previously computed result
instead of evaluating the Bessel function again.
"""
