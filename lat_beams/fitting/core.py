"""
Shared building blocks for various models and fitting routines
"""

import astropy.units as u
import numba
import numpy as np
from joblib import Memory
from numba import njit
from numpy.typing import NDArray
from pixell.enmap import ndmap
from scipy.special import factorial, jv

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
except ImportError:
    myrank = 0

location = f"/tmp/lat_beams_{myrank}"
memory = Memory(location, verbose=0)


@njit
def gaussian2d(posmap, a, xi0, eta0, fwhm_xi, fwhm_eta, phi, off):
    """
    Stolen from analyze_bright_ptsrc

    Simulate a time stream with an Gaussian beam model
    Args
    ------
    pixmap:
        (eta, xi) of model.
    a: float
        amplitude of the Gaussian beam model
    xi0, eta0: float, float
        center position of the Gaussian beam model
    fwhm_xi, fwhm_eta, phi: float, float, float
        fwhm along the xi, eta axis (rotated)
        and the rotation angle (in radians)
    off: offset to add to beam

    Ouput:
    ------
    sim_data: 1d array of float
        Time stream at sampling points given by xieta
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


def gaussian2d_wing(
    posmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, off, wing_r0, wing_amp
):
    gauss = gaussian2d(posmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, 0)
    r = np.sqrt((posmap[0] - dy) ** 2 + (posmap[1] - dx) ** 2)
    r_msk = r > wing_r0
    gauss[r_msk] = wing_amp * np.power(r[r_msk], -3)

    return gauss + off


def multipole(theta, mp, sin):
    order = mp
    if mp > 0:
        order = 2 ** (mp - 1)
    elif mp < 0:
        raise ValueError("Negetive multipole orders not allowed!")
    order = mp
    return np.cos(theta * order - sin * np.pi / 2)


def multipole_decomp(
    base_beam, imap, sigma, n_multipoles, theta, gs=False, check_chisq=False
):
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


def multipole_expansion(base_beam, amps, theta):
    beam = np.zeros_like(base_beam)
    for n in range(len(amps)):
        for i in (0, 1):
            mp = multipole(theta, n, i)
            beam += amps[n, i] * mp * base_beam
    return beam


def bessel_term( r: NDArray[np.float64], ell_max: float, i: int,) -> NDArray[np.float64]:
    """
    Evaluate the normalized Bessel basis function.
    Computes
        J_i(r * ell_max) / (r * ell_max)
    with the limiting value at `r = 0` handled explicitly.
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
