import astropy.units as u
import numpy as np
from scipy.special import spherical_jn 
from functools import lru_cache

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
    xi_sft = xi - xi0
    eta_sft = eta - eta0
    xi_rot = xi_sft * np.cos(phi) - eta_sft * np.sin(phi)
    eta_rot = xi_sft * np.sin(phi) + eta_sft * np.cos(phi)
    factor = 2 * np.sqrt(2 * np.log(2))
    xi_coef = -0.5 * (xi_rot) ** 2 / (fwhm_xi / factor) ** 2
    eta_coef = -0.5 * (eta_rot) ** 2 / (fwhm_eta / factor) ** 2
    sim_data = a * np.exp(xi_coef + eta_coef)
    return sim_data + off


def multipole(theta, mp, sin):
    order = mp
    if mp > 0:
        order = 2 ** (mp - 1)
    elif mp < 0:
        raise ValueError("Negetive multipole orders not allowed!")
    return np.cos(theta * order - sin * np.pi / 2)


def multipole_decomp(base_beam, imap, sigma, n_multipoles, theta, gs=False, check_chisq=False):
    amps = np.zeros((n_multipoles, 2))
    beam_model = imap
    if gs or check_chisq:
        beam_model = imap.copy()
        beam_model[:] = 0
    chisq = np.inf
    if check_chisq:
        chisq = np.nansum(sigma * (imap - beam_model)**2) 
    for n in range(n_multipoles):
        for i in (0, 1):
            mp = multipole(theta, n, i)
            model = mp * base_beam
            _sigma = sigma.copy()
            _sigma[~np.isfinite(model)] = 0
            model[~np.isfinite(model)] = 0
            norm = np.nansum(_sigma * model * model)
            if norm == 0:
                continue
            amp = np.nansum(_sigma * (imap - gs*beam_model) * model) / norm
            if np.isnan(amp):
                continue
            if check_chisq:
                new_chisq = np.nansum(sigma * (imap - beam_model - amp*model)**2)
                if new_chisq > chisq:
                    continue
                chisq = new_chisq
            if check_chisq or gs:
                beam_model += amp*model
            amps[n, i] = amp
    return amps


def multipole_expansion(base_beam, amps, theta):
    beam = np.zeros_like(base_beam)
    for n in range(len(amps)):
        for i in (0, 1):
            mp = multipole(theta, n, i)
            beam += amps[n, i] * mp * base_beam
    return beam

def bessel_term(r, ell_max, i):
    with np.errstate(divide="ignore", invalid="ignore"):
        bessel = spherical_jn(i, r*ell_max)/(r*ell_max)
    return bessel

def bessel_beam(posmap, xi0, eta0, off, ell_max, amps):
    eta, xi = posmap
    xi = xi - xi0
    eta = eta - eta0
    r = np.sqrt(xi**2 + eta**2)
    theta = np.arctan2(eta, xi)

    beam_model = np.zeros_like(xi)
    for n0 in range(len(amps)):
        b0 = bessel_term(r, ell_max, n0)
        for n1 in range(len(amps)):
            b1 = bessel_term(r, ell_max, n1)
            base_beam = b0*b1
            beam_model += multipole_expansion(base_beam, amps[n0, n1], theta)

    return beam_model + off

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


def bessel_beam_from_aman(posmap, aman):
    return bessel_beam(
        posmap,
        aman.gaussian.xi0.to(u.radian).value,
        aman.gaussian.eta0.to(u.radian).value,
        aman.gaussian.off.value,
        aman.bessel.ell_max.value,
        aman.bessel.amps.value,
        )

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


def scatter_beam(r, n_terms, lmd, sang, corr, eps):
    var = (4 * np.pi * eps / lmv) ** 2
    prefac = (sang / (4 * pi)) * ((2 * np.pi * corr / lmd) ** 2) * np.exp(-1 * var)
    x = -1 * (corr * np.pi * np.sin(r) / lmd) ** 2
    profile = np.zeros_like(r)
    for n in range(1, n_terms + 1):
        profile += (var**n / (n * np.fac(n))) * np.exp(x / n)
    profile *= prefac
    return profile


def dr4_beam(r, ell_max, r_c, alpha, off, amps, n_scatter, scatter_pars=None):
    """
    1D beam profile as modeled in Lungu et al. (https://arxiv.org/pdf/2112.12226).
    Does not include scattering term yet.
    """
    profile = np.zeros_like(r)
    profile[r == 0] = 1.0

    # Core beam
    msk = (r <= r_c) * (r > 0)
    r_ell = r[msk] * ell_max
    for n, amp in enumerate(amps):
        profile[msk] += amp * jv(2 * n + 1, r_ell) / r_ell

    # Wing
    msk = r > r_c
    if np.sum(msk) > 0:
        profile[msk] = off + alpha * (r[msk][0] ** 3) / np.power(r[msk], 3)
        # Scattering beam
        if scatter_pars is not None:
            profile[msk] += scatter_beams(r[msk], **scatter_pars)

    return profile
