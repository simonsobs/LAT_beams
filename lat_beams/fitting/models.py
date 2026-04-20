import astropy.units as u
import numpy as np
from astropy.nddata import block_reduce, block_replicate
from scipy.special import factorial, jv, spherical_jn
from joblib import Memory

location = "/tmp/lat_beams"
memory = Memory(location, verbose=0)


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

def gaussian2d_wing(posmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, off, wing_r0, wing_amp):
    gauss = gaussian2d(posmap, amp, dx, dy, fwhm_xi, fwhm_eta, phi, 0)
    r = np.sqrt((posmap[0] - dy)**2 + (posmap[1] - dx)**2)
    r_msk = r > wing_r0
    gauss[r_msk] = (wing_amp * np.power(r[r_msk], -3))

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


def bessel_term(r, ell_max, i):
    with np.errstate(divide="ignore", invalid="ignore"):
        bessel = jv(i, r * ell_max) / (r * ell_max)
    return bessel

bessel_term_cached = memory.cache(bessel_term)

def bessel_beam(
    posmap,
    xi0,
    eta0,
    gauss_amp,
    gauss_off,
    ell_max,
    amps,
    off_bessel,
    r0_wing,
    amp_wing,
    thetas,
    mask_size,
    n_sigma,
):
    eta, xi = posmap
    xi = xi - xi0
    eta = eta - eta0
    r = np.sqrt(xi**2 + eta**2)
    theta = np.arctan2(eta, xi)

    beam_model = np.zeros_like(xi) + off_bessel
    for n0 in range(len(amps)):
        b0 = np.array(bessel_term_cached(r, ell_max, n0))
        for n1 in range(n0, len(amps)):
            b1 = np.array(bessel_term_cached(r, ell_max, n1))
            base_beam = b0 * b1
            beam_model += multipole_expansion(
                base_beam, amps[n0, n1], theta
            )

    if len(thetas) == 0:
        return beam_model + gauss_off

    thresh = gauss_amp * np.exp(-.5 * (n_sigma**2))
    wmsk = (beam_model < thresh)
    beam_model += gauss_off
    tbins = np.digitize(theta, thetas)

    for tb in np.unique(tbins):
        tmsk = (tbins == tb)
        twmsk = tmsk * wmsk
        r0 = r0_wing[tb]
        amp = amp_wing[tb]
        rmsk = ((r > 1.*mask_size) + (r > r0)) * tmsk
        beam_model[twmsk + rmsk] = amp * (r0/r[twmsk + rmsk])**3

    return beam_model


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
        aman.gauss.xi0.to(u.radian).value,
        aman.gauss.eta0.to(u.radian).value,
        aman.gauss.amp.value,
        aman.gauss.off.value,
        aman.bessel.ell_max.value,
        aman.bessel.amps.value,
        aman.bessel.off.value,
        aman.bessel.r0_wing.to(u.radian).value,
        aman.bessel.amp_wing.value,
        aman.bessel.thetas.to(u.radian).value,
        aman.bessel.mask_size.value,
        aman.bessel.n_sigma.value,
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
    var = (4 * np.pi * eps / lmd) ** 2
    prefac = (sang / (4 * np.pi)) * ((2 * np.pi * corr / lmd) ** 2) * np.exp(-1 * var)
    x = -1 * (corr * np.pi * np.sin(r) / lmd) ** 2
    profile = np.zeros_like(r)
    for n in range(1, n_terms + 1):
        profile += (var**n / (n * factorial(n))) * np.exp(x / n)
    profile *= prefac
    return profile


def add_profile_wing(profile, r, r_c, alpha, off, scatter_pars):
    msk = r > r_c
    if np.sum(msk) > 0:
        profile[msk] = off + alpha * (r[msk][0] ** 3) / np.power(r[msk], 3)
        # Scattering beam
        if scatter_pars is not None:
            profile[msk] += scatter_beam(r[msk], **scatter_pars)


def dr4_beam(r, ell_max, r_c, alpha, off, amps, n_scatter, scatter_pars=None):
    """
    1D beam profile as modeled in Lungu et al. (https://arxiv.org/pdf/2112.12226).
    """
    profile = np.zeros_like(r)
    profile[r == 0] = 1.0

    # Core beam
    msk = (r <= r_c) * (r > 0)
    r_ell = r[msk] * ell_max
    for n, amp in enumerate(amps):
        profile[msk] += amp * jv(2 * n + 1, r_ell) / r_ell

    # Wing
    profile = add_profile_wing(profile, r, r_c, alpha, off, scatter_pars)

    return profile
