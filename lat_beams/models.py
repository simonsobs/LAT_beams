import numpy as np
from scipy.special import jv


def bessel_beam(x, y, n_modes, n_multipoles, dx, dy, off, l_max, coeffs):
    x = x - dx
    y = y - dy
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    rl = r * l_max
    # This only makes sense because I assume O(n_modes) ~ 1
    # If its expected to be >>1 then we should not cache
    with np.errstate(divide="ignore", invalid="ignore"):
        bessels = np.array([jv(2 * i + 1, rl) / rl for i in range(n_modes)])

    beam_model = np.sum(coeffs[:n_modes][..., None, None] * bessels, axis=0)
    coeffs = coeffs[n_modes:]
    for m in range(n_multipoles):
        order = 2**m
        for i, op in enumerate((np.sin, np.cos)):
            mp = op(theta * order)
            cf = coeffs[(2 * m + i) * n_modes : (2 * m + i + 1) * n_modes][
                ..., None, None
            ]
            beam_model += np.sum(bessels * cf * mp, axis=0)
    return beam_model + off


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
    off: offset to add to beam

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


def multipole(theta, mp, sin):
    order = mp
    if mp > 0:
        order = 2 ** (mp - 1)
    elif mp < 0:
        raise ValueError("Negetive multipole orders not allowed!")
    if sin:
        return np.sin(theta * order)
    return np.cos(theta * order)

def multipole_decomp(base_beam, imap, sigma, multipoles, theta, gs=False):
    amps = np.zeros(len(multipoles) * 2)
    beam = imap
    mod = imap
    if gs:
        beam = imap.copy()
        mod = np.zeros_like(beam)
    for m, n in enumerate(multipoles):
        if gs:
            mod[:] = 0.
        for i in (0, 1):
            mp = multipole(theta, n, i)
            model = mp * base_beam
            _sigma = sigma.copy()
            _sigma[~np.isfinite(model)] = 0
            model[~np.isfinite(model)] = 0
            j = (2 * m) + i
            norm = np.sum(_sigma*model*model)
            if norm == 0:
                continue
            amp = np.sum(_sigma*beam*model)/norm
            amps[j] = amp
            if gs:
                mod += amp*model
        if gs:
            beam -= mod
    return amps

def multipole_expansion(base_beam, amps, multipoles, theta):
    beam = np.zeros_like(base_beam)
    for m, n in enumerate(multipoles):
        for i in (0, 1):
            mp = multipole(theta, n, i)
            j = (2 * m) + i
            beam += amps[j] * mp * base_beam
    return beam 

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
    var = (4*np.pi*eps/lmv)**2
    prefac = (sang/(4*pi))*((2*np.pi*corr/lmd)**2)*np.exp(-1*var)
    x = -1*(corr * np.pi * np.sin(r)/lmd)**2
    profile = np.zeros_like(r)
    for n in range(1, n_terms + 1):
        profile += (var**n/(n*np.fac(n)))*np.exp(x/n) 
    profile *= prefac
    return profile

def dr4_beam(r, ell_max, r_c, alpha, off, amps, n_scatter, scatter_pars=None):
    """
    1D beam profile as modeled in Lungu et al. (https://arxiv.org/pdf/2112.12226).
    Does not include scattering term yet.
    """
    profile = np.zeros_like(r)
    profile[r == 0] = 1.

    # Core beam
    msk = (r <= r_c) * (r > 0)
    r_ell = r[msk] * ell_max
    for n, amp in enumerate(amps):
        profile[msk] += amp * jv(2*n + 1, r_ell)/r_ell

    # Wing
    msk = r > r_c
    if np.sum(msk) > 0:
        profile[msk] = off + alpha*(r[msk][0]**3)/np.power(r[msk], 3)
        # Scattering beam
        if scatter_pars is not None:
            profile[msk] += scatter_beams(r[msk], **scatter_pars)

    return profile
