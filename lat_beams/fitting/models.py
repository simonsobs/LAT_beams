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
    b_coef = -sin2phi / (4 * sigma_eta**2) + sin2phi / (4 * sigma_xi * 2)
    c_coef = sin2 / (2 * sigma_eta**2) + cos2 / (2 * sigma_xi**2)

    deta = eta - eta0
    dxi = xi - xi0

    model = (
        a
        * np.exp(-(a_coef * deta * deta + 2 * b_coef * deta * dxi + c_coef * dxi * dxi))
        + off
    )
    return model


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


def bessel_term(r, ell_max, i):
    with np.errstate(divide="ignore", invalid="ignore"):
        bessel = jv(i, r * ell_max) / (r * ell_max)
    return bessel


bessel_term_cached = memory.cache(bessel_term)


@numba.jit(nopython=True, fastmath=True, cache=True)
def fast_wing_transition(
    r_fit: NDArray,
    loga_coeff: NDArray,
    logr_coeff: NDArray,
    p_val: float,
    F_mat_T: NDArray,
) -> tuple[NDArray, NDArray]:
    r"""
    Evaluate the positive $r^{-3}$ wing and smooth core-to-wing transition.

    The wing amplitude and transition radius are Fourier expansions in
    angle:

    $$
    a(\theta) = \exp[F(\theta)\,c_a], \qquad
    r_0(\theta) = \exp[F(\theta)\,c_{r_0}]
    $$

    giving the asymptotic wing
    ;

    $$
    W(r,\theta) = \frac{a(\theta)}{r^3}
    $$

    The mixing weight is 0 below

    $$
    r_{\rm lo} = r_0 - \frac{\delta}{2}
    $$

    and 1 above

    $$
    r_{\rm hi} = r_0 + \frac{\delta}{2}
    $$

    and a quintic smoothstep within the transition region.
    Where

    $$
    \delta = \frac{0.3\,r_0}{p}
    $$

    and

    $$
    w(u) = 10u^3 - 15u^4 + 6u^5
    $$

    The sharpness parameter is constrained to $0.5 \leq p \leq 20$.

    Parameters
    ----------
    r_fit : NDArray
        Radial coordinates of the fitted pixels, in radians.
    loga_coeff : NDArray
        Fourier coefficients of $\log a(\theta)$.
    logr_coeff : NDArray
        Fourier coefficients of $\log r_0(\theta)$.
    p_val : float
        Logarithm of the transition sharpness, $\log p$.
    F_mat_T : NDArray
        Transposed Fourier design matrix with shape
        `(n_pixels, n_fourier)`.

    Returns
    -------
    pure_wing : NDArray
        Unblended wing model $a(\theta)/r^3$ at each fitted pixel.
    wing_weight : NDArray
        Core-to-wing mixing weight, ranging from 0 to 1.
    """
    loga = F_mat_T @ loga_coeff
    logr = F_mat_T @ logr_coeff
    a = np.exp(loga)
    r0 = np.exp(logr)
    p = max(0.5, min(20.0, np.exp(p_val)))

    n_fit = len(r_fit)
    pure_wing = np.empty(n_fit)
    wing_weight = np.empty(n_fit)

    for i in range(n_fit):
        r = max(r_fit[i], 1e-5)
        r0_i = max(r0[i], 1e-4)
        delta = 0.3 * r0_i / p
        r_lo = r0_i - 0.5 * delta
        r_hi = r0_i + 0.5 * delta

        if r <= r_lo:
            weight = 0.0
        elif r >= r_hi:
            weight = 1.0
        else:
            u = (r - r_lo) / delta
            weight = u**3 * (10.0 + u * (-15.0 + 6.0 * u))

        wing_weight[i] = weight
        wing = min(a[i], 1e20) / r**3
        pure_wing[i] = wing if np.isfinite(wing) else 0.0

    return pure_wing, wing_weight


def bessel_beam(
    posmap: ndmap,
    xi0: float,
    eta0: float,
    ell_max: float,
    amps: NDArray[np.floating],
    bessel_off: float,
    wing_params: NDArray[np.floating],
    off: float,
) -> ndmap:
    r"""
    Evaluate a fitted Bessel-core plus smooth $r^{-3}$-wing model.

    The model is

    $$
    M(r,\theta) =
    [1-w(r,\theta)]\,C(r,\theta)
    + w(r,\theta)\,W(r,\theta)
    + b,
    $$

    where $C$ is the Bessel/multipole core, $W$ is the positive
    asymptotic wing, $w$ is the smooth transition weight, and $b$ is
    the global offset.

    The wing transition and its parameterization are defined by
    `fast_wing_transition`. See that function for details.

    Parameters
    ----------
    posmap : ndmap, ndmap
        Two-dimensional coordinate maps `(eta, xi)` in radians.
    xi0 : float
        Beam center in the `xi` coordinate, in radians.
    eta0 : float
        Beam center in the `eta` coordinate, in radians.
    ell_max : float
        Maximum multipole used to construct the Bessel basis.
    amps : NDArray[np.floating]
        Bessel/multipole coefficients with shape
        `(n_bessel, n_bessel, n_multipoles, 2)`. The final axis
        contains cosine and sine coefficients.
    bessel_off : float
        Additive offset applied to the Bessel core.
    wing_params : NDArray[np.floating]
        Nonlinear wing parameters containing the Fourier coefficients
        of $\log a$, the Fourier coefficients of $\log r_0$, and
        $\log p$, in that order.
    off : float
        Global additive model offset.

    Returns
    -------
    bessel_beam : ndmap
        Model evaluated on the same two-dimensional grid as `posmap`.
    """

    n_bessel = amps.shape[0]
    n_multipoles = amps.shape[2]
    n_ang = 1 + 2 * n_multipoles

    eta, xi = posmap
    xi_rel = xi - xi0
    eta_rel = eta - eta0
    theta = np.arctan2(eta_rel, xi_rel)
    r = np.hypot(xi_rel, eta_rel)

    orig_shape = r.shape
    r_flat = r.ravel()
    theta_flat = theta.ravel()
    n_pixels = r_flat.size

    # Fourier basis for the angular dependence.
    F_mat = np.empty((n_ang, n_pixels), dtype=float)
    F_mat[0] = 1.0

    if n_multipoles:
        m = np.arange(1, n_multipoles + 1)[:, None]
        cos_m_theta = np.cos(m * theta_flat)
        sin_m_theta = np.sin(m * theta_flat)

        for i in range(n_multipoles):
            F_mat[1 + 2 * i] = cos_m_theta[i]
            F_mat[2 + 2 * i] = sin_m_theta[i]

    F_mat_T = F_mat.T

    # Construct the Bessel pair basis.
    b_terms = np.column_stack(
        [bessel_term_cached(r_flat, ell_max, n) for n in range(n_bessel)]
    )
    n0, n1 = np.triu_indices(n_bessel)
    base_beams = np.nan_to_num(
        b_terms[:, n0] * b_terms[:, n1],
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Reconstruct the pure Bessel/multipole core.
    core_flat = np.full(n_pixels, bessel_off, dtype=float)

    for pair_idx, (i0, i1) in enumerate(zip(n0, n1)):
        pair_beam = base_beams[:, pair_idx]

        for m_idx in range(n_multipoles):
            c = amps[i0, i1, m_idx, 0]
            if c != 0:
                angular = 1.0 if m_idx == 0 else cos_m_theta[m_idx - 1]
                core_flat += c * pair_beam * angular

            if m_idx > 0:
                s = amps[i0, i1, m_idx, 1]
                if s != 0:
                    core_flat += s * pair_beam * sin_m_theta[m_idx - 1]

    # Evaluate the wing and transition using the same implementation as
    # the fitting code.
    _, wing_weight = fast_wing_transition(
        r_flat,
        wing_params[:n_ang],
        wing_params[n_ang : 2 * n_ang],
        wing_params[-1],
        F_mat_T,
    )

    loga = F_mat_T @ wing_params[:n_ang]
    a = np.exp(loga)

    pure_wing = np.nan_to_num(
        a / np.maximum(r_flat, 1e-5) ** 3,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    model = ((1.0 - wing_weight) * core_flat + wing_weight * pure_wing + off).reshape(
        orig_shape
    )
    return ndmap(model.reshape(orig_shape), posmap.wcs)


def bessel_beam_from_aman(posmap, aman):
    return bessel_beam(
        posmap,
        aman.gauss.xi0.to(u.radian).value,
        aman.gauss.eta0.to(u.radian).value,
        aman.bessel.ell_max.value,
        aman.bessel.amps.value,
        aman.bessel.bessel_off.value,
        aman.bessel.wing_params.value,
        aman.bessel.off.value,
    )


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
