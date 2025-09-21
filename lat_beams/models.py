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


def guass_multipole_beam(
    x, y, multipoles, dx, dy, off, amps, fwhm_xis, fwhm_etas, phis
):
    # r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y - dy, x - dx)

    xieta = (x, y)
    beam_model = gaussian2d(
        xieta, amps[0], dx, dy, fwhm_xis[0], fwhm_etas[0], phis[0], 0
    )
    for m, n in enumerate(multipoles):
        order = 2 ** (n - 1)
        for i, op in enumerate((np.sin,)):
            mp = op(theta * order)
            j = 2 * m + i + 1
            beam_model += (
                gaussian2d(
                    xieta, amps[j], dx, dy, fwhm_xis[j], fwhm_etas[j], phis[j], 0
                )
                * mp
            )
    return beam_model + off


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
