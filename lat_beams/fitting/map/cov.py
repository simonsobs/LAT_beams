from typing import Optional, cast

import numpy as np
from healpy.sphtfunc import beam2bl
from jaxtyping import Float
from numpy.typing import NDArray
from pixell import enmap
from pixell.enmap import ndmap
from sotodlib.core import AxisManager, IndexAxis


def estimate_noise_autocov(
    noise_cutouts: Float[ndmap, "ncutouts ny nx"],
    max_lag: int = 30,
) -> Float[np.ndarray, "{max_lag}*2+1 {max_lag}*2+1"]:
    r"""
    Estimate the spatial noise autocovariance from noise-only image cutouts.

    Each cutout is mean-subtracted, and its spatial autocorrelation is
    computed using the 2D Fourier transform. The autocorrelations are
    averaged across cutouts and normalized by the number of overlapping
    pixel pairs at each lag.

    Assuming spatially stationary noise, the autocovariance at lag
    (dy, dx) is estimated as

    $$
    \widehat{C}(dy, dx) =
    \frac{1}{K\,n_{\mathrm{pairs}}(dy, dx)}
    \sum_{k=1}^{K}
    \sum_{(y,x)\in P(dy,dx)}
    X_k(y,x) X_k(y+dy,x+dx),
    $$

    where $X_k$ is the mean-subtracted noise in cutout $k$, $P(dy,dx)$
    contains pixel positions where both pixels lie within the cutout, and
    $n_{\mathrm{pairs}}$ is the number of such pairs per cutout. For
    rectangular cutouts,

    $$
    n_{\mathrm{pairs}}(dy, dx) =
    (n_y-|dy|)(n_x-|dx|).
    $$

    The FFT calculation uses zero-padding to avoid periodic wraparound
    correlations. The estimate assumes that the cutouts share a common
    autocovariance function and contain no significant residual sources
    or spatially varying background structure.

    Subtracting each cutout's sample mean can introduce bias because the
    estimated mean is itself noisy. Let $M=n_y n_x$ be the number of
    pixels in a cutout, $N_i$ the noise at pixel $i$, and

    $$
    X_i = N_i - \overline{N},
    \qquad
    \overline{N} = \frac{1}{M}\sum_{i=1}^{M} N_i.
    $$

    If the true noise covariance is $C_{ij}=\\mathbb{E}[N_iN_j]$ and
    the noise has zero mean, then the covariance after mean subtraction is

    $$
    \mathbb{E}[X_iX_j]
    =
    C_{ij}
    -\frac{1}{M}\sum_a C_{ia}
    -\frac{1}{M}\sum_b C_{bj}
    +\frac{1}{M^2}\sum_{a,b} C_{ab}.
    $$

    Thus, the estimated covariance can differ from the true covariance,
    particularly for small cutouts or noise with long-range correlations.
    For independent, identically distributed noise with variance
    $\sigma^2$, mean subtraction reduces the expected variance to
    $\sigma^2(1-1/M)$ and introduces an expected covariance of
    $-\sigma^2/M$ between distinct pixels.

    The FFT-based estimate does not correct for this bias. For large
    cutouts with short-range correlations, the effect is often small.
    Correcting it for correlated noise generally requires a model of the
    underlying covariance.

    Parameters
    ----------
    noise_cutouts: Float[ndmap, "ncutouts ny nx"]
        Noise-only image cutouts, should be the same geometry as my map.
    max_lag : int, default=30
        Maximum absolute pixel lag along each axis. The output includes
        lags from `-max_lag` to `+max_lag` in both directions.

    Returns
    -------
    cov : Float[np.ndarray, "{2 * max_lag + 1} {2 * max_lag + 1}"]
        Estimated spatial autocovariance. The central element,
        `cov[max_lag, max_lag]`, is the noise variance. Nonzero lags
        describe covariance between pixels at the corresponding offsets.
        `max_lag` sets the maximum displacement measured, not the noise
        correlation length.
    """
    cutouts = np.asarray(noise_cutouts, dtype=float)
    K, ny, nx = cutouts.shape

    # Remove each cutout's mean.
    cutouts = cutouts - np.mean(cutouts, axis=(-2, -1), keepdims=True)

    # Zero-pad to avoid periodic wraparound correlations.
    shape = (2 * ny, 2 * nx)
    power = np.zeros(shape, dtype=float)

    for im in cutouts:
        ft = np.fft.fft2(im, s=shape)
        power += np.fft.ifft2(ft * ft.conj()).real

    # Number of overlapping pixel pairs at each lag.
    overlap = np.zeros(shape, dtype=float)
    overlap[:ny, :nx] = 1.0
    overlap = np.fft.ifft2(np.fft.fft2(overlap) * np.fft.fft2(overlap)).real

    # The overlap array above is for a rectangular support.
    # Compute the lag normalization directly for clarity.
    corr = power / K
    out = np.zeros((2 * max_lag + 1, 2 * max_lag + 1))

    for dy in range(-max_lag, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            # FFT correlation at positive/negative lags.
            iy = dy if dy >= 0 else shape[0] + dy
            ix = dx if dx >= 0 else shape[1] + dx

            n_pairs = (ny - abs(dy)) * (nx - abs(dx))
            out[dy + max_lag, dx + max_lag] = corr[iy, ix] / n_pairs

    return out


def empirical_profile_covariance(
    all_prof: Float[NDArray, "n_samp n_radial"],
    radial_centers: Float[NDArray, "n_radial"],
    lmax: int,
    n_modes: int = -1,
) -> Optional[AxisManager]:
    r"""
    Estimate profile and beam-window covariance directly from an ensemble
    of normalized radial profiles.

    This is the sample-space analogue of `bessel_profile_covariance`.
    Rather than estimating a covariance in the potentially degenerate
    parameter space, the covariance is estimated directly from the
    ensemble of model profiles.

    Parameters
    ----------
    all_prof : Float[NDArray, "n_samp n_radial"]
        Ensemble of normalized radial profile realizations. Each row is
        one realization and each column corresponds to `radial_centers`.
    radial_centers : Float[NDArray, "n_radial"]
        Radial coordinates corresponding to the profile bins, in radians.
    lmax : int
        Maximum multipole for the beam window function.
    n_modes : int, optional
        Number of covariance modes to retain in `profile_modes` and
        `bl_modes`. If <= 0, all modes are retained.

    Returns
    -------
    out : Optional[AxisManager]
        AxisManager with a structure matching `bessel_profile_covariance`
        as closely as possible.

        Fields include:

        - `r`: radial bin centers.
        - `profile`: mean normalized profile.
        - `profile_sigma`: empirical 1-sigma profile uncertainty.
        - `profile_cov`: empirical profile covariance.
        - `profile_modes`: covariance modes including their 1-sigma
          amplitudes.
        - `ell`: multipoles from 0 through `lmax`.
        - `bl`: beam window function of the mean profile.
        - `bl_sigma`: empirical 1-sigma window-function uncertainty.
        - `bl_modes`: covariance modes of the window function including
          their 1-sigma amplitudes.
        - `cov_eigenvalues`: profile covariance eigenvalues.
        - `n_modes`: number of retained covariance modes.
        - `n_samp`: number of profile realizations.

        The eigenvalues/eigenvectors correspond to the profile-space
        covariance, rather than the original parameter-space covariance.
        If `len(all_prof) == 1` then `None` is returned.
    """
    all_prof = np.asarray(all_prof, dtype=float)
    radial_centers = np.asarray(radial_centers, dtype=float)
    n_samp, n_radial = all_prof.shape

    if n_samp < 2:
        return None
    if len(radial_centers) != n_radial:
        raise ValueError(
            "all_prof and radial_centers have incompatible shapes: "
            f"{n_radial} != {len(radial_centers)}"
        )
    profile_mean = np.mean(all_prof, axis=0)
    centered = all_prof - profile_mean[None, :]
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    evals_full = s**2 / (n_samp - 1)
    evecs_full = Vt.T

    if n_modes <= 0:
        n_modes_actual = len(evals_full)
    else:
        n_modes_actual = min(n_modes, len(evals_full))

    evals = evals_full[:n_modes_actual]
    evecs = evecs_full[:, :n_modes_actual]
    profile_modes = evecs * np.sqrt(evals)[None, :]
    profile_sigma = np.sqrt(np.sum(profile_modes**2, axis=1))

    ell = np.arange(lmax + 1)
    bl = np.asarray(beam2bl(profile_mean, radial_centers, lmax=lmax))
    bl_modes = np.empty((len(ell), n_modes_actual), dtype=float)

    for i in range(n_modes_actual):
        bl_modes[:, i] = np.asarray(
            beam2bl(
                profile_modes[:, i],
                radial_centers,
                lmax=lmax,
            )
        )
    bl_sigma = np.sqrt(np.sum(bl_modes**2, axis=1))

    r_ax = IndexAxis("r", n_radial)
    ell_ax = IndexAxis("ell", len(ell))
    mode_ax = IndexAxis("mode", n_modes_actual)
    out = AxisManager(r_ax, ell_ax, mode_ax)
    out.wrap("r", 3600 * np.rad2deg(radial_centers), [(0, "r")])
    out.wrap("profile", profile_mean, [(0, "r")])
    out.wrap("profile_sigma", profile_sigma, [(0, "r")])
    out.wrap("profile_modes", profile_modes, [(0, "r"), (1, "mode")])
    out.wrap("ell", ell, [(0, "ell")])
    out.wrap("bl", bl, [(0, "ell")])
    out.wrap("bl_sigma", bl_sigma, [(0, "ell")])
    out.wrap("bl_modes", bl_modes, [(0, "ell"), (1, "mode")])
    out.wrap("cov_eigenvalues", evals, [(0, "mode")])
    out.wrap("cov_eigenvectors", evecs, [(0, "r"), (1, "mode")])
    out.wrap("n_modes", n_modes_actual)
    out.wrap("n_samp", n_samp)
    out.wrap("lmax", lmax)

    return out


def estimate_noise_ps2d(
    residual: Float[ndmap, "ny nx"],
    ivar: Float[ndmap, "ny nx"],
    guess: AxisManager,
    window: Optional[Float[ndmap, "ny nx"]] = None,
    smooth_scale: float = 1.0,
    floor_fraction: float = 1e-3,
) -> Float[ndmap, "ny nx"]:
    r"""
    Estimate a regularized two-dimensional noise power spectrum.

    Whiten the map residual using the per-pixel inverse variance, estimate
    its two-dimensional power spectrum, and smooth the spectrum using a
    scale derived from the nominal beam width. Non-finite and non-positive
    spectral values are replaced by a positive floor defined relative to
    the estimated white-noise reference level.

    Parameters
    ----------
    residual : Float[ndmap, "ny nx"]
        Full-map data minus the fitted model, in map units.
    ivar : Float[ndmap, "ny nx"]
        Per-pixel inverse variance, in inverse squared map units. Pixels
        with non-finite or non-positive inverse variance are excluded from
        the noise estimate.
    guess : AxisManager
        Beam estimate containing `fwhm_xi` and `fwhm_eta`. Both quantities
        must support conversion to radians using `to_value("rad")`.
    window : Optional[Float[ndmap, "ny nx"]], default=None
        Optional apodization window. If `None`, a binary validity window
        is constructed from `residual` and `ivar`. Otherwise, the supplied
        window is multiplied by the validity mask.
    smooth_scale : float, default=1.0
        Dimensionless multiplier for the beam-derived spectral smoothing
        width.
    floor_fraction : float, default=1e-3
        Minimum spectral power as a fraction of the estimated white-noise
        reference level. Must be positive.

    Returns
    -------
    corr_noise : Float[ndmap, "ny nx"]
        Regularized two-dimensional noise power spectrum, represented as
        an `ndmap` with the WCS of `residual`.

    Raises
    ------
    ValueError
        If `floor_fraction` is not positive or no positive finite power
        is found in the residual spectrum.
    """
    valid = np.isfinite(residual) * np.isfinite(ivar) * (ivar > 0)

    if window is None:
        window = enmap.ndmap(valid.astype(float), residual.wcs)
    else:
        window = cast(enmap.ndmap, window * valid)

    # Whiten the residual using the per-pixel noise amplitude
    whitened = enmap.ndmap(
        np.where(valid, residual * np.sqrt(np.maximum(ivar, 0)), 0), residual.wcs
    )

    # Remove the weighted mean to reduce DC contamination
    weights = np.asarray(window)
    z = np.asarray(whitened)
    if np.sum(weights) > 0:
        z = z - np.sum(weights * z) / np.sum(weights)

    whitened = enmap.ndmap(z * weights, residual.wcs)

    harm = enmap.map2harm(whitened)  # , normalize="phys")
    ps2d = enmap.calc_ps2d(harm)

    # Estimate a smoothing scale from the nominal beam FWHM
    fwhm_xi = float(guess.fwhm_xi.to_value("rad"))
    fwhm_eta = float(guess.fwhm_eta.to_value("rad"))
    fwhm = 0.5 * (fwhm_xi + fwhm_eta)

    ly, lx = enmap.laxes(residual.shape, residual.wcs)
    dly = float(np.median(np.abs(np.diff(ly))))
    dlx = float(np.median(np.abs(np.diff(lx))))

    sigma_beam = fwhm / np.sqrt(8 * np.log(2))
    sigma_y = 1.0 / max(sigma_beam * dly, 1e-12)
    sigma_x = 1.0 / max(sigma_beam * dlx, 1e-12)
    smooth_width = smooth_scale * np.sqrt(sigma_x * sigma_y)

    ps2d = enmap.smooth_spectrum(ps2d, width=smooth_width)

    ps = np.asarray(ps2d)
    good = np.isfinite(ps) & (ps > 0)
    if not np.any(good):
        raise ValueError("No positive finite power found in residual spectrum.")

    # Whitened residuals have unit white-noise variance
    white_level = np.median(ps[good])
    floor = floor_fraction * white_level

    ps = np.nan_to_num(ps, nan=floor, posinf=floor, neginf=floor)
    ps = np.maximum(ps, floor)
    ps /= np.median(ps)

    return enmap.ndmap(ps, residual.wcs)


def apply_harm_noise(x, ivar, ps2d, include_corr):
    r"""
    Apply the inverse covariance for the approximate noise model

    $$
    N = D N_{\mathrm{harm}} D,
    $$

    where $D$ is the diagonal map-domain noise-amplitude operator and
    $N_{\mathrm{harm}}$ is the harmonic-space covariance represented by
    `ps2d`.

    The inverse covariance is applied by scaling the input map using the
    per-pixel inverse noise amplitude, optionally filtering it in harmonic
    space, and applying the map-domain scaling a second time.

    Parameters
    ----------
    x : Float[ndmap, "ny nx ncomp"]
        Input map with spatial dimensions `ny` and `nx` and a trailing
        component dimension `ncomp`.
    ivar : Float[ndmap, "ny nx"]
        Per-pixel inverse variance, in inverse squared map units. Pixels
        with non-finite or non-positive inverse variance receive zero
        weight.
    ps2d : Float[ndmap, "ny nx"]
        Two-dimensional noise power spectrum. Non-finite and non-positive
        values are excluded when constructing the inverse spectrum. Its
        shape and WCS must be compatible with the harmonic representation
        of `x`.
    include_corr : bool
        If `True`, apply the harmonic-space inverse power spectrum. If
        `False`, omit harmonic filtering and apply only the map-domain
        inverse-variance weighting.

    Returns
    -------
    Ninv_x : Float[ndmap, "ny nx ncomp"]
        Map obtained by applying the approximate inverse noise covariance to `x`.
    """
    # Inverse harmonic covariance
    inv_ps2d = np.zeros_like(ps2d)
    valid_ps = np.isfinite(ps2d) & (ps2d > 0)
    inv_ps2d[valid_ps] = 1.0 / ps2d[valid_ps]

    # D = sqrt(1/ivar)
    D_inv = np.zeros_like(ivar)
    valid_noise = np.isfinite(ivar) & (ivar > 0)
    D_inv[valid_noise] = np.sqrt(ivar[valid_noise])

    x_scaled = x * D_inv[..., None]

    # Apply the inverse harmonic noise covariance
    if include_corr:
        x_harm = enmap.map2harm(x_scaled)  # , normalize="phys")
        x_filtered_harm = x_harm * inv_ps2d[..., None]
        x_filtered = enmap.harm2map(x_filtered_harm)  # , normalize="phys")
    else:
        x_filtered = x_scaled

    # Apply the second D^{-1}
    Ninv_x = x_filtered * D_inv[..., None]

    return Ninv_x
