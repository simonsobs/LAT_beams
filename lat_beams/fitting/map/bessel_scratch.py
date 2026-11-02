def empirical_bessel_noise_covariance(
    noise_stacks: Float[NDArray, "n_samp nx ny"],
    ivar: Float[NDArray, "nx ny"],
    posmap: Float[ndmap, "2 nx ny"],
    fit: AxisManager,
    guess: AxisManager,
    lmax: int,
    n_modes: int = -1,
    n_radial: int = 200,
    fit_kwargs: Optional[dict] = None,
    noise_ivar: Optional[Float[NDArray, "n_samp nx ny"]] = None,
) -> AxisManager:
    r"""
    Estimate beam-parameter, radial-profile, and beam-window covariance
    by fitting a beam injected into empirical noise-only stacks.

    Each realization is

        imap_k = noise_stacks[k] + beam_signal,

    where beam_signal is the best-fit model with its additive offset
    removed. Each realization is independently refitted with
    `fit_bessel_map`.

    The returned covariance matrices are sample covariances across
    realizations; they do not use the inverse Hessian from the individual
    fits.

    Parameters
    ----------
    noise_stacks : ndarray, shape (n_samp, nx, ny)
        Independent noise-only stacked cutouts. Each must have the same
        pixel geometry and units as the original planet stack.
    ivar : ndarray, shape (nx, ny)
        Inverse-variance map used for fitting. Ignored if `noise_ivar`
        is provided.
    posmap : ndmap, shape (2, nx, ny)
        Coordinate maps in radians, as used for the original fit.
    fit : AxisManager
        Best-fit output from `fit_bessel_map`, for the planet stack.
        Used to construct the injected beam signal.
    guess : AxisManager
        Initial guess passed to each call of `fit_bessel_map`.
        Prefer the original planet-fit guess, or a consistent guess
        close to the best-fit solution.
    lmax : int
        Maximum multipole for the beam window function.
    n_modes : int, optional
        Number of profile covariance eigenmodes to retain. If <= 0,
        retain all available modes.
    n_radial : int, optional
        Number of radial bins.
    fit_kwargs : dict, optional
        Additional keyword arguments for `fit_bessel_map`, e.g.
        `n_bessel`, `n_multipoles`, `mask_size`, `skip_multipoles`,
        and `n_opt_pixels`. `calc_cov` is always disabled.
    noise_ivar : ndarray, shape (n_samp, nx, ny), optional
        Observation-matched inverse-variance maps for each noise stack.
        If omitted, the same `ivar` is used for all realizations.

    Returns
    -------
    out : AxisManager
        Empirical covariance products. Fields include:

        - `param_samples`, `param_cov`: fitted parameter samples/covariance.
        - `profile_samples`, `profile_cov`: normalized radial profiles
          and their sample covariance.
        - `bl_samples`, `bl_cov`: window-function samples/covariance.
        - `profile_sigma`, `bl_sigma`: marginal 1-sigma uncertainties.
        - `profile_modes`, `bl_modes`: retained covariance modes with
          their 1-sigma amplitudes.
        - `fit_success`, `fit_chi2`: per-realization diagnostics.
        - `profile_mean`, `bl_mean`: sample means.
        - `n_samp`, `n_modes`, `lmax`: metadata.

    Notes
    -----
    The covariance describes statistical variation under the supplied
    noise-stack ensemble and the chosen fitting procedure. The noise
    stacks must reproduce the relevant mapmaking, registration, and
    stacking operations. This function does not estimate mapmaker bias.
    """
    noise_stacks = np.asarray(noise_stacks, dtype=float)
    ivar = np.asarray(ivar, dtype=float)

    if noise_stacks.ndim != 3:
        raise ValueError("noise_stacks must have shape (n_samp, nx, ny).")
    n_samp, nx, ny = noise_stacks.shape
    if n_samp < 2:
        raise ValueError("At least two noise stacks are required.")
    if ivar.shape != (nx, ny):
        raise ValueError("ivar must have shape (nx, ny).")
    if tuple(posmap.shape[-2:]) != (nx, ny):
        raise ValueError("posmap and noise_stacks have incompatible shapes.")

    if noise_ivar is not None:
        noise_ivar = np.asarray(noise_ivar, dtype=float)
        if noise_ivar.shape != noise_stacks.shape:
            raise ValueError("noise_ivar must have shape (n_samp, nx, ny).")

    kwargs = dict(fit_kwargs or {})
    if "calc_cov" in kwargs:
        raise ValueError("Do not pass calc_cov in fit_kwargs; it is set to False.")

    # Construct the injected signal from the original best-fit model.
    xi0 = float(fit.xi0.to(u.rad).value)
    eta0 = float(fit.eta0.to(u.rad).value)
    ell_max = float(fit.ell_max.value)
    amps = np.asarray(fit.amps.value)
    wing_params = np.asarray(fit.wing_params.value)
    off = float(np.asarray(fit.off.value))

    beam_model = bessel_beam(
        posmap,
        xi0,
        eta0,
        ell_max,
        amps,
        0.0,
        wing_params,
        off,
    )
    beam_signal = np.asarray(beam_model, dtype=float) - off

    # Coordinate setup for consistent radial profiling.
    profiles = []
    bl_samples = []
    param_samples = []
    success = []
    chi2 = []
    radial_centers = None

    for i in range(n_samp):
        this_ivar = ivar if noise_ivar is None else noise_ivar[i]
        imap = noise_stacks[i] + beam_signal

        # Refit the injected beam with the same fitting configuration.
        fitted, model = fit_bessel_map(
            imap=ndmap(imap, posmap.wcs),
            ivar=ndmap(this_ivar, posmap.wcs),
            posmap=posmap,
            guess=guess,
            calc_cov=False,
            **kwargs,
        )

        # Match the parameter ordering used by bessel_profile_covariance.
        params = np.concatenate(
            [
                np.asarray(fitted.linear_coeffs.value),
                np.asarray(fitted.wing_params.value),
            ]
        )
        param_samples.append(params)
        success.append(bool(fitted.wing_success))
        chi2.append(float(fitted.chi2))

        # Remove the fitted offset and normalize each profile to unit peak.
        model_no_off = np.asarray(model, dtype=float) - float(
            np.asarray(fitted.off.value)
        )
        radial_centers_i, profile, _ = radial_profile_lin(
            ndmap(model_no_off, posmap.wcs),
            posmap,
            xi0=float(fitted.xi0.to(u.rad).value),
            eta0=float(fitted.eta0.to(u.rad).value),
            n_bins=n_radial,
        )

        peak = np.nanmax(model_no_off)
        if not np.isfinite(peak) or peak <= 0:
            raise RuntimeError(f"Invalid fitted beam peak in noise realization {i}.")
        profile = np.asarray(profile, dtype=float) / peak

        if radial_centers is None:
            radial_centers = np.asarray(radial_centers_i, dtype=float)
        elif not np.allclose(radial_centers_i, radial_centers):
            raise RuntimeError(
                "Radial bins differ between realizations; "
                "use a common fixed radial grid."
            )

        profiles.append(profile)
        bl_samples.append(np.asarray(beam2bl(profile, radial_centers, lmax=lmax)))
    if radial_centers is None:
        raise ValueError("Radial centers is None?")

    profiles = np.asarray(profiles)
    bl_samples = np.asarray(bl_samples)
    param_samples = np.asarray(param_samples)
    success = np.asarray(success, dtype=bool)
    chi2 = np.asarray(chi2)

    # Empirical sample covariances.
    profile_mean = np.mean(profiles, axis=0)
    bl_mean = np.mean(bl_samples, axis=0)
    param_mean = np.mean(param_samples, axis=0)

    profile_centered = profiles - profile_mean
    bl_centered = bl_samples - bl_mean
    param_centered = param_samples - param_mean

    profile_cov = profile_centered.T @ profile_centered / (n_samp - 1)
    bl_cov = bl_centered.T @ bl_centered / (n_samp - 1)
    param_cov = param_centered.T @ param_centered / (n_samp - 1)

    profile_sigma = np.sqrt(np.maximum(0.0, np.diag(profile_cov)))
    bl_sigma = np.sqrt(np.maximum(0.0, np.diag(bl_cov)))

    # Profile covariance eigenmodes, ordered by descending variance.
    evals, evecs = np.linalg.eigh(profile_cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]

    if n_modes <= 0:
        n_modes_actual = len(evals)
    else:
        n_modes_actual = min(n_modes, len(evals))

    evals = evals[:n_modes_actual]
    evecs = evecs[:, :n_modes_actual]

    profile_modes = evecs * np.sqrt(evals)[None, :]

    # Propagate the same empirical modes through the linear beam2bl map.
    ell = np.arange(lmax + 1)
    bl_modes = np.empty((len(ell), n_modes_actual))
    for j in range(n_modes_actual):
        bl_modes[:, j] = np.asarray(
            beam2bl(profile_modes[:, j], radial_centers, lmax=lmax)
        )

    # Package results.
    r_ax = IndexAxis("r", len(radial_centers))
    ell_ax = IndexAxis("ell", len(ell))
    mode_ax = IndexAxis("mode", n_modes_actual)
    param_ax = IndexAxis("param", param_samples.shape[1])
    samp_ax = IndexAxis("sample", n_samp)

    out = AxisManager(r_ax, ell_ax, mode_ax, param_ax, samp_ax)

    out.wrap(
        "r",
        3600 * np.rad2deg(radial_centers),
        [(0, "r")],
    )
    out.wrap("ell", ell, [(0, "ell")])
    out.wrap("profile_mean", profile_mean, [(0, "r")])
    out.wrap("profile_sigma", profile_sigma, [(0, "r")])
    out.wrap("profile_cov", profile_cov, [(0, "r"), (1, "r")])
    out.wrap("profile_samples", profiles, [(0, "sample"), (1, "r")])
    out.wrap("profile_modes", profile_modes, [(0, "r"), (1, "mode")])
    out.wrap("cov_eigenvalues", evals, [(0, "mode")])
    out.wrap("cov_eigenvectors", evecs, [(0, "r"), (1, "mode")])

    out.wrap("bl_mean", bl_mean, [(0, "ell")])
    out.wrap("bl_sigma", bl_sigma, [(0, "ell")])
    out.wrap("bl_cov", bl_cov, [(0, "ell"), (1, "ell")])
    out.wrap("bl_samples", bl_samples, [(0, "sample"), (1, "ell")])
    out.wrap("bl_modes", bl_modes, [(0, "ell"), (1, "mode")])

    out.wrap("param_mean", param_mean, [(0, "param")])
    out.wrap("param_cov", param_cov, [(0, "param"), (1, "param")])
    out.wrap("param_samples", param_samples, [(0, "sample"), (1, "param")])
    out.wrap("fit_success", success, [(0, "sample")])
    out.wrap("fit_chi2", chi2, [(0, "sample")])
    out.wrap("n_samp", n_samp)
    out.wrap("n_modes", n_modes_actual)
    out.wrap("lmax", lmax)

    return out
