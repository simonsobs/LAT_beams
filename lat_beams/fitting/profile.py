import logging

import numpy as np
from astropy import units as u
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.special import spherical_jn
from sotodlib.tod_ops.filters import logger as flog

from .models import dr4_beam, scatter_beam

flog.setLevel(logging.ERROR)


def fit_dr4_profile(r, rprof, fwhm, d, lmd, sang, corr, eps, r_calc, max_modes=30):
    # Assuming unitful profiles here...
    # TODO: Covariance and scattering
    r_rad = r.to(u.radian).value
    r_calc_rad = r_calc.to(u.radian).value
    prof = rprof.value
    ell_0 = (2 * np.pi * d / lmd).decompose().value
    r_0 = fwhm.to(u.radian).value / 2.355
    par_names = ["ell_max", "r_c", "alpha", "off"]
    par_units = [u.dimensionless_unscaled, r.unit, rprof.unit, rprof.unit]
    par_units_fit = [u.dimensionless_unscaled, u.radian, rprof.unit, rprof.unit]
    guess = [1, 5000, 1, 0]
    bounds = [(0.7, 1.3), (5000, 5000), (0, 0), (0, 0)]

    # Fix the number of modes to half the data points within 5 sigma and setup amps
    n_modes = min(max_modes, int(np.sum(r_rad < 5 * r_0) / 2))
    par_names += [f"amp_{i}" for i in range(n_modes)]
    par_units += [rprof.unit] * n_modes
    par_units_fit += [rprof.unit] * n_modes
    guess += [0] * n_modes
    bounds += [(-np.inf, np.inf)] * n_modes

    # Setup scatter beam
    scatter_pars = {
        "n_terms": 5,
        "lmd": lmd.to(u.m).value,
        "sang": sang.to(u.sr).value,
        "corr": corr.to(u.m).value,
        "eps": np.sqrt(2) * eps.to(u.m).value,
    }

    def _dr4_model(pars, r_use, ell_0, r_0, scatter_pars):
        ell_max, r_c, alpha, off = pars[:4]
        amps = pars[4:]
        model = dr4_beam(
            r_use, ell_max * ell_0, r_c * r_0, alpha, off, amps, scatter_pars
        )
        return model

    def _dr4_objective(pars, prof, r_use, ell_0, r_0, scatter_pars):
        model = _dr4_model(pars, r_use, ell_0, r_0, scatter_pars)
        return np.nansum((prof - model) ** 2) * 1e10

    res = minimize(
        _dr4_objective,
        guess,
        bounds=bounds,
        args=(prof, r_rad, ell_0, r_0, scatter_pars),
        options={"maxcor": 10, "maxfun": 100000},
    )
    if res.success is False:
        return None, None, None
    model = _dr4_model(res.x, r_rad, ell_0, r_0, scatter_pars)
    frac_res = abs(rprof - model) / model
    bad_fit = (frac_res >= 1) * (r_rad > 5 * r_0)
    guess = res.x
    guess[1] = min(5, 0.9 * np.max(r_rad) / r_0)
    if np.sum(bad_fit) > 0:
        r_c = r_rad[int(np.percentile(np.where(bad_fit)[0], 5))]
        guess[1] = r_c / r_0
    bounds[0] = (0.1, 10.0)
    bounds[1] = (guess[1] * 0.7, np.max(r_rad) / r_0)
    bounds[2] = (0, np.inf)
    bounds[3] = (-np.inf, np.inf)
    res = minimize(
        _dr4_objective,
        guess,
        bounds=bounds,
        args=(prof, r_rad, ell_0, r_0, scatter_pars),
        options={"maxcor": 10, "maxfun": 100000},
    )

    model = _dr4_model(res.x, r_calc_rad, ell_0, r_0, scatter_pars)
    model_oto = _dr4_model(res.x, r_rad, ell_0, r_0, scatter_pars)
    pars = res.x
    pars[0] *= ell_0
    pars[1] *= r_0
    params = {
        n: (v * uf).to(un)
        for n, un, uf, v in zip(par_names, par_units, par_units_fit, pars)
    }
    # TODO: include uncertainties here
    mprofile = np.column_stack((r_calc.value, model))
    mprofile_oto = np.column_stack((r.value, model_oto))

    return mprofile, params, mprofile_oto


def fit_bessel_profile(r, rprof, fwhm, d, lmd, sang, corr, eps, r_calc, max_modes=100):
    # TODO: Move model to models.py
    # Assuming unitful profiles here...
    r_rad = r.to(u.radian).value
    r_calc_rad = r_calc.to(u.radian).value
    prof = rprof.value
    ell_0 = (2 * np.pi * d / lmd).decompose().value
    r_0 = fwhm.to(u.radian).value / 2.355

    n_modes = max_modes
    par_names = [f"amp_{i}" for i in range(n_modes)]
    par_units = [rprof.unit] * n_modes
    par_units_fit = [rprof.unit] * n_modes
    r_msk = r > 0
    amps = np.zeros(n_modes)
    chisq = np.nansum(rprof**2)
    rprof_use = rprof[r_msk]
    r_use = r_rad[r_msk] * ell_0
    r_calc_use = r_calc_rad * ell_0
    model_oto = np.zeros_like(r_use)
    model = np.zeros_like(r_calc_use)
    for i in range(n_modes):
        term = (spherical_jn(i, r_use) / r_use) ** 2
        amp = np.nansum(term * (rprof_use - model_oto)) / np.nansum(term**2)
        if not np.isfinite(amp):
            continue
        new_chisq = np.nansum((rprof_use - model_oto - amp * term) ** 2)
        if new_chisq > chisq:
            continue
        amps[i] = amp
        model_oto += amp * term
        with np.errstate(divide="ignore", invalid="ignore"):
            model += amp * (spherical_jn(i, r_calc_use) / r_calc_use) ** 2
        chisq = new_chisq
    model_oto = np.insert(model_oto, 0, 1)
    interp = PchipInterpolator(r_rad[model_oto <= 1], model_oto[model_oto <= 1])
    model[r_calc_use == 0] = 1
    interp_msk = (r_calc_rad > 0) * (r_calc_rad <= np.min(r_rad[r_rad > 0]))
    model[interp_msk] = interp(r_calc_rad[interp_msk])
    model_oto[model_oto > 1] = interp(r_rad[model_oto > 1])
    pars = amps

    # Setup scatter beam
    scatter_pars = {
        "n_terms": 5,
        "lmd": lmd.to(u.m).value,
        "sang": sang.to(u.sr).value,
        "corr": corr.to(u.m).value,
        "eps": np.sqrt(2) * eps.to(u.m).value,
    }

    # Fit for a symmetric r^-3 wing
    wing_model_oto = model_oto.copy()

    def _wing(coeffs, wmodel, beam_model, r_use):
        r0_msk = r_use == 0
        rc, alpha, off_wing, off_core = coeffs
        wmodel[(r_use <= rc) * ~r0_msk] = beam_model[(r_use <= rc) * ~r0_msk] + off_core
        wmodel[r_use > rc] = off_wing + alpha * (rc**3) / np.power(r_use[r_use > rc], 3)
        wmodel[r_use > rc] += scatter_beam(r_use[r_use > rc], **scatter_pars)
        return wmodel

    def _wing_obj(coeffs):
        wing_model = _wing(coeffs, wing_model_oto, model_oto, r_rad)

        return np.nansum((rprof - wing_model) ** 2)

    # Initial guess of rc
    mask_size = np.max(r_rad)
    rc = min(7 * r_0, 0.9 * mask_size)
    guess = [rc, 0, 0, 0]
    bounds = [(rc * 0.5, mask_size), (0, np.inf), (0, 0), (0, 0)]
    res = minimize(_wing_obj, guess, bounds=bounds)
    if res.success:
        r_c, alpha, off_wing, off_core = res.x
    else:
        r_c, alpha, off_wing, off_core = np.inf, 0, 0, 0

    model_oto = _wing((r_c, alpha, off_wing, off_core), model_oto, model_oto, r_rad)
    model = _wing((r_c, alpha, off_wing, off_core), model, model, r_calc_rad)
    par_names += ["r_c", "alpha", "off_wing", "off_core"]
    par_units += [r.unit, u.dimensionless_unscaled, rprof.unit, rprof.unit]
    par_units_fit += [u.radian, u.dimensionless_unscaled, rprof.unit, rprof.unit]
    pars = np.hstack((amps, (r_c, alpha, off_wing, off_core)))

    params = {
        n: (v * uf).to(un)
        for n, un, uf, v in zip(par_names, par_units, par_units_fit, pars)
    }
    mprofile = np.column_stack((r_calc.value, model))
    mprofile_oto = np.column_stack((r.value, model_oto))

    return mprofile, params, mprofile_oto
