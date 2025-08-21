import numpy as np
import pprint
from so3g.proj import quat
from scipy.optimize import minimize

default_params = {
    'enc_offset_az': 0, 
    'enc_offset_el': 0,
    'enc_offset_cr': 0,
    'el_axis_center_xi0': 0,
    'el_axis_center_eta0': 0,
    'cr_center_xi0': 0,
    'cr_center_eta0': 0,
    'mir_center_xi0': 0,
    'mir_center_eta0': 0,
}

def _to_deg(pars):
    return {key: np.rad2deg(val) for key, val in  pars.items()}

def transform(x, az, el, roll, q_data):
    enc_offset_az, enc_offset_el, enc_offset_cr, el_axis_center_xi0, el_axis_center_eta0, cr_center_xi0, cr_center_eta0, mir_center_xi0, mir_center_eta0 = x
    _el = np.deg2rad(el)
    _az = np.deg2rad(az)
    _cr = np.deg2rad(el - roll - 60)

    q_lonlat = quat.rotation_lonlat(-(_az + enc_offset_az), enc_offset_el + _el, 0)
    q_lonlat_nom = quat.rotation_lonlat(-_az, _el)
    q_roll = quat.rotation_xieta(0, 0, 1 * np.deg2rad(roll))

    q_mir_center = ~quat.rotation_xieta(mir_center_xi0, mir_center_eta0)
    q_el_roll = quat.euler(2, _el + enc_offset_el - np.deg2rad(60))
    q_el_axis_center = ~quat.rotation_xieta(el_axis_center_xi0, el_axis_center_eta0)
    q_cr_roll = quat.euler(2, -1 * (_cr + enc_offset_cr))
    q_cr_center = ~quat.rotation_xieta(cr_center_xi0, cr_center_eta0)

    rhs = (
        q_lonlat
        * q_mir_center
        * q_el_roll 
        * q_el_axis_center
        * q_cr_roll
        * q_cr_center
    )
    lhs = q_lonlat_nom * q_roll
    rot = lhs * q_data * ~rhs
    xi0, eta0, _ = quat.decompose_xieta(rot)
    return xi0, eta0

def joint_transform(x, tmsks, pmsks, az, el, roll, q_data):
    xi0 = np.zeros(len(q_data)) + np.nan
    eta0 = np.zeros(len(q_data)) + np.nan
    for tmsk, pmsk in zip(tmsks, pmsks):
        if len(pmsk) == 0:
            continue
        xi0[tmsk], eta0[tmsk] = transform(x[pmsk], az[tmsk], el[tmsk], roll[tmsk], quat.G3VectorQuat(np.array(q_data)[tmsk]))
    return xi0, eta0

def fit_func(x, tmsks, pmsks, az, el, roll, q_data):
    xi0, eta0 = joint_transform(x, tmsks, pmsks, az, el, roll, q_data)
    rms = 3600*np.sqrt(np.nanmean(np.hstack([np.rad2deg(xi0), np.rad2deg(eta0)]) ** 2))
    return rms


def fit(time_ranges, ctime, az, el, roll, q_data, d):
    # Make masks
    tmsks = []
    pmsks = []
    q_datas = []
    orig_pars = np.array(list(default_params.keys()))
    par_mapping = np.argsort(np.argsort(orig_pars))
    par_list = orig_pars.copy()
    for t0, t1, to_fit, indep_list in time_ranges:
        tmsk = (ctime >= t0) * (ctime < t1)
        if not to_fit:
            tmsks += [tmsk]
            continue
        tmsk *= np.abs(d - np.median(d[tmsk])) < np.deg2rad(.25)
        tmsks += [tmsk]
        if np.sum(np.isin(indep_list, par_list)) != len(indep_list):
            raise ValueError(f"Invalid independant parameters in time range starting with {t0}")
        indep_list = [f"{n}+{t0}" for n in indep_list]
        par_list = np.hstack((par_list, indep_list))
    for t0, t1, to_fit, indep_list in time_ranges:
        if not to_fit:
            pmsks += [[]]
            continue
        pmsk = np.zeros(len(par_list), bool)
        pmsk[:len(orig_pars)] = True
        pmsk[np.isin(par_list, indep_list)] = False
        pmsk += np.array([str(t0) in par for par in par_list])
        if np.sum(pmsk) != len(default_params):
            raise ValueError(f"Time range starting with {t0} somehow has the wrong number of parameters!")
        # A boolean may have the wrong order, lets make it indexes and get it in order
        pmsk = np.where(pmsk)[0]
        mypars = np.array([par.split(f"+{t0}")[0] for par in par_list[pmsk]])
        srt = np.argsort(mypars)
        pmsks += [pmsk[srt][par_mapping]]

    x = np.zeros(len(par_list))
    bounds = [(-np.inf, np.inf)]*len(x)
    # bounds[3] = (0, 0)
    # bounds[4] = (0, 0)
    res = minimize(fit_func, x, args=(tmsks, pmsks, az, el, roll, q_data), bounds=bounds, method="Nelder-Mead", options={"adaptive": True}) #jac="3-point", method="trust-constr") #, ) #, method="TNC", options={"maxfun":10000})
    print(res)
    xi0, eta0 = joint_transform(res.x, tmsks, pmsks, az, el, roll, q_data)
    params = []
    for pmsk, (t0, t1, to_fit, _) in zip(pmsks, time_ranges):
        if not to_fit:
            params += [default_params]
            continue
        params += [{name.item() : val for name, val in zip(orig_pars, res.x[pmsk])}]
        print(f"{t0} to {t1}")
        pprint.pp(_to_deg(params[-1]))
    return params, xi0, eta0, tmsks 
