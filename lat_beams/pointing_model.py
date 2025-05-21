import numpy as np
from so3g.proj import quat

# lat_params = {
#     "az_offset": 1.49788300e-08,
#     "el_offset": -2.08882807e-01,
#     "roll_offset": 3.40844271e-02,
#     "xi_offset": 1.21677030e-01,
#     "eta_offset": -2.08870814e-01,
# }
lat_params_simple = {
    "az_offset": 4.03704816e-08,
    "el_offset": -1.39564042e-01,
    "cr_offset": -1.75482402e-01,
    "xi_offset": 1.40683504e-01,
    "eta_offset": -3.15087378e-01,
}

lat_params = {
    "az_offset": 4.03704816e-08,
    "el_offset": -1.39564042e-01,
    "cr_offset": -1.75482402e-01,
    "el_xi_offset": 1.40683504e-01,
    "el_eta_offset": -3.15087378e-01,
    "rx_xi_offset": 1.40683504e-01,
    "rx_eta_offset": -3.15087378e-01,
}

lat_params = {
    "az_offset": -6.431949463522369e-07,
    "el_offset": 0.10784213916209494,
    "cr_offset": 0.1068009204473848,
    "el_xi_offset": 0.07524197484976551,
    "el_eta_offset": 0.13704409641376075,
    "rx_xi_offset": -0.03523515856319279,
    "rx_eta_offset": -0.005858421274865713,
    "mir_xi_offset": -0.17792288910750848,
    "mir_eta_offset": 0.21440894087003562,
}


def apply_pointing_model(params, az, el, roll):
    cr = el - roll - np.deg2rad(60)
    _params = lat_params.copy()
    _params.update(params)
    params = _params
    for key, val in params.items():
        params[key] = 1 * np.deg2rad(val)
    q_enc = quat.rotation_lonlat(
        -1 * (az.copy() + params["az_offset"]), el.copy() + params["el_offset"]
    )
    q_mir = quat.rotation_xieta(params["mir_xi_offset"], params["mir_eta_offset"])
    q_el_roll = quat.euler(2, el.copy() + params["el_offset"] - np.deg2rad(60))
    q_tel = quat.rotation_xieta(params["el_xi_offset"], params["el_eta_offset"])
    q_cr_roll = quat.euler(2, -1 * cr - params["cr_offset"])
    q_rx = quat.rotation_xieta(params["rx_xi_offset"], params["rx_eta_offset"])
    new_az, el, roll = (
        quat.decompose_lonlat(q_enc * q_mir * q_el_roll * q_tel * q_cr_roll * q_rx)
        * np.array([-1, 1, 1])[..., None]
    )

    # Stolen from elle
    change = ((new_az - az) + np.pi) % (2 * np.pi) - np.pi
    az = az.copy() + change

    return az, el, roll


def apply_pointing_model_simple(params, az, el, roll):
    _params = lat_params_simple.copy()
    _params.update(params)
    params = _params
    for key, val in params.items():
        params[key] = -1 * np.deg2rad(val)
    q_enc = quat.rotation_lonlat(
        -1 * (az.copy() + params["az_offset"]), el.copy() + params["el_offset"]
    )
    q_tel = quat.rotation_xieta(params["xi_offset"], params["eta_offset"])
    q_roll = quat.euler(2, roll + params["el_offset"] - params["cr_offset"])
    new_az, el, roll = (
        quat.decompose_lonlat(q_enc * q_tel * q_roll) * np.array([-1, 1, 1])[..., None]
    )

    # Stolen from elle
    change = ((new_az - az) + np.pi) % (2 * np.pi) - np.pi
    az = az.copy() + change

    return az, el, roll
