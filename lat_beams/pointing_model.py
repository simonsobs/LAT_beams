from so3g.proj import quat
import numpy as np

# lat_params = {
#     "az_offset": 1.49788300e-08,
#     "el_offset": -2.08882807e-01,
#     "roll_offset": 3.40844271e-02,
#     "xi_offset": 1.21677030e-01,
#     "eta_offset": -2.08870814e-01,
# }
lat_params = {
    "az_offset": 4.03704816e-08,
    "el_offset": -1.39564042e-01,
    "roll_offset": -1.75482402e-01,
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


def apply_pointing_model(params, az, el, roll):
    cr = np.deg2rad(el - roll - np.deg2rad(60))
    _params = lat_params.copy()
    _params.update(params)
    params = _params
    for key, val in params.items():
        params[key] = -1 * np.deg2rad(val)
    # param keys: az_offset, el_offset, roll_offset, xi_offset, eta_offset
    q_enc = quat.rotation_lonlat(
        -1 * (az.copy() + params["az_offset"]), el.copy() + params["el_offset"]
    )
    q_el_roll = quat.euler(2, el.copy() - np.deg2rad(60))
    q_tel = quat.rotation_xieta(params["el_xi_offset"], params["el_eta_offset"])
    q_cr_roll = quat.euler(2, -1*cr - params["cr_offset"])
    q_rx = quat.rotation_xieta(params["rx_xi_offset"], params["rx_eta_offset"])
    new_az, el, roll = (
        quat.decompose_lonlat(q_enc * q_el_roll * q_tel * q_cr_roll * q_rx) * np.array([-1, 1, 1])[..., None]
    )

    # Stolen from elle
    change = ((new_az - az) + np.pi) % (2 * np.pi) - np.pi
    az = az.copy() + change

    return az, el, roll
