from so3g.proj import quat
import numpy as np

# lat_params = {"az_offset":-8.17579902e-07, "el_offset":-2.09855826e-01, "roll_offset":2.57302291e-02, "xi_offset":1.61256976e-01, "eta_offset":-2.09820301e-01}
lat_params = {
    "az_offset": 1.49788300e-08,
    "el_offset": -2.08882807e-01,
    "roll_offset": 3.40844271e-02,
    "xi_offset": 1.21677030e-01,
    "eta_offset": -2.08870814e-01,
}
lat_params = {
    "az_offset": 4.03704816e-08,
    "el_offset": -1.39564042e-01,
    "roll_offset": -1.75482402e-01,
    "xi_offset": 1.40683504e-01,
    "eta_offset": -3.15087378e-01,
}


def apply_pointing_model(params, az, el, roll):
    _params = lat_params.copy()
    _params.update(params)
    params = _params
    for key, val in params.items():
        params[key] = -1 * np.deg2rad(val)
    # param keys: az_offset, el_offset, roll_offset, xi_offset, eta_offset
    q_enc = quat.rotation_lonlat(
        -1 * (az.copy() + params["az_offset"]), el.copy() + params["el_offset"]
    )
    q_tel = quat.rotation_xieta(params["xi_offset"], params["eta_offset"])
    q_roll = quat.euler(2, roll.copy() + params["el_offset"] - params["roll_offset"])
    new_az, el, roll = (
        quat.decompose_lonlat(q_enc * q_tel * q_roll) * np.array([-1, 1, 1])[..., None]
    )

    # Stolen from elle
    change = ((new_az - az) + np.pi) % (2 * np.pi) - np.pi
    az = az.copy() + change

    return az, el, roll
