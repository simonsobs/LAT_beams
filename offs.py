# coding: utf-8
import datetime as dt
import os

import corner
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from so3g.proj import quat
from sotodlib.coords import fp_containers as fpc
from sotodlib.core import AxisManager, Context, metadata

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set3.colors)
ctx = Context("/so/metadata/lat/contexts/smurf_detcal.yaml")
if ctx.obsdb is None:
    raise ValueError("No obsdb")
rxs = fpc.Receiver.load_file(
    "/so/home/saianeesh/data/pointing/lat/finalize_focal_plane/results/per_obs/focal_plane.h5"
)
stream_ids = np.hstack(
    [[fp.stream_id for fp in rxs[obs_id].focal_planes] for obs_id in rxs.keys()]
)
xi_sft = np.hstack(
    [
        [fp.transform.shift[0] for fp in rxs[obs_id].focal_planes]
        for obs_id in rxs.keys()
    ]
)
eta_sft = np.hstack(
    [
        [fp.transform.shift[1] for fp in rxs[obs_id].focal_planes]
        for obs_id in rxs.keys()
    ]
)
rot = np.hstack(
    [[fp.transform.rot for fp in rxs[obs_id].focal_planes] for obs_id in rxs.keys()]
)
el = np.hstack(
    [
        [ctx.obsdb.get(obs_id)["el_center"] for _ in rxs[obs_id].focal_planes]
        for obs_id in rxs.keys()
    ]
)
roll = np.hstack(
    [
        [ctx.obsdb.get(obs_id)["roll_center"] for _ in rxs[obs_id].focal_planes]
        for obs_id in rxs.keys()
    ]
)
weights = np.hstack(
    [
        [np.nansum(fp.weights) for fp in rxs[obs_id].focal_planes]
        for obs_id in rxs.keys()
    ]
)
corot = np.round(roll - el + 60, -1)
obs_ids = np.hstack(
    [[obs_id for _ in rxs[obs_id].focal_planes] for obs_id in rxs.keys()]
)
ctime = (
    np.hstack(
        [
            [
                ctx.obsdb.get(obs_id)["start_time"] + ctx.obsdb.get(obs_id)["stop_time"]
                for _ in rxs[obs_id].focal_planes
            ]
            for obs_id in rxs.keys()
        ]
    )
    / 2
)
dates = [dt.date.fromtimestamp(ct) for ct in ctime]
tdelt = np.array(
    [
        ct - dt.datetime(year=d.year, month=d.month, day=d.day).timestamp()
        for ct, d in zip(ctime, dates)
    ]
)
dist = np.hstack(
    [
        [np.nanmedian(fp.dist) for fp in rxs[obs_id].focal_planes]
        for obs_id in rxs.keys()
    ]
)
xi_cent = np.hstack(
    [
        [fp.center_transformed[0][0] for fp in rxs[obs_id].focal_planes]
        for obs_id in rxs.keys()
    ]
)
eta_cent = np.hstack(
    [
        [fp.center_transformed[0][1] for fp in rxs[obs_id].focal_planes]
        for obs_id in rxs.keys()
    ]
)
xi_dist = np.hstack(
    [
        [
            fp.center_transformed[0][0] - fp.center[0][0]
            for fp in rxs[obs_id].focal_planes
        ]
        for obs_id in rxs.keys()
    ]
)
eta_dist = np.hstack(
    [
        [
            fp.center_transformed[0][1] - fp.center[0][1]
            for fp in rxs[obs_id].focal_planes
        ]
        for obs_id in rxs.keys()
    ]
)
wmsk = (weights > 300) * (dist < np.deg2rad(20 / 3600))

q_roll = quat.rotation_xieta(0, 0, 1 * np.deg2rad(roll))
q_data = quat.rotation_xieta(xi_dist, eta_dist)
xi0, eta0, _ = quat.decompose_xieta(q_roll * q_data * ~q_roll)
pmsk = (np.abs(xi0 - np.median(xi0)) / np.std(xi0) < 5) * (
    np.abs(eta0 - np.median(eta0)) / np.std(eta0) < 5
)  # * (eta0/quat.DEG > -.2)
wmsk *= pmsk
# wmsk *= np.isin( stream_ids, ["ufm_mv24", "ufm_mv21", "ufm_mv14", "ufm_mv13"])
# wmsk *= np.isin( stream_ids, ["ufm_mv13"])
q_roll = quat.rotation_xieta(0, 0, 1 * np.deg2rad(roll[wmsk]))
q_data = quat.rotation_xieta(xi_dist[wmsk], eta_dist[wmsk])  # , np.nan*eta_dist[wmsk])


def fmt(q):
    xi, eta, _ = quat.decompose_xieta(~q_roll * q * q_roll)
    for _x, _e in zip(xi, eta):
        print("%8.4f %8.4f" % (_x, _e))


def go(r, ang):
    ang = ang * np.pi / 180
    a, b = r * np.cos(ang), r * np.sin(ang)
    fmt(quat.rotation_xieta(a, b))


def transform(x):
    a, e, c, xi_el, eta_el, xi_rx, eta_rx, xi_mir, eta_mir = np.deg2rad(x)
    _el = np.deg2rad(el[wmsk])
    _cr = np.deg2rad(el - roll - 60)[wmsk]
    q_enc = quat.rotation_lonlat(-a, e + _el, 0)
    q_enc_nom = quat.rotation_lonlat(0, _el)
    q_tel = quat.rotation_xieta(xi_el, eta_el)
    q_rx = quat.rotation_xieta(xi_rx, eta_rx)
    q_mir = quat.rotation_xieta(xi_mir, eta_mir)
    # rot = q_enc*q_tel*q_z*q_roll*q_data*~q_roll*q_enc_inv
    # rot = q_enc*quat.euler(2, _el + e - np.deg2rad(60))*q_tel*quat.euler(2, -1*(_cr + c))*q_data*~q_roll*q_enc_inv
    rhs = (
        q_enc
        * q_mir
        * quat.euler(2, _el + e - np.deg2rad(60))
        * q_tel
        * quat.euler(2, -1 * (_cr + c))
        * q_rx
    )
    lhs = q_enc_nom * q_roll
    rot = lhs * q_data * ~rhs
    xi0, eta0, _ = quat.decompose_xieta(rot)
    return xi0, eta0


def fit_func(x):
    xi0, eta0 = transform(x)
    print(np.sqrt(np.mean(np.hstack([xi0, eta0]) ** 2)))
    return np.sqrt(np.mean(np.hstack([xi0, eta0]) ** 2))


# Back it out
xi0, eta0, _ = quat.decompose_xieta(q_roll * q_data * ~q_roll)
res = minimize(fit_func, (0, 0, 0, 0, 0, 0, 0, 0, 0))
xi0, eta0 = transform(res.x)
params = {
    "az_offset": res.x[0],
    "el_offset": res.x[1],
    "cr_offset": res.x[2],
    "el_xi_offset": res.x[3],
    "el_eta_offset": res.x[4],
    "rx_xi_offset": res.x[5],
    "rx_eta_offset": res.x[6],
    "mir_xi_offset": res.x[7],
    "mir_eta_offset": res.x[8],
}
print(params)


# Save
outdir = "/so/home/saianeesh/data/pointing/lat/pointing_model"
os.makedirs(outdir, exist_ok=True)
scheme = metadata.ManifestScheme()
scheme.add_range_match("obs:timestamp")
scheme.add_data_field("dataset")
metadata.ManifestDb(scheme=scheme).to_file(os.path.join(outdir, "db.sqlite"))
db = metadata.ManifestDb(os.path.join(outdir, "db.sqlite"))
times = [(0, 1744848000), (1744848000, 2e9)]
for time in times:
    aman = AxisManager()
    aman.wrap("version", "lat_v1")
    for key, val in params.items():
        if time[0] == 0:
            val = 0
        aman.wrap(key, val)
    aman.save(os.path.join(outdir, "pointing_model.h5"), f"t{time[0]}")
    entry = {"obs:timestamp": (time[0], time[1]), "dataset": f"t{time[0]}"}
    db.add_entry(entry, filename="pointing_model.h5", replace=True)

pmsk = np.ones_like(
    xi0, bool
)  # (np.abs(xi0-np.mean(xi0))/np.std(xi0) < 5) * (np.abs(eta0-np.mean(eta0))/np.std(eta0) < 5)

for ufm in np.unique(stream_ids):
    msk = stream_ids[wmsk] == ufm
    plt.scatter(
        xi0[msk * pmsk] / quat.DEG, eta0[msk * pmsk] / quat.DEG, marker="+", label=ufm
    )  # , c=corot[wmsk])
plt.suptitle("Mispointing of the Rx boresight (colored by array)")
plt.title("Note +eta is equivalent to el encoder reading low.")
plt.xlabel("xi (deg)")
plt.ylabel("eta (deg)")
plt.legend()
plt.savefig("/so/home/saianeesh/public_html/offs_array.png")
plt.clf()

fig, axs = plt.subplots(2, 4, sharey=False, figsize=(45, 12))
plt.rcParams["figure.dpi"] = 30
axs[0, 0].scatter(roll[wmsk][pmsk], xi0[pmsk] / quat.DEG, marker="+", s=300)
axs[0, 1].scatter(el[wmsk][pmsk], xi0[pmsk] / quat.DEG, marker="+", s=300)
axs[0, 2].scatter(corot[wmsk][pmsk], xi0[pmsk] / quat.DEG, marker="+", s=300)
axs[0, 3].scatter(tdelt[wmsk][pmsk] / 3600, xi0[pmsk] / quat.DEG, marker="+", s=300)
axs[1, 0].scatter(roll[wmsk][pmsk], eta0[pmsk] / quat.DEG, marker="+", s=300)
axs[1, 1].scatter(el[wmsk][pmsk], eta0[pmsk] / quat.DEG, marker="+", s=300)
axs[1, 2].scatter(corot[wmsk][pmsk], eta0[pmsk] / quat.DEG, marker="+", s=300)
axs[1, 3].scatter(tdelt[wmsk][pmsk] / 3600, eta0[pmsk] / quat.DEG, marker="+", s=300)
axs[0, 0].set_ylabel("Xi (deg)")
axs[1, 0].set_ylabel("Eta (deg)")
axs[1, 0].set_xlabel("Roll (deg)")
axs[1, 1].set_xlabel("El (deg)")
axs[1, 2].set_xlabel("Corot (deg)")
axs[1, 3].set_xlabel("Time of day (hours)")
plt.suptitle("Mispointing of the Rx boresight")
plt.savefig("/so/home/saianeesh/public_html/offs.png")
plt.clf()

dat = np.column_stack([xi0, eta0, el[wmsk], corot[wmsk], tdelt[wmsk] / 3600])
corner.corner(dat, labels=["xi", "eta", "el", "corot", "time"])
plt.savefig("/so/home/saianeesh/public_html/corner.png")
