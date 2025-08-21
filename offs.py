# coding: utf-8
import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from so3g.proj import quat
from sotodlib.coords import fp_containers as fpc
from sotodlib.core import AxisManager, Context, metadata
from lat_beams.pointing_model import fit
import sys

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
ctx = Context("/global/cfs/cdirs/sobs/metadata/lat/contexts/smurf_detcal_local.yaml")
if ctx.obsdb is None:
    raise ValueError("No obsdb")
rxs = fpc.Receiver.load_file(
    "/global/cfs/cdirs/sobs/users/skh/data/pointing/lat/finalize_focal_plane/results/per_obs/focal_plane.h5"
)
stream_ids = np.hstack(
    [[fp.stream_id for fp in rxs[obs_id].focal_planes] for obs_id in rxs.keys()]
)
el = np.hstack(
    [
        [ctx.obsdb.get(obs_id)["el_center"] for _ in rxs[obs_id].focal_planes]
        for obs_id in rxs.keys()
    ]
)
az = np.hstack(
    [
        [ctx.obsdb.get(obs_id)["az_center"] for _ in rxs[obs_id].focal_planes]
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
q_roll = quat.rotation_xieta(0, 0, 1 * np.deg2rad(roll))
q_data = quat.rotation_xieta(xi_dist, eta_dist)
xi0, eta0, _ = quat.decompose_xieta(q_roll * q_data * ~q_roll)
d = np.sqrt(xi0**2 + eta0**2)

wmsk = (weights > 400) # * (np.abs(d - np.median(d[weights > 600])) < np.deg2rad(.25))

q_roll = quat.rotation_xieta(0, 0, 1 * np.deg2rad(roll[wmsk]))
q_data = quat.rotation_xieta(xi_dist[wmsk], eta_dist[wmsk])
ctime = ctime[wmsk]
az = az[wmsk]
el = el[wmsk]
roll = roll[wmsk]
corot = roll - el + 60
d = d[wmsk]
stream_ids = stream_ids[wmsk]
xi_cent = xi_cent[wmsk]
eta_cent = eta_cent[wmsk]

# Back it out
xi0, eta0, _ = quat.decompose_xieta(q_data)

# Save
outdir = "/global/homes/s/skh/data/pointing/lat/pointing_model"
pltdir = "/global/homes/s/skh/plots/pointing/lat/pointing_model"
os.makedirs(outdir, exist_ok=True)
os.makedirs(pltdir, exist_ok=True)
if not os.path.isfile(os.path.join(outdir, "db.sqlite")):
    scheme = metadata.ManifestScheme()
    scheme.add_range_match("obs:timestamp")
    scheme.add_data_field("dataset")
    metadata.ManifestDb(scheme=scheme).to_file(os.path.join(outdir, "db.sqlite"))
db = metadata.ManifestDb(os.path.join(outdir, "db.sqlite"))
times = [(0, 1744848000, False, []), (1744848000, 1745150000, True, ['enc_offset_cr', 'el_axis_center_xi0', 'el_axis_center_eta0']), (1745150000, 1749600000, False, []), (1749600000, int(2e9), True, ['enc_offset_cr', 'el_axis_center_xi0', 'el_axis_center_eta0'])] 

params, xi_corr, eta_corr, tmsks = fit(times, ctime, az, el, roll, q_data, d) 
d_corr = np.sqrt(xi_corr**2 + eta_corr**2)

for i, time in enumerate(times):
    aman = AxisManager()
    aman.wrap("version", "lat_v1")
    for key, val in params[i].items():
        aman.wrap(key, val)
    
    if not time[2]:
        entries = db.inspect({'obs:timestamp': (time[0] + time[1])/2.})
        if len(entries) > 0:
            continue
                             
    aman.save(os.path.join(outdir, "pointing_model.h5"), f"t{time[0]}", overwrite=True)
    entry = {"obs:timestamp": (time[0], time[1]), "dataset": f"t{time[0]}"}
    db.add_entry(entry, filename="pointing_model.h5", replace=True)

    if not time[2]:
        continue

    # With time
    tmsk = tmsks[i]
    plt.scatter(ctime[tmsk], d_corr[tmsk]/quat.DEG)
    plt.xlabel("ctime")
    plt.ylabel("Disagreement from Model (deg)")
    plt.title("Disagreement After Correction")
    plt.savefig(os.path.join(pltdir, f"{time[0]}_{time[1]}_offs_t_corr.png"))
    plt.close()
    plt.scatter(ctime[tmsk], d[tmsk]/quat.DEG)
    plt.xlabel("ctime")
    plt.ylabel("Disagreement from Model (deg)")
    plt.title("Disagreement Without Correction")
    plt.savefig(os.path.join(pltdir, f"{time[0]}_{time[1]}_offs_uncorr_t.png"))
    plt.close()

    # By array
    fig, ax = plt.subplots(1, 1, layout='constrained')
    for ufm in np.unique(stream_ids):
        msk = stream_ids[tmsk] == ufm
        ax.scatter(
            xi_corr[tmsk][msk] / quat.DEG, eta_corr[tmsk][msk] / quat.DEG, marker="+", label=ufm
        )  # , c=corot[wmsk])
    plt.title("Corrected Mispointing of the Rx boresight (colored by array)")
    plt.xlabel("xi (deg)")
    plt.ylabel("eta (deg)")
    fig.legend(loc="outside right upper")
    plt.savefig(os.path.join(pltdir, f"{time[0]}_{time[1]}_offs_arr_corr.png"))
    plt.close()
    fig, ax = plt.subplots(1, 1, layout='constrained')
    for ufm in np.unique(stream_ids):
        msk = stream_ids[tmsk] == ufm
        ax.scatter(
            xi0[tmsk][msk] / quat.DEG, eta0[tmsk][msk] / quat.DEG, marker="+", label=ufm
        )  # , c=corot[wmsk])
    plt.title("Uncorrected Mispointing of the Rx boresight (colored by array)")
    plt.xlabel("xi (deg)")
    plt.ylabel("eta (deg)")
    fig.legend(loc="outside right upper")
    plt.savefig(os.path.join(pltdir, f"{time[0]}_{time[1]}_offs_arr_uncorr.png"))
    plt.clf()

    # By roll 
    plt.scatter(xi_corr[tmsk] / quat.DEG, eta_corr[tmsk] / quat.DEG, marker="+", c=el[tmsk])
    plt.title("Corrected Mispointing of the Rx boresight (colored by el)")
    plt.xlabel("xi (deg)")
    plt.ylabel("eta (deg)")
    plt.colorbar(label="el")
    plt.savefig(os.path.join(pltdir, f"{time[0]}_{time[1]}_offs_roll_corr.png"))
    plt.close()
    plt.scatter(xi0[tmsk] / quat.DEG, eta0[tmsk] / quat.DEG, marker="+", c=el[tmsk])
    plt.title("Uncorrected Mispointing of the Rx boresight (colored by el)")
    plt.xlabel("xi (deg)")
    plt.ylabel("eta (deg)")
    plt.colorbar(label="el")
    plt.savefig(os.path.join(pltdir, f"{time[0]}_{time[1]}_offs_roll_uncorr.png"))
    plt.close()

    # Get corrected centers 
    xi_cent_corr = xi_cent[tmsk] + (xi_corr[tmsk] - xi0[tmsk])
    eta_cent_corr = eta_cent[tmsk] + (eta_corr[tmsk] - eta0[tmsk])

    # By array
    fig, ax = plt.subplots(1, 1, layout='constrained')
    for ufm in np.unique(stream_ids):
        msk = stream_ids[tmsk] == ufm
        ax.scatter(
            xi_cent_corr[msk] / quat.DEG, eta_cent_corr[msk] / quat.DEG, marker="+", label=ufm
        )  # , c=corot[wmsk])
    plt.title("Corrected Center of the Array (colored by array)")
    plt.xlabel("xi (deg)")
    plt.ylabel("eta (deg)")
    fig.legend(loc="outside right upper")
    plt.savefig(os.path.join(pltdir, f"{time[0]}_{time[1]}_cent_arr_corr.png"))
    plt.close()
    fig, ax = plt.subplots(1, 1, layout='constrained')
    for ufm in np.unique(stream_ids):
        msk = stream_ids[tmsk] == ufm
        ax.scatter(
            xi_cent[tmsk][msk] / quat.DEG, eta_cent[tmsk][msk] / quat.DEG, marker="+", label=ufm
        )  # , c=corot[wmsk])
    plt.title("Uncorrected Center of the Array (colored by array)")
    plt.xlabel("xi (deg)")
    plt.ylabel("eta (deg)")
    fig.legend(loc="outside right upper")
    plt.savefig(os.path.join(pltdir, f"{time[0]}_{time[1]}_cent_arr_uncorr.png"))
    plt.clf()


    # By roll
    plt.scatter(xi_cent_corr / quat.DEG, eta_cent_corr / quat.DEG, marker="+", label=ufm, c=roll[tmsk])
    plt.title("Corrected Center of the Array (colored by roll)")
    plt.xlabel("xi (deg)")
    plt.ylabel("eta (deg)")
    plt.colorbar(label="roll")
    plt.savefig(os.path.join(pltdir, f"{time[0]}_{time[1]}_cent_roll_corr.png"))
    plt.clf()
    plt.scatter(xi_cent[tmsk] / quat.DEG, eta_cent[tmsk] / quat.DEG, marker="+", label=ufm, c=roll[tmsk])
    plt.title("Uncorrected Center of the Array (colored by roll)")
    plt.xlabel("xi (deg)")
    plt.ylabel("eta (deg)")
    plt.colorbar(label="roll")
    plt.savefig(os.path.join(pltdir, f"{time[0]}_{time[1]}_cent_roll_uncorr.png"))
    plt.clf()


# d = np.sqrt(xi0**2 + eta0**2)
# plt.scatter(ctime[wmsk], d/quat.DEG)
# for t in times[:-1]:
#     plt.axvline(t[1])
# plt.xlim((1750200000, None))
# plt.xlabel("ctime")
# plt.ylabel("Disagreement from Model (deg)")
# plt.savefig("offs_t.png")
# plt.clf()
#
#
#
# fig, ax = plt.subplots(1, 1, layout='constrained')
# for ufm in np.unique(stream_ids):
#     msk = stream_ids[wmsk] == ufm
#     msk *= tmsk
#     ax.scatter(
#         xi0[msk * pmsk] / quat.DEG, eta0[msk * pmsk] / quat.DEG, marker="+", label=ufm
#     )  # , c=corot[wmsk])
# plt.suptitle("Mispointing of the Rx boresight (colored by array)")
# plt.title("Note +eta is equivalent to el encoder reading low.")
# plt.xlabel("xi (deg)")
# plt.ylabel("eta (deg)")
# fig.legend(loc="outside right upper")
# plt.savefig("offs_array.png")
# plt.clf()

# fig, axs = plt.subplots(2, 4, sharey=False, figsize=(45, 12))
# plt.rcParams["figure.dpi"] = 30
# axs[0, 0].scatter(roll[wmsk][pmsk], xi0[pmsk] / quat.DEG, marker="+", s=300)
# axs[0, 1].scatter(el[wmsk][pmsk], xi0[pmsk] / quat.DEG, marker="+", s=300)
# axs[0, 2].scatter(corot[wmsk][pmsk], xi0[pmsk] / quat.DEG, marker="+", s=300)
# axs[0, 3].scatter(tdelt[wmsk][pmsk] / 3600, xi0[pmsk] / quat.DEG, marker="+", s=300)
# axs[1, 0].scatter(roll[wmsk][pmsk], eta0[pmsk] / quat.DEG, marker="+", s=300)
# axs[1, 1].scatter(el[wmsk][pmsk], eta0[pmsk] / quat.DEG, marker="+", s=300)
# axs[1, 2].scatter(corot[wmsk][pmsk], eta0[pmsk] / quat.DEG, marker="+", s=300)
# axs[1, 3].scatter(tdelt[wmsk][pmsk] / 3600, eta0[pmsk] / quat.DEG, marker="+", s=300)
# axs[0, 0].set_ylabel("Xi (deg)")
# axs[1, 0].set_ylabel("Eta (deg)")
# axs[1, 0].set_xlabel("Roll (deg)")
# axs[1, 1].set_xlabel("El (deg)")
# axs[1, 2].set_xlabel("Corot (deg)")
# axs[1, 3].set_xlabel("Time of day (hours)")
# plt.suptitle("Mispointing of the Rx boresight")
# plt.savefig("offs.png")
# plt.clf()
