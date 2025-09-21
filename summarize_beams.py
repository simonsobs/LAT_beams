# coding: utf-8
import datetime as dt
import os

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sotodlib.core import AxisManager


def avg_prof(aman_list, prof="rprof", r="r"):
    all_rs = np.unique(np.hstack([aman[r].value for aman in aman_list]))
    all_profs = np.vstack(
        [
            np.interp(
                all_rs,
                aman[r].value,
                aman[prof].value / aman[prof][0].value,
                left=np.nan,
                right=np.nan,
            )
            for aman in aman_list
        ]
    )
    avg_prof = np.nanmedian(all_profs, axis=0)

    # msk = abs(1 - all_profs[:, 0]/avg_prof[0]) < .2
    # all_profs = all_profs[msk]
    avg_prof = np.nanmedian(all_profs, axis=0)
    err_prof = np.nanstd(all_profs, axis=0)
    n_vals = np.sum(np.isfinite(all_profs), axis=0).astype(float)

    return np.column_stack((all_rs, avg_prof, err_prof, n_vals))  # , msk


nominal_fwhm = {"f090": 2, "f150": 1.3, "f220": 0.95, "f280": 0.83}  # arcmin
fpath = "/global/cfs/cdirs/sobs/users/skh/data/beams/lat/source_maps/per_obs/fits/beam_pars.h5"
plot_dir = "/global/cfs/cdirs/sobs/users/skh/plots_raw/beams/lat/source_maps/per_obs/fits/summary"
# fpath = "/global/cfs/cdirs/sobs/users/skh/data/beams/lat/source_maps/pointing_model/fits/beam_pars.h5"
# plot_dir = "/global/cfs/cdirs/sobs/users/skh/plots_raw/beams/lat/source_maps/pointing_model/fits/summary"
data_dir = os.path.dirname(fpath)
os.makedirs(plot_dir, exist_ok=True)

f = h5py.File(fpath, mode="r")
obs_ids = []
times = []
stream_ids = []
bands = []
for o in f.keys():
    for s in f[o].keys():
        for b in f[o][s].keys():
            obs_ids += [o]
            times += [float(o.split("_")[1])]
            stream_ids += [s]
            bands += [b]

limit_bands = ["f090", "f150", "f220", "f280"]
msk = np.isin(bands, limit_bands)
obs_ids = np.array(obs_ids)[msk]
times = np.array(times)[msk]
stream_ids = np.array(stream_ids)[msk]
bands = np.array(bands)[msk]

dates = [dt.date.fromtimestamp(ct) for ct in times]
tdelt = (
    np.array(
        [
            ct - dt.datetime(year=d.year, month=d.month, day=d.day).timestamp()
            for ct, d in zip(times, dates)
        ]
    )
    / 3600
)

amans = np.array(
    [
        AxisManager.load(f[os.path.join(o, s, b)])
        for o, s, b in zip(obs_ids, stream_ids, bands)
    ]
)
amp = u.Quantity([aman.amp for aman in amans])
amp_err = u.Quantity([aman.amp_err for aman in amans])
noise = u.Quantity([aman.noise for aman in amans])
fwhm_x = u.Quantity([aman.fwhm_ra for aman in amans])
fwhm_y = u.Quantity([aman.fwhm_dec for aman in amans])
fwhm_data = u.Quantity([aman.data_fwhm for aman in amans])
solid_angle = u.Quantity([aman.model_solid_angle_true for aman in amans])
solid_angle_data = u.Quantity([aman.data_solid_angle_corr for aman in amans])
eps = np.abs(fwhm_x - fwhm_y) / (fwhm_x + fwhm_y)
snr = amp / noise

msk = amp / amp_err > 3
msk *= snr > 20
msk *= abs(1 - fwhm_x / fwhm_y) < 0.2
msk *= abs(1 - fwhm_data / fwhm_x) < 0.2
msk *= abs(1 - fwhm_data / fwhm_y) < 0.2
msk *= solid_angle_data > 0

fwhm_ratio = []
ts = []
sids = []
bs = []
for band in np.unique(bands[msk]):
    sang_exp = 2 * np.pi * (np.deg2rad(nominal_fwhm[band] / 2.355 / 60) ** 2)
    bmsk = msk * (bands == band)
    bmsk *= fwhm_data < 2 * nominal_fwhm[band] * 60 * u.arcsec
    bmsk *= fwhm_data < np.percentile(fwhm_data[bmsk], 95)
    bmsk *= solid_angle < 3 * sang_exp * u.sr
    print(f"{band}: {np.mean(fwhm_data[bmsk])} +- {np.std(fwhm_data[bmsk])}")

    profile = avg_prof(amans[bmsk])
    np.savetxt(os.path.join(data_dir, f"profile_{band}.txt"), profile)
    profile = profile[profile[:, 0] < 300]
    plt.plot(
        profile[:, 0], profile[:, 1], label="Average profile", color="b", marker="x"
    )
    plt.fill_between(
        profile[:, 0],
        profile[:, 1] - profile[:, 2],
        profile[:, 1] + profile[:, 2],
        alpha=0.25,
        color="b",
    )
    plt.plot(
        profile[:, 0],
        np.exp(-0.5 * (profile[:, 0] ** 2) / ((nominal_fwhm[band] * 60 / 2.355) ** 2)),
        linestyle="--",
        label="Nominal",
        color="r",
    )
    plt.legend()
    plt.title(f"{band} Profile")
    plt.xlabel('r (")')
    plt.ylabel("Profile")
    plt.savefig(os.path.join(plot_dir, f"profile_{band}.png"))
    plt.close()

    plt.plot(
        profile[:, 0], profile[:, 1], label="Average profile", color="b", marker="x"
    )
    plt.plot(
        profile[:, 0],
        np.exp(-0.5 * (profile[:, 0] ** 2) / ((nominal_fwhm[band] * 60 / 2.355) ** 2)),
        linestyle="--",
        label="Nominal",
        color="r",
    )
    plt.ylim((0.9 * np.min(profile[:, 1]), 1.1 * np.max(profile[:, 1])))
    plt.legend()
    plt.yscale("log")
    plt.title(f"{band} Log Profile")
    plt.xlabel('r (")')
    plt.ylabel("Log Profile")
    plt.savefig(os.path.join(plot_dir, f"profile_{band}_log.png"))
    plt.close()

    plt.hist(fwhm_x[bmsk], alpha=0.5, bins=20, label="FWHM_x")
    plt.hist(fwhm_y[bmsk], alpha=0.5, bins=20, label="FWHM_y")
    plt.hist(fwhm_data[bmsk], alpha=0.5, bins=20, label="FWHM Data")
    plt.legend()
    plt.axvline(nominal_fwhm[band] * 60)
    plt.title(f"FWHM at {band}")
    plt.xlabel('FWHM (")')
    plt.ylabel("Beam Maps")
    plt.savefig(os.path.join(plot_dir, f"fwhm_{band}.png"))
    plt.close()
    fwhm_ratio += [fwhm_data[bmsk] / (nominal_fwhm[band] * 60)]
    ts += [tdelt[bmsk]]
    sids += [stream_ids[bmsk]]
    bs += [band] * np.sum(bmsk)

    plt.hist(eps[bmsk], bins=50)
    plt.xlabel("Ellipticity")
    plt.title(f"Ellipticity at {band}")
    plt.savefig(os.path.join(plot_dir, f"ellipticity_{band}.png"))
    plt.close()

    plt.scatter(tdelt[bmsk], fwhm_x[bmsk], alpha=0.5, label="FWHM_x")
    plt.scatter(tdelt[bmsk], fwhm_y[bmsk], alpha=0.5, label="FWHM_y")
    plt.scatter(tdelt[bmsk], fwhm_data[bmsk], alpha=0.5, label="FWHM_data")
    plt.legend()
    plt.axhline(nominal_fwhm[band] * 60)
    plt.xlabel("Time of day (hour UTC)")
    plt.ylabel('FWHM (")')
    plt.title(f"FWHM by Time at {band}")
    plt.savefig(os.path.join(plot_dir, f"fwhm_time_{band}.png"))
    plt.close()

    plt.scatter(tdelt[bmsk], eps[bmsk])
    plt.xlabel("Time of day (hour UTC)")
    plt.ylabel("Ellipticity")
    plt.title(f"Ellipticity by Time at {band}")
    plt.savefig(os.path.join(plot_dir, f"ellipticity_time_{band}.png"))
    plt.close()

    plt.hist(solid_angle[bmsk], alpha=0.5, bins=20, label="Model")
    plt.hist(solid_angle_data[bmsk], alpha=0.5, bins=20, label="Data")
    plt.legend()
    # plt.xlim((exp / 3, 3 * exp))
    plt.axvline(sang_exp)
    plt.title(f"Solid Angle at {band}")
    plt.xlabel("Solid Angle (sr)")
    plt.ylabel("Beam Maps")
    plt.savefig(os.path.join(plot_dir, f"sang_{band}.png"))
    plt.close()

    plt.scatter(snr[bmsk], fwhm_x[bmsk], alpha=0.5, label="FWHM_x")
    plt.scatter(snr[bmsk], fwhm_y[bmsk], alpha=0.5, label="FWHM_y")
    plt.scatter(snr[bmsk], fwhm_data[bmsk], alpha=0.5, label="FWHM_data")
    plt.legend()
    plt.axhline(nominal_fwhm[band] * 60)
    plt.xlabel("Naive SNR")
    plt.ylabel('FWHM (")')
    plt.title(f"FWHM by SNR at {band}")
    plt.savefig(os.path.join(plot_dir, f"fwhm_snr_{band}.png"))
    plt.close()

plt.scatter(np.hstack(ts), np.hstack(fwhm_ratio), alpha=0.3)
plt.xlabel("Time of Day (UTC)")
plt.ylabel("FWHM/Nominal FWHM")
plt.axvline(21, color="red", label="Approximate Sunset")
plt.legend()
plt.savefig(os.path.join(plot_dir, f"fwhm_ratio_time.png"), bbox_inches="tight")
plt.close()

sids = np.hstack(sids)
fwhm_ratio = np.hstack(fwhm_ratio)
data = {"ufm": sids, "fwhm_ratio": fwhm_ratio, "band": bs}
splt = sns.boxplot(data=data, x="ufm", y="fwhm_ratio", hue="band")
splt.set_xticklabels(splt.get_xticklabels(), rotation=45)
plt.savefig(os.path.join(plot_dir, f"fwhm_ratio_ufm.png"), bbox_inches="tight")
plt.close()

meds = {
    band: {
        stream_id: [np.nan * u.arcsec, np.nan * u.arcsec, np.nan * u.arcsec]
        for stream_id in np.unique(stream_ids[msk])
    }
    for band in np.unique(bands[msk])
}
for stream_id in np.unique(stream_ids[msk]):
    smsk = msk * (stream_ids == stream_id)
    for band in np.unique(bands[smsk]):
        sang_exp = 2 * np.pi * (np.deg2rad(nominal_fwhm[band] / 2.355 / 60) ** 2)
        bmsk = smsk * (bands == band)
        bmsk *= fwhm_data < 2 * nominal_fwhm[band] * 60 * u.arcsec
        bmsk *= solid_angle < 3 * sang_exp * u.sr
        meds[band][stream_id] = [
            np.median(fwhm_x[bmsk]),
            np.median(fwhm_y[bmsk]),
            np.median(fwhm_data[bmsk]),
        ]

        print(f"{stream_id} {band}: {np.sum(bmsk)} good maps")
        print(obs_ids[bmsk])
        plt.hist(fwhm_x[bmsk], alpha=0.5, bins=20, label="FWHM_x")
        plt.hist(fwhm_y[bmsk], alpha=0.5, bins=20, label="FWHM_y")
        plt.hist(fwhm_data[bmsk], alpha=0.5, bins=20, label="FWHM Data")
        plt.xlim((0.5 * nominal_fwhm[band] * 60, 2 * nominal_fwhm[band] * 60))
        plt.legend()
        plt.axvline(nominal_fwhm[band] * 60)
        plt.title(f"{stream_id} FWHM at {band}")
        plt.xlabel('FWHM (")')
        plt.ylabel("Beam Maps")
        plt.savefig(os.path.join(plot_dir, f"fwhm_{stream_id}_{band}.png"))
        plt.close()

        plt.hist(eps[bmsk], bins=50)
        plt.xlabel("Ellipticity")
        plt.title(f"{stream_id} Ellipticity at {band}")
        plt.savefig(os.path.join(plot_dir, f"ellipticity_{stream_id}_{band}.png"))
        plt.close()

        plt.scatter(tdelt[bmsk], fwhm_x[bmsk], alpha=0.5, label="FWHM_x")
        plt.scatter(tdelt[bmsk], fwhm_y[bmsk], alpha=0.5, label="FWHM_y")
        plt.scatter(tdelt[bmsk], fwhm_data[bmsk], alpha=0.5, label="FWHM_data")
        plt.legend()
        plt.axhline(nominal_fwhm[band] * 60)
        plt.xlabel("Time of day (hour UTC)")
        plt.ylabel('FWHM (")')
        plt.title(f"{stream_id} FWHM by Time at {band}")
        plt.savefig(os.path.join(plot_dir, f"fwhm_time_{stream_id}_{band}.png"))
        plt.close()

        plt.scatter(tdelt[bmsk], eps[bmsk])
        plt.xlabel("Time of day (hour UTC)")
        plt.ylabel("Ellipticity")
        plt.title(f"Ellipticity by Time at {band}")
        plt.savefig(os.path.join(plot_dir, f"ellipticity_time_{stream_id}_{band}.png"))
        plt.close()

        plt.hist(solid_angle[bmsk], alpha=0.5, bins=20, label="Model")
        plt.hist(solid_angle_data[bmsk], alpha=0.5, bins=20, label="Data")
        plt.legend()
        plt.axvline(sang_exp)
        plt.title(f"Solid Angle at {band}")
        plt.xlabel("Solid Angle (sr)")
        plt.ylabel("Beam Maps")
        plt.savefig(os.path.join(plot_dir, f"sang_{stream_id}_{band}.png"))
        plt.close()

        plt.scatter(snr[bmsk], fwhm_x[bmsk], alpha=0.5, label="FWHM_x")
        plt.scatter(snr[bmsk], fwhm_y[bmsk], alpha=0.5, label="FWHM_y")
        plt.scatter(snr[bmsk], fwhm_data[bmsk], alpha=0.5, label="FWHM_data")
        plt.legend()
        plt.axhline(nominal_fwhm[band] * 60)
        plt.xlabel("Naive SNR")
        plt.ylabel('FWHM (")')
        plt.title(f"{stream_id} FWHM by SNR at {band}")
        plt.savefig(os.path.join(plot_dir, f"fwhm_snr_{stream_id}_{band}.png"))
        plt.close()

for band in meds.keys():
    ufms = list(meds[band].keys())
    x = u.Quantity([meds[band][ufm][0] for ufm in ufms])
    y = u.Quantity([meds[band][ufm][1] for ufm in ufms])
    d = u.Quantity([meds[band][ufm][2] for ufm in ufms])
    plt.scatter(ufms, x, label="FWHM_x", alpha=0.5)
    plt.scatter(ufms, y, label="FWHM_y", alpha=0.5)
    plt.scatter(ufms, d, label="FWHM_data", alpha=0.5)
    plt.legend()
    plt.axhline(nominal_fwhm[band] * 60)
    plt.xticks(rotation=60)
    plt.title(f"Median FWHM at {band}")
    plt.ylabel('FWHM (")')
    plt.savefig(os.path.join(plot_dir, f"fwhm_ufm_{band}.png"))
    plt.close()
