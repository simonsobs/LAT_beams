import argparse
import datetime as dt
import os

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sotodlib.core import AxisManager

from lat_beams import beam_utils as bu


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


nominal_fwhm = {"f090": 2.0, "f150": 1.3, "f220": 0.95, "f280": 0.83}  # arcmin

parser = argparse.ArgumentParser()
parser.add_argument("cfg", help="Path to the config file")
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

# Get some global setting
epochs = cfg.get("epochs", [(0, 2e10)])
pointing_type = cfg.get("pointing_type", "pointing_model")
nominal_fwhm = cfg.get("nominal_fwhm", nominal_fwhm)
split_by = cfg.get("split_by", ["band", "stream_id"])

# Setup folders
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "beams/lat")
plot_dir = os.path.join(
    root_dir, "plots", project_dir, "source_maps", pointing_type, "fits", "summary"
)
data_dir = os.path.join(
    root_dir,
    "data",
    project_dir,
    "source_maps",
    pointing_type,
    "fits",
)
fpath = os.path.join(data_dir, "beam_pars.h5")
os.makedirs(plot_dir, exist_ok=True)

# Load and limit data
all_fits = bu.load_beam_fits(fpath)
limit_bands = list(nominal_fwhm.keys())
all_fits = all_fits[np.isin(all_fits["band"], limit_bands)]

snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
fwhm_x = bu.get_fit_vec(all_fits, "fwhm_xi")
fwhm_y = bu.get_fit_vec(all_fits, "fwhm_eta")
xi0 = bu.get_fit_vec(all_fits, "xi0")
eta0 = bu.get_fit_vec(all_fits, "eta0")
r = np.sqrt(xi0**2 + eta0**2)
data_fwhm = bu.get_fit_vec(all_fits, "data_fwhm")
solid_angle = bu.get_fit_vec(all_fits, "model_solid_angle_true")
msk = snr > 20
msk *= abs(1 - fwhm_x / fwhm_y) < 0.2
msk *= abs(1 - data_fwhm / fwhm_x) < 0.2
msk *= abs(1 - data_fwhm / fwhm_y) < 0.2
msk *= solid_angle > 0
msk *= r < 180 * u.arcsec
all_fits = all_fits[msk]

# Plot by splits
for split in split_by:
    print(f"Splitting by {split}")
    split_vec = all_fits[split]

    time = []
    fwhm_ratio = []
    ellipticity = []
    amp_ratio = []
    amp_names = []
    spl_name = []
    for spl in np.unique(split_vec):
        plot_dir_spl = os.path.join(plot_dir, split, spl)
        os.makedirs(plot_dir_spl, exist_ok=True)
        sfits = all_fits[split_vec == spl]
        fwhm_exp = np.array([nominal_fwhm[band] for band in sfits["band"]]) * u.arcmin
        sang_exp = (2 * np.pi * (fwhm_exp.to(u.radian) / 2.355) ** 2).to(u.sr)
        data_fwhm = bu.get_fit_vec(sfits, "data_fwhm")
        solid_angle = bu.get_fit_vec(sfits, "data_solid_angle_corr")
        msk = data_fwhm < 2 * fwhm_exp
        msk *= data_fwhm < np.percentile(data_fwhm[msk], 95)
        msk *= solid_angle < 3 * sang_exp
        sfits = sfits[msk]
        fwhm_exp = fwhm_exp[msk]
        sang_exp = sang_exp[msk]

        for epoch in epochs:
            print(f"\tEpoch {epoch}")
            times = sfits["time"]
            tmsk = (times >= epoch[0]) * (times < epoch[1])
            if np.sum(tmsk) == 0:
                print(f"\t\tNo maps found! Skipping...")
                continue
            fits = sfits[tmsk]
            data_fwhm = bu.get_fit_vec(fits, "data_fwhm")
            print(f"\t\t{spl}: {np.mean(data_fwhm)} +- {np.std(data_fwhm)}")

            # Profile
            profile = avg_prof(fits["aman"])
            np.savetxt(
                os.path.join(data_dir, f"profile_{spl}_{epoch[0]}_{epoch[1]}.txt"),
                profile,
            )
            profile = profile[profile[:, 0] < 300]
            plt.plot(
                profile[:, 0],
                profile[:, 1],
                label="Average profile",
                color="b",
                marker="x",
            )
            plt.fill_between(
                profile[:, 0],
                profile[:, 1] - profile[:, 2],
                profile[:, 1] + profile[:, 2],
                alpha=0.25,
                color="b",
            )
            # plt.plot(
            #     profile[:, 0],
            #     np.exp(-0.5 * (profile[:, 0] ** 2) / ((nominal_fwhm[band] * 60 / 2.355) ** 2)),
            #     linestyle="--",
            #     label="Nominal",
            #     color="r",
            # )
            plt.legend()
            plt.title(f"{spl} Profile")
            plt.xlabel('r (")')
            plt.ylabel("Profile")
            plt.savefig(
                os.path.join(plot_dir_spl, f"profile_{spl}_{epoch[0]}_{epoch[1]}.png")
            )
            plt.close()

            plt.plot(
                profile[:, 0],
                profile[:, 1],
                label="Average profile",
                color="b",
                marker="x",
            )
            # plt.plot(
            #     profile[:, 0],
            #     np.exp(-0.5 * (profile[:, 0] ** 2) / ((nominal_fwhm[band] * 60 / 2.355) ** 2)),
            #     linestyle="--",
            #     label="Nominal",
            #     color="r",
            # )
            plt.ylim((0.9 * np.min(profile[:, 1]), 1.1 * np.max(profile[:, 1])))
            plt.legend()
            plt.yscale("log")
            plt.title(f"{spl} Log Profile")
            plt.xlabel('r (")')
            plt.ylabel("Log Profile")
            plt.savefig(
                os.path.join(
                    plot_dir_spl, f"profile_{spl}_{epoch[0]}_{epoch[1]}_log.png"
                )
            )
            plt.close()

            # All params by time of day
            for par in fits["aman"][0]._fields.keys():
                if np.ndim(fits["aman"][0][par]) > 0:
                    continue
                dat = bu.get_fit_vec(fits, par)
                plt.scatter(fits["hour"], dat, alpha=0.4)
                plt.title(f"{par} by time of day")
                plt.xlabel("Hour (UTC)")
                plt.ylabel(f"{par} ({dat.unit.name})")
                plt.savefig(
                    os.path.join(
                        plot_dir_spl, f"{par}_{spl}_{epoch[0]}_{epoch[1]}_by_time.png"
                    )
                )
                plt.close()

            # Histograms of all params
            amp_pars = []
            for par in fits["aman"][0]._fields.keys():
                if np.ndim(fits["aman"][0][par]) > 0:
                    continue
                if "amp" in par and par != "amp":
                    amp_pars += [par]
                dat = bu.get_fit_vec(fits, par)
                plt.hist(dat, bins=20)
                plt.title(f"{par} Distribution")
                plt.xlabel("Beam Maps (#)")
                plt.ylabel(f"{par} ({dat.unit.name})")
                plt.savefig(
                    os.path.join(plot_dir_spl, f"{par}_{spl}_{epoch[0]}_{epoch[1]}.png")
                )
                plt.close()

            # Now more specific plots
            for par in ("fwhm_xi", "fwhm_eta", "data_fwhm"):
                dat = bu.get_fit_vec(fits, par)
                plt.hist(dat, bins="auto", label=par, alpha=0.4)
            if split == "band":
                plt.axvline(nominal_fwhm[spl] * 60)
            plt.title(f"FWHM for {spl}")
            plt.xlabel("Beam Maps (#)")
            plt.ylabel(f'FWHM (")')
            plt.legend()
            plt.savefig(
                os.path.join(plot_dir_spl, f"fwhm_{spl}_{epoch[0]}_{epoch[1]}.png")
            )
            plt.close()

            fwhm_x = bu.get_fit_vec(fits, "fwhm_xi")
            fwhm_y = bu.get_fit_vec(fits, "fwhm_eta")
            eps = np.abs(fwhm_x - fwhm_y) / (fwhm_x + fwhm_y)
            plt.hist(eps, bins="auto")
            plt.title(f"Ellipticity for {spl}")
            plt.xlabel("Beam Maps (#)")
            plt.ylabel(f'FWHM (")')
            plt.savefig(
                os.path.join(
                    plot_dir_spl, f"ellipticity_{spl}_{epoch[0]}_{epoch[1]}.png"
                )
            )
            plt.close()
            plt.plot(fits["hour"], eps)
            plt.title(f"Ellipticity by time of day")
            plt.xlabel("Hour (UTC)")
            plt.ylabel(f"Ellipticity")
            plt.savefig(
                os.path.join(
                    plot_dir_spl, f"ellipticity_{spl}_{epoch[0]}_{epoch[1]}_by_time.png"
                )
            )
            plt.close()

            solid_angle = bu.get_fit_vec(fits, "model_solid_angle_true")
            solid_angle_data = bu.get_fit_vec(fits, "data_solid_angle_corr")
            plt.hist(solid_angle, alpha=0.5, bins=20, label="Model")
            plt.hist(solid_angle_data, alpha=0.5, bins=20, label="Data")
            plt.legend()
            if split == "band":
                plt.axvline(sang_exp[0].value)
            plt.title(f"Solid Angle for {spl}")
            plt.xlabel("Solid Angle (sr)")
            plt.ylabel("Beam Maps")
            plt.savefig(
                os.path.join(plot_dir_spl, f"sang_{spl}_{epoch[0]}_{epoch[1]}.png")
            )
            plt.close()

            # Save some stuff for a bigger plot
            norm = np.array([aman.rprof[0].value for aman in fits["aman"]])
            time += [fits["time"]]
            fwhm_ratio += [data_fwhm / fwhm_exp[tmsk]]
            ellipticity += [eps]
            amp_ratio += [
                {name: bu.get_fit_vec(fits, name) / norm for name in amp_pars}
            ]
            amp_names += amp_pars
            spl_name += [[spl] * len(fits)]

    # Now some cross epoch plots
    plot_dir_spl = os.path.join(plot_dir, split, "by_time")
    os.makedirs(plot_dir_spl, exist_ok=True)
    if len(time) == 0:
        continue
    time = np.hstack(time)
    fwhm_ratio = np.hstack(fwhm_ratio)
    ellipticity = np.hstack(ellipticity)
    amp_names = np.unique(amp_names)
    amp_ratio = {
        name: np.hstack([ar.get(name, np.array([])) for ar in amp_ratio])
        for name in amp_names
    }
    epoch_lines = np.unique(epochs)
    epoch_lines = epoch_lines[
        (epoch_lines >= np.min(time)) * (epoch_lines < np.max(time))
    ]
    spl_name = np.hstack(spl_name)

    plt.scatter(time, spl_name, c=fwhm_ratio, alpha=0.4, marker="x")
    plt.colorbar(label="Data FWHM/Nominal FWHM", ax=plt.gca())
    for e in epoch_lines:
        plt.axvline(e)
    plt.title(f"FWHM Ratio by {split}")
    plt.xlabel("ctime (s)")
    plt.ylabel(split)
    plt.savefig(os.path.join(plot_dir_spl, f"fwhm_ratio_{split}.png"))
    plt.close()

    plt.scatter(time, spl_name, c=ellipticity, alpha=0.4, marker="x")
    plt.colorbar(label="Ellipticity", ax=plt.gca())
    for e in epoch_lines:
        plt.axvline(e)
    plt.title(f"Ellipticity by {split}")
    plt.xlabel("ctime (s)")
    plt.ylabel(split)
    plt.savefig(os.path.join(plot_dir_spl, f"ellipticity_{split}.png"))
    plt.close()

    for name, amp in amp_ratio.items():
        plt.scatter(
            time,
            spl_name,
            c=amp.value,
            alpha=0.4,
            marker="x",
            vmin=np.percentile(amp.value, 5),
            vmax=np.percentile(amp.value, 95),
        )
        plt.colorbar(label="Normalized Amplitude", ax=plt.gca())
        for e in epoch_lines:
            plt.axvline(e)
        plt.title(f"Normalized {name} by {split}")
        plt.xlabel("ctime (s)")
        plt.ylabel(split)
        plt.savefig(os.path.join(plot_dir_spl, f"{name}_{split}.png"))
        plt.close()
