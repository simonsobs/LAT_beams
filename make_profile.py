import argparse
import datetime as dt
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from healpy.sphtfunc import beam2bl
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
split_by = cfg.get("split_by", ["band"])
append = cfg["append"] = cfg.get("append", "")
lmax = cfg["lmax"] = cfg.get("lmax", 10000)

# Setup folders
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "beams/lat")
plot_dir = os.path.join(
    root_dir,
    "plots",
    project_dir,
    f"{pointing_type}{(append!="")*'_'}{append}",
    "profiles",
)
data_dir = os.path.join(
    root_dir,
    "data",
    project_dir,
    f"{pointing_type}{(append!="")*'_'}{append}",
)
os.makedirs(data_dir, exist_ok=True)
fpath = os.path.join(data_dir, "beam_pars.h5")

# Load and limit data
all_fits = bu.load_beam_fits(fpath)
limit_bands = list(nominal_fwhm.keys())
all_fits = all_fits[np.isin(all_fits["band"], limit_bands)]

snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
fwhm_x = bu.get_fit_vec(all_fits, "fwhm_xi")
fwhm_y = bu.get_fit_vec(all_fits, "fwhm_eta", "fwhm_xi")
xi0 = bu.get_fit_vec(all_fits, "xi0")
eta0 = bu.get_fit_vec(all_fits, "eta0")
data_fwhm = bu.get_fit_vec(all_fits, "data_fwhm")
solid_angle = bu.get_fit_vec(all_fits, "model_solid_angle_true")
msk = snr > 20
# msk *= abs(1 - fwhm_x / fwhm_y) < 0.2
# msk *= abs(1 - data_fwhm / fwhm_x) < 0.2
# msk *= abs(1 - data_fwhm / fwhm_y) < 0.2
msk *= solid_angle > 0
all_fits = all_fits[msk]


# Plot by splits
for split in split_by:
    print(f"Splitting by {split}")
    split_vec = all_fits[split]
    spl_name = []
    profiles = []
    windows = []
    mprofiles = []
    mwindows = []
    for spl in np.unique(split_vec):
        data_dir_spl = os.path.join(data_dir, "profiles", split, spl)
        plot_dir_spl = os.path.join(plot_dir, split, spl)
        os.makedirs(data_dir_spl, exist_ok=True)
        os.makedirs(plot_dir_spl, exist_ok=True)

        sfits = all_fits[split_vec == spl]
        fwhm_exp = np.array([nominal_fwhm[band] for band in sfits["band"]]) * u.arcmin
        sang_exp = (2 * np.pi * (fwhm_exp.to(u.radian) / 2.355) ** 2).to(u.sr)
        data_fwhm = bu.get_fit_vec(sfits, "data_fwhm")
        solid_angle = bu.get_fit_vec(sfits, "data_solid_angle_corr")
        msk = data_fwhm < 3 * fwhm_exp
        msk *= data_fwhm < np.percentile(data_fwhm[msk], 95)
        msk *= solid_angle < 3 * sang_exp
        sfits = sfits[msk]

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

            # Compute and save profile
            profile = avg_prof(fits["aman"])
            np.savetxt(
                os.path.join(data_dir_spl, f"profile_{spl}_{epoch[0]}_{epoch[1]}.txt"),
                profile,
            )
            mprofile = avg_prof(fits["aman"], prof="mprof")
            np.savetxt(
                os.path.join(data_dir_spl, f"model_profile_{spl}_{epoch[0]}_{epoch[1]}.txt"),
                mprofile,
            )

            # Compute and save b_ell
            bl = beam2bl(profile[:, 1], np.deg2rad(profile[:, 0] / 3600), lmax)
            ells = np.arange(lmax + 1)
            window = np.column_stack((ells, bl))
            np.savetxt(
                os.path.join(data_dir_spl, f"window_{spl}_{epoch[0]}_{epoch[1]}.txt"),
                profile,
            )
            mbl = beam2bl(mprofile[:, 1], np.deg2rad(mprofile[:, 0] / 3600), lmax)
            mwindow = np.column_stack((ells, mbl))
            np.savetxt(
                os.path.join(data_dir_spl, f"model_window_{spl}_{epoch[0]}_{epoch[1]}.txt"),
                profile,
            )

            # Save for plots
            profile = profile[profile[:, 0] < 300]
            profiles += [profile]
            mprofile = mprofile[mprofile[:, 0] < 300]
            mprofiles += [mprofile]
            windows += [window]
            mwindows += [mwindow]

        for epoch, profile in zip(epochs, profiles):
            plt.plot(
                profile[:, 0],
                profile[:, 1],
                label=epoch,
                marker="x",
            )
        plt.legend()
        plt.title(f"{spl} Profile")
        plt.xlabel('r (")')
        plt.ylabel("Profile")
        plt.savefig(os.path.join(plot_dir_spl, f"profile_{spl}.png"))

        plt.yscale("log")
        plt.title(f"{spl} Log Profile")
        plt.xlabel('r (")')
        plt.ylabel("Log Profile")
        plt.savefig(os.path.join(plot_dir_spl, f"profile_{spl}_log.png"))
        plt.close()

        for epoch, profile in zip(epochs, mprofiles):
            plt.plot(
                profile[:, 0],
                profile[:, 1],
                label=epoch,
                marker="x",
            )
        plt.legend()
        plt.title(f"{spl} Model Profile")
        plt.xlabel('r (")')
        plt.ylabel("Profile")
        plt.savefig(os.path.join(plot_dir_spl, f"model_profile_{spl}.png"))

        plt.yscale("log")
        plt.title(f"{spl} Log Model Profile")
        plt.xlabel('r (")')
        plt.ylabel("Log Profile")
        plt.savefig(os.path.join(plot_dir_spl, f"model_profile_{spl}_log.png"))
        plt.close()

        for epoch, window in zip(epochs, windows):
            plt.loglog(
                window[:, 0],
                window[:, 1],
                label=epoch,
                marker="x",
            )
        plt.legend()
        plt.title(f"{spl} Window Function")
        plt.xlabel("l")
        plt.ylabel("b_l")
        plt.savefig(os.path.join(plot_dir_spl, f"window_{spl}.png"))
        plt.close()

        for epoch, window in zip(epochs, mwindows):
            plt.loglog(
                window[:, 0],
                window[:, 1],
                label=epoch,
                marker="x",
            )
        plt.legend()
        plt.title(f"{spl} Model Window Function")
        plt.xlabel("l")
        plt.ylabel("b_l")
        plt.savefig(os.path.join(plot_dir_spl, f"model_window_{spl}.png"))
        plt.close()
