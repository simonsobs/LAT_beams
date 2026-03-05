import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from healpy.sphtfunc import beam2bl
from sotodlib.core import Context

from lat_beams import beam_utils as bu
from lat_beams.fitting import profile as bf
from lat_beams.utils import get_args_cfg, make_jobdb, setup_cfg, setup_paths


def avg_prof(aman_list, prof="rprof", r="r"):
    all_rs = np.unique(np.hstack([aman[r].value for aman in aman_list]))
    with np.errstate(divide="ignore", invalid="ignore"):
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


# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg(args, cfg_dict, {"map_mask_size", "mask_size"})
ctx = Context(cfg.ctx_path)
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
cfg.extent *= u.arcsec
cfg.r_step *= u.arcsec
cfg.corr_primary *= u.mm
cfg.eps_primary *= u.um
cfg.mask_size *= u.degree

# Setup folders
plot_dir, data_dir = setup_paths(
    cfg.root_dir,
    "beams",
    cfg.tel,
    f"{cfg.pointing_type}{(cfg.append!="")*'_'}{cfg.append}",
)
plot_dir = os.path.join(plot_dir, "profiles")
os.makedirs(plot_dir, exist_ok=True)
fpath = os.path.join(data_dir, "beam_pars.h5")
jdb = make_jobdb(None, data_dir)

# Get jobs
mjobdict = {
    f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['band']}": job
    for job in jdb.get_jobs(jclass="beam_map", jstate="done")
}
fjobdict = {
    f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['band']}": job
    for job in jdb.get_jobs(jclass="fit_map", jstate="done")
}
alljobstr = list(set(list(mjobdict.keys())) & set(list(fjobdict.keys())))
mjobs = np.array([mjobdict[jobstr] for jobstr in alljobstr])
fjobs = np.array([fjobdict[jobstr] for jobstr in alljobstr])

print(f"{len(alljobstr)} maps to use")
if len(alljobstr) == 0:
    sys.exit(0)

# Load fits
all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs)
limit_bands = list(cfg.nominal_fwhm.keys())
all_fits = all_fits[np.isin(all_fits["band"], limit_bands)]

snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
data_fwhm = bu.get_fit_vec(all_fits, "data_fwhm")
solid_angle = bu.get_fit_vec(all_fits, "data_solid_angle_corr")
msk = snr > 20
msk *= solid_angle > 0
all_fits = all_fits[msk]

r_calc = np.arange(0, 2 * cfg.extent.value, cfg.r_step.value) * u.arcsec

# Plot by splits
for split in cfg.split_by:
    print(f"Splitting by {split}")
    split_vec = bu.get_split_vec(all_fits, split, ctx)
    for spl in np.unique(split_vec):
        profiles = []
        windows = []
        mprofiles = []
        mwindows = []
        motos = []
        data_dir_spl = os.path.join(data_dir, "profiles", split, spl)
        plot_dir_spl = os.path.join(plot_dir, split, spl)
        os.makedirs(data_dir_spl, exist_ok=True)
        os.makedirs(plot_dir_spl, exist_ok=True)

        sfits = all_fits[split_vec == spl]
        fwhm_exp = (
            np.array([cfg.nominal_fwhm[band] for band in sfits["band"]]) * u.arcmin
        )
        sang_exp = (2 * np.pi * (fwhm_exp.to(u.radian) / 2.355) ** 2).to(u.sr)
        data_fwhm = bu.get_fit_vec(sfits, "data_fwhm")
        solid_angle = bu.get_fit_vec(sfits, "data_solid_angle_corr")
        msk = data_fwhm < 3 * fwhm_exp
        msk *= data_fwhm < np.percentile(data_fwhm[msk], 95)
        msk *= solid_angle < 2 * sang_exp
        msk *= solid_angle > 0.5 * sang_exp
        sfits = sfits[msk]

        for epoch in cfg.epochs:
            print(f"\tEpoch {epoch}")
            times = sfits["time"]
            tmsk = (times >= epoch[0]) * (times < epoch[1])
            if np.sum(tmsk) == 0:
                print(f"\t\tNo maps found! Skipping...")
                continue
            fits = sfits[tmsk]
            data_fwhm = bu.get_fit_vec(fits, "data_fwhm")
            solid_angle = bu.get_fit_vec(fits, "data_solid_angle_corr")
            print(f"\t\t{spl} Data FWHM: {np.mean(data_fwhm)} +- {np.std(data_fwhm)}")
            print(
                f"\t\t{spl} Solid Angle: {np.mean(solid_angle)} +- {np.std(solid_angle)}"
            )

            band = float(fits["band"][0][1:]) * u.GHz
            fscale_fac = 90.0 * u.GHz / band
            mask_rad = (mask_size * fscale_fac).to(u.arcsec).value

            # Compute and save profile
            profile = avg_prof(fits["aman"])
            profile = profile[profile[:, 0] < mask_rad]
            np.savetxt(
                os.path.join(data_dir_spl, f"profile_{spl}_{epoch[0]}_{epoch[1]}.txt"),
                profile,
            )
            profile = np.genfromtxt(
                os.path.join(data_dir_spl, f"profile_{spl}_{epoch[0]}_{epoch[1]}.txt"),
            )

            # Fit model
            mprofile, mpars, moto = bf.fit_bessel_profile(
                profile[:, 0] * u.arcsec,
                profile[:, 1] * u.dimensionless_unscaled,
                np.median(data_fwhm),
                6 * u.m,
                const.c / band,
                np.median(solid_angle),
                cfg.corr_primary,
                cfg.eps_primary,
                r_calc,
            )
            if mprofile is None or mpars is None:
                raise ValueError("Fit failed!")
            print(mpars)
            np.savetxt(
                os.path.join(
                    data_dir_spl, f"model_profile_{spl}_{epoch[0]}_{epoch[1]}.txt"
                ),
                mprofile,
            )
            mprofile = np.genfromtxt(
                os.path.join(
                    data_dir_spl, f"model_profile_{spl}_{epoch[0]}_{epoch[1]}.txt"
                ),
            )

            # Compute and save b_ell
            bl = beam2bl(profile[:, 1], np.deg2rad(profile[:, 0] / 3600), cfg.lmax)
            ells = np.arange(cfg.lmax + 1)
            window = np.column_stack((ells, bl))
            np.savetxt(
                os.path.join(data_dir_spl, f"window_{spl}_{epoch[0]}_{epoch[1]}.txt"),
                window,
            )
            mbl = beam2bl(mprofile[:, 1], np.deg2rad(mprofile[:, 0] / 3600), cfg.lmax)
            mwindow = np.column_stack((ells, mbl))
            np.savetxt(
                os.path.join(
                    data_dir_spl, f"model_window_{spl}_{epoch[0]}_{epoch[1]}.txt"
                ),
                mwindow,
            )

            # Save for plots
            profiles += [profile]
            mprofiles += [mprofile]
            motos += [moto]
            windows += [window]
            mwindows += [mwindow]

        for epoch, profile, mprofile in zip(cfg.epochs, profiles, motos):
            plt.plot(
                profile[:, 0],
                profile[:, 1],
                label=str(epoch) + "Data",
                marker="x",
                alpha=0.5,
            )
            plt.plot(
                mprofile[:, 0],
                mprofile[:, 1],
                label=str(epoch) + "Model",
                marker="+",
                linestyle="--",
                color=plt.gca().lines[-1].get_color(),
                alpha=0.5,
            )
        plt.ylim((1e-5, 1))
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

        for epoch, profile in zip(cfg.epochs, mprofiles):
            plt.plot(
                profile[:, 0],
                profile[:, 1],
                label=str(epoch),
                marker="x",
                alpha=0.5,
            )
        plt.ylim((1e-5, 1))
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

        for epoch, profile, mprofile in zip(cfg.epochs, profiles, motos):
            plt.plot(
                profile[:, 0],
                np.abs(profile[:, 1] - mprofile[:, 1]) / mprofile[:, 1],
                label=str(epoch),
                marker="x",
                alpha=0.5,
            )
        plt.ylim((1e-5, 1))
        plt.legend()
        plt.title(f"{spl} Fractional Residual Profile")
        plt.xlabel('r (")')
        plt.ylabel("Profile")
        plt.savefig(os.path.join(plot_dir_spl, f"resid_profile_{spl}.png"))

        plt.yscale("log")
        plt.title(f"{spl} Log Fractional Residual Profile")
        plt.xlabel('r (")')
        plt.ylabel("Log Profile")
        plt.savefig(os.path.join(plot_dir_spl, f"resid_profile_{spl}_log.png"))
        plt.close()

        for epoch, window in zip(cfg.epochs, windows):
            plt.loglog(
                window[:, 0],
                window[:, 1],
                label=str(epoch),
                marker="x",
                alpha=0.5,
            )
        plt.legend()
        plt.title(f"{spl} Window Function")
        plt.xlabel("l")
        plt.ylabel("b_l")
        plt.savefig(os.path.join(plot_dir_spl, f"window_{spl}.png"))
        plt.close()

        for epoch, window in zip(cfg.epochs, mwindows):
            plt.loglog(
                window[:, 0],
                window[:, 1],
                label=str(epoch),
                marker="x",
                alpha=0.5,
            )
        plt.legend()
        plt.title(f"{spl} Model Window Function")
        plt.xlabel("l")
        plt.ylabel("b_l")
        plt.savefig(os.path.join(plot_dir_spl, f"model_window_{spl}.png"))
        plt.close()
