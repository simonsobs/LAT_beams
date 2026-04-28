import os
from glob import glob

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from pixell import enmap
from scipy.interpolate import PchipInterpolator
from sotodlib.core import AxisManager
from healpy.sphtfunc import beam2bl

import lat_beams.fitting.models as bm
from lat_beams.beam_utils import get_fwhm_radial_bins, radial_profile
from lat_beams.fitting.map import fit_bessel_map, fit_gauss_map, make_guess
from lat_beams.plotting import plot_map_complete
from lat_beams.utils import get_args_cfg, setup_cfg, setup_paths


def view_TQU(imap):
    padded = imap
    if len(imap) == 1:
        padded = enmap.zeros((3,) + imap.shape[1:], imap.wcs)
        padded[0][:] = imap[0][:]
    return padded


nominal_fwhm = {"f090": 2.0, "f150": 1.3, "f220": 0.95, "f280": 0.83}  # arcmin

# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg(
    args,
    cfg_dict,
    {
        "fit_source_list": "source_list",
        "map_mask_size": "mask_size",
        "fwhm_tol_map": "fwhm_tol",
    },
)
cfg.aperature *= u.m
pixsize = 3600 * np.rad2deg(cfg.res)

# Setup folders
plot_dir, data_dir = setup_paths(
    cfg.root_dir,
    "beams",
    cfg.tel,
    f"{cfg.pointing_type}{(cfg.append!="")*'_'}{cfg.append}",
)
out_file = os.path.join(data_dir, "stacks", "beam_pars.h5")

# Make highres template
ext_rad = np.deg2rad(cfg.extent_highres / 3600)
pix_extent = 2 * int(cfg.extent_highres // pixsize)
# rowmajor = True here to match sotodlib
twcs = enmap.wcsutils.build(
    [0, 0],
    res=np.rad2deg(cfg.res),
    shape=(pix_extent, pix_extent),
    system="tan",
    rowmajor=True,
)
posmap_highres = enmap.posmap((pix_extent, pix_extent), twcs)

# Get det splits 
det_split_names = [""] + [os.path.splitext(os.path.basename(fname))[0] for fname in glob(os.path.join(cfg.det_split_dir, "*.txt"))]

# Loop through splits
# TODO: Have make_stacked_map save paths in jobdb
for split in cfg.split_by:
    print(f"Splitting by {split}")
    data_dir_spl = os.path.join(data_dir, "stacks", split)
    if not os.path.isdir(data_dir_spl):
        print("\tNo splits found!")
        continue
    if "band" not in split:
        print("Can't fit a non band split!")
        continue
    band_idx = np.where(np.array(split.split("+")) == "band")[0][0]
    for spl_dir in sorted([f.path for f in os.scandir(data_dir_spl) if f.is_dir()]):
        spl_rel = os.path.relpath(spl_dir, data_dir_spl)
        plot_dir_spl = os.path.join(plot_dir, "stacks", split, spl_rel)
        prof_plot_dir = os.path.join(plot_dir, "stack_profiles", split, spl_rel)
        os.makedirs(plot_dir_spl, exist_ok=True)
        os.makedirs(prof_plot_dir, exist_ok=True)
        band = spl_rel.split("+")[band_idx]
        fscale_fac = 90.0 / float(band[1:])
        band_mask_size = np.deg2rad(fscale_fac * cfg.mask_size)
        labels = []
        rprofiles = []
        windows = []
        mprofiles = []
        mwindows = []
        for epoch in cfg.epochs:
            plot_dir_epc = os.path.join(plot_dir_spl, f"{epoch[0]}_{epoch[1]}")
            os.makedirs(plot_dir_epc, exist_ok=True)
            for det_split in det_split_names:
                dstr = f"{'_'*bool(det_split)}{det_split}"
                print(f"\t{spl_rel} {epoch}{dstr.replace('_', ' ')}")
                map_path = os.path.join(
                    spl_dir, f"{spl_rel}_{epoch[0]}_{epoch[1]}{dstr}_stack.fits"
                )
                ivar_path = os.path.join(
                    spl_dir, f"{spl_rel}_{epoch[0]}_{epoch[1]}{dstr}_stack_ivar.fits"
                )
                if not os.path.isfile(map_path):
                    print("\t\tMap not found!")
                    continue
                # TODO: Need to save and load ivar as well
                imap = enmap.read_map(map_path)[0]  # Just T for now
                ivar = enmap.read_map(ivar_path)[0]  # Just T for now
                posmap = imap.posmap()

                # Setup aman for output
                aman = AxisManager()

                # Fit a gaussian to start
                # TODO: Process models to produce solid angle and stuff
                guess = make_guess(
                    amp=1,
                    fwhm_xi=np.deg2rad(cfg.nominal_fwhm[band] / 60.0),
                    fwhm_eta=np.deg2rad(cfg.nominal_fwhm[band] / 60.0),
                    xi0=0,
                    eta0=0,
                    phi=0,
                    off=0,
                )
                gauss_params, model = fit_gauss_map(
                    imap,
                    ivar,
                    posmap,
                    guess,
                    "pW",
                    cfg.sym_gauss,
                    1000,
                )
                if gauss_params is None or model is None:
                    print("\t\tGauss fit failed")
                    continue
                aman.wrap("gauss", gauss_params)
                for to_parent in ["amp", "off", "xi0", "eta0"]:
                    aman.wrap(to_parent, gauss_params[to_parent])

                # Get FWHM from data
                cent = np.unravel_index(
                    np.argmin(posmap[0] ** 2 + posmap[1] ** 2, axis=None), posmap.shape
                )
                rprof = radial_profile(imap, cent[::-1])
                r = np.linspace(0, len(rprof), len(rprof)) * pixsize
                rmsk = r < 3 * 60 * cfg.nominal_fwhm[band] / 2.355
                data_fwhm = (
                    get_fwhm_radial_bins(r[rmsk], rprof[rmsk], interpolate=True) * u.arcsec
                )
                aman.wrap("data_fwhm", data_fwhm)
                aman.wrap("r", r * u.arcsec)
                aman.wrap("rprof", rprof * u.pW)

                # Now fit the bessel beam
                bessel_beam_params, model = fit_bessel_map(
                    imap,
                    ivar,
                    posmap,
                    gauss_params,
                    "pW",
                    cfg.n_bessel,
                    cfg.n_multipoles,
                    cfg.aperature,
                    const.c / (float(band[1:]) * u.GHz),
                    band_mask_size,
                    cfg.bessel_wing_n_sigma
                )
                if bessel_beam_params is None or model is None:
                    print("\t\tBessel fit failed")
                    continue
                aman.wrap("bessel", bessel_beam_params)
                aman.wrap("final_model", "bessel")

                # Make and save a higher resolution profile
                # We want to remove the offset here so the beam goes to 0 and inf
                model_highres = bm.bessel_beam_from_aman(posmap_highres, aman) - aman.bessel.off.value
                cent = np.unravel_index(
                    np.argmin(posmap_highres[0] ** 2 + posmap_highres[1] ** 2, axis=None),
                    posmap_highres.shape,
                )
                mprof = radial_profile(model_highres, cent[::-1])
                mprof[0] = 1.0
                mr = np.linspace(0, len(mprof), len(mprof)) * pixsize
                interp = PchipInterpolator(mr, mprof)
                mr_highres = np.arange(0, cfg.extent_highres, cfg.pixsize_highres)
                mprofile = np.column_stack((mr_highres, interp(mr_highres)))
                rprofile = np.column_stack((r, rprof))
                prof_dir = os.path.join(data_dir, "stack_profiles", split, spl_rel)
                os.makedirs(prof_dir, exist_ok=True)
                np.savetxt(
                    os.path.join(
                        prof_dir, f"model_profile_{spl_rel}_{epoch[0]}_{epoch[1]}{dstr}.txt"
                    ),
                    mprofile,
                )
                np.savetxt(
                    os.path.join(prof_dir, f"profile_{spl_rel}_{epoch[0]}_{epoch[1]}{dstr}.txt"),
                    rprofile,
                )
                # Compute and save b_ell
                bl = beam2bl(rprofile[:, 1], np.deg2rad(rprofile[:, 0] / 3600), cfg.lmax)
                ells = np.arange(cfg.lmax + 1)
                window = np.column_stack((ells, bl))
                np.savetxt(
                    os.path.join(prof_dir, f"window_{spl_rel}_{epoch[0]}_{epoch[1]}{dstr}.txt"),
                    window,
                )
                mbl = beam2bl(mprofile[:, 1], np.deg2rad(mprofile[:, 0] / 3600), cfg.lmax)
                mwindow = np.column_stack((ells, mbl))
                np.savetxt(
                    os.path.join(prof_dir, f"model_window_{spl_rel}_{epoch[0]}_{epoch[1]}{dstr}.txt"),
                    mwindow,
                )

                # Save for plots
                labels += [f"{epoch[0]}_{epoch[1]}{dstr}"]
                rprofiles += [rprofile]
                mprofiles += [mprofile]
                windows += [window]
                mwindows += [mwindow]


                # Save and plot maps
                aman.save(
                    out_file,
                    os.path.join(split, spl_rel, f"{epoch[0]}_{epoch[1]}{dstr}"),
                    overwrite=True,
                )
                posmap = np.rad2deg(posmap) * 3600
                resid = imap.copy()
                resid -= model
                for omap, name in [
                    (model - aman.bessel.off.value, "model"),
                    (resid, "resid"),
                ]:
                    enmap.write_map(
                        os.path.join(
                            spl_dir, f"{spl_rel}_{epoch[0]}_{epoch[1]}{dstr}_stack_{name}.fits"
                        ),
                        omap,
                        "fits",
                        allow_modify=True,
                    )
                    plot_map_complete(
                        omap,
                        posmap,
                        pixsize,
                        cfg.extent,
                        (0, 0),
                        plot_dir_epc,
                        f"{spl_rel} {epoch[0]} {epoch[1]}{dstr.replace('_', ' ')}",
                        comps="T",
                        log_thresh=cfg.log_thresh,
                        append="stack_" + name,
                        units='"',
                        lognorm=1,
                    )

        # Plot profiles and windows
        plt.close()
        for label, rprofile, mprofile in zip(labels, rprofiles, mprofiles):
            label=label.replace("_", " ")
            plt.plot(
                rprofile[:, 0],
                rprofile[:, 1],
                label=label + " Data",
                marker="x",
                alpha=0.4,
            )
            plt.plot(
                mprofile[:, 0],
                mprofile[:, 1],
                label=label + " Model",
                linestyle="--",
                color=plt.gca().lines[-1].get_color(),
                alpha=0.4,
            )
        plt.xlim(0, 3600*np.rad2deg(band_mask_size))
        plt.legend()
        plt.title(f"{spl_rel} Profile")
        plt.xlabel('r (")')
        plt.ylabel("Profile")
        plt.savefig(os.path.join(prof_plot_dir, f"profile_{spl_rel}.png"))

        plt.yscale("log")
        plt.title(f"{spl_rel} Log Profile")
        plt.xlabel('r (")')
        plt.ylabel("Log Profile")
        plt.savefig(os.path.join(prof_plot_dir, f"profile_{spl_rel}_log.png"))
        plt.close()

        for label, window in zip(labels, windows):
            label=label.replace("_", " ")
            plt.loglog(
                window[:, 0],
                window[:, 1],
                label=label,
                alpha=0.5,
            )
        plt.legend()
        plt.title(f"{spl_rel} Window Function")
        plt.xlabel("l")
        plt.ylabel("b_l")
        plt.savefig(os.path.join(prof_plot_dir, f"window_{spl_rel}.png"))
        plt.close()

        for label, window in zip(labels, mwindows):
            label=label.replace("_", " ")
            plt.loglog(
                window[:, 0],
                window[:, 1],
                label=label,
                alpha=0.5,
            )
        plt.legend()
        plt.title(f"{spl_rel} Model Window Function")
        plt.xlabel("l")
        plt.ylabel("b_l")
        plt.savefig(os.path.join(prof_plot_dir, f"model_window_{spl_rel}.png"))
        plt.close()
