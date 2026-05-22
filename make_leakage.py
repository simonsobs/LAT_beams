import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from pixell import enmap, curvedsky
from healpy.sphtfunc import beam2bl
from lat_beams.beam_utils import get_fwhm_radial_bins, radial_profile

from lat_beams.plotting import plot_map
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

# Loop through splits
for split in cfg.split_by:
    print(f"Splitting by {split}")
    data_dir_spl = os.path.join(data_dir, "ml_maps", split)
    if not os.path.isdir(data_dir_spl):
        print("\tNo splits found!")
        continue
    band_idx = np.where(np.array(split.split("+")) == "band")[0][0]
    for spl_dir in sorted([f.path for f in os.scandir(data_dir_spl) if f.is_dir()]):
        spl_rel = os.path.relpath(spl_dir, data_dir_spl)
        plot_dir_spl = os.path.join(plot_dir, "ml_maps", split, spl_rel)
        prof_plot_dir = os.path.join(plot_dir, "leakage_profiles", split, spl_rel)
        os.makedirs(plot_dir_spl, exist_ok=True)
        os.makedirs(prof_plot_dir, exist_ok=True)
        labels = []
        profiles = []
        for epoch in cfg.epochs:
            plot_dir_epc = os.path.join(plot_dir_spl, f"{epoch[0]}_{epoch[1]}")
            os.makedirs(plot_dir_epc, exist_ok=True)
            print(f"\t{spl_rel} {epoch}")
            map_path = os.path.join(
                spl_dir,
                f"{epoch[0]}_{epoch[1]}",
                f"{spl_rel}_{epoch[0]}_{epoch[1]}_pass3_sky_map.fits",
            )
            if not os.path.isfile(map_path):
                print("\t\tMap not found!")
                continue
            imap = enmap.read_map(map_path)  # Just T for now
            posmap = imap.posmap()

            # Convert to TQU
            alm = curvedsky.map2alm(imap, lmax=cfg.lmax, spin=[0, 2])
            imap = curvedsky.alm2map(alm, imap, spin=[0, 0])

            # Get center pixel
            cent = np.unravel_index(
                np.argmin(posmap[0] ** 2 + posmap[1] ** 2, axis=None),
                posmap.shape,
            )

            # Make T profile and bl
            tprof = radial_profile(imap[0], cent[::-1])
            r = np.linspace(0, len(tprof), len(tprof)) * pixsize
            tbl = beam2bl(tprof, np.deg2rad(r / 3600), cfg.lmax)
            ells = np.arange(cfg.lmax + 1)

            # Make E profile and bl
            eprof = radial_profile(imap[1], cent[::-1])
            r = np.linspace(0, len(tprof), len(tprof)) * pixsize
            ebl = beam2bl(eprof, np.deg2rad(r / 3600), cfg.lmax)

            # Make and save leakage
            prof_dir = os.path.join(data_dir, "leakage_profiles", split, spl_rel)
            os.makedirs(prof_dir, exist_ok=True)
            lbl = ebl / tbl
            leakage = np.column_stack((ells, lbl))
            np.savetxt(
                os.path.join(
                    prof_dir,
                    f"leakage_{spl_rel}_{epoch[0]}_{epoch[1]}.txt",
                ),
                leakage,
            )

            # Save for plots
            labels += [f"{epoch[0]}_{epoch[1]}"]
            profiles += [leakage]

            # Plot maps
            posmap = np.rad2deg(posmap) * 3600
            for im, comp in zip(imap[1:], ("E", "B")):
                for log in (False, True):
                    plot_map(
                        im,
                        posmap,
                        pixsize,
                        cfg.extent,
                        (0, 0),
                        plot_dir_epc,
                        f"{spl_rel} {epoch[0]} {epoch[1]}",
                        comp=comp,
                        log=log,
                        log_thresh=cfg.log_thresh,
                        append="ML",
                        units='"',
                    )
        # Plot profiles and windows
        plt.close()
        for label, profile in zip(labels, profiles):
            label = label.replace("_", " ")
            plt.plot(
                profile[:, 0],
                profile[:, 1],
                label=label,
                alpha=0.6,
            )
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f"{spl_rel} Leakage")
        plt.xlabel("ell")
        plt.ylim((-0.1, 0.1))
        plt.ylabel("Leakage")
        plt.savefig(
            os.path.join(prof_plot_dir, f"leakage_{spl_rel}.png"), bbox_inches="tight"
        )
