import os

import astropy.units as u
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from pixell import enmap, curvedsky
from healpy.sphtfunc import beam2bl
from lat_beams.beam_utils import radial_profile

from lat_beams.plotting import plot_map, auto_relplot
from lat_beams.utils import get_args_cfg, setup_cfg, setup_paths


def harmbin(imap, iau=False, brel=1, spin=[0, 2]):
    '''
    Return b_ell by binning in I, E, B harmonic space.

    Parameters
    ----------
    imap : (3, Ny, Nx) enmap
        Input enmap with I. Q, U elements.
    iau : bool, optional
        Use the IAU polarization convention.
    brel : float, optional
        Coarsen the harmonic spacing by this amount
    spin : int, array-like
        Spin of the map2harm operation

    Returns
    -------
    ells : (nell) array
        Multipoles
    b_ell : (3, nell) array
        Main beam, TE beam and TB beam.
    '''

    imap_shift = enmap.ifftshift(imap.copy())
    fmap = enmap.map2harm(imap_shift, iau=iau, spin=spin, normalize="phy").real
    b_ell, ells = fmap.lbin(brel=brel)

    return ells, b_ell

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
fine=True #False

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
    prof_plot_dir = os.path.join(plot_dir, "leakage_profiles", split)
    os.makedirs(prof_plot_dir, exist_ok=True)
    to_plot = {s : [] for s in split.split("+")}
    to_plot["ell"] = []
    to_plot["te_leakage"] = []
    to_plot["tb_leakage"] = []
    to_plot["epoch"] = []
    for spl_dir in sorted([f.path for f in os.scandir(data_dir_spl) if f.is_dir()]):
        spl_rel = os.path.relpath(spl_dir, data_dir_spl)
        plot_dir_spl = os.path.join(plot_dir, "ml_maps", split, spl_rel)
        os.makedirs(plot_dir_spl, exist_ok=True)
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
            imap = enmap.read_map(map_path)
            posmap = imap.posmap()

            if fine: 
                # Convert to TQU
                alm = curvedsky.map2alm(imap, lmax=cfg.lmax, spin=[0, 2])
                imap = curvedsky.alm2map(alm, imap, spin=[0, 0])

                # Get center pixel
                cent = np.unravel_index(
                    np.argmin(posmap[0] ** 2 + posmap[1] ** 2, axis=None),
                    posmap.shape,
                )

                # Make profiles and bl
                b_ell = []
                ells = np.arange(cfg.lmax + 1)
                for i in range(3):
                    prof = radial_profile(imap[i], cent[::-1])
                    r = np.linspace(0, len(prof), len(prof)) * pixsize
                    bl = beam2bl(prof, np.deg2rad(r / 3600), cfg.lmax)
                    b_ell += [bl]

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
            else:
                ells, b_ell = harmbin(imap)

            # Load model profile
            mod_path = os.path.join(data_dir, "stack_profiles", split, spl_rel, f"model_profile_{spl_rel}_{epoch[0]}_{epoch[1]}.txt")

            # Make leakage
            prof_dir = os.path.join(data_dir, "leakage_profiles", split, spl_rel)
            os.makedirs(prof_dir, exist_ok=True)
            te_leakage = np.column_stack((ells, b_ell[1]))
            tb_leakage = np.column_stack((ells, b_ell[2]))

            # Load model profile and compute correction
            mod_path = os.path.join(data_dir, "stack_profiles", split, spl_rel, f"model_window_{spl_rel}_{epoch[0]}_{epoch[1]}.txt")
            model = np.genfromtxt(mod_path)
            rat_ell = np.arange(4000, 6000, 1)
            b_ml = np.interp(rat_ell, ells, b_ell[0], np.nan, np.nan)
            b_mod = np.interp(rat_ell, model[:, 0], model[:, 1], np.nan, np.nan)
            corr = np.nanmean(b_ml/b_mod)
            te_leakage[:, 1] /= model[0, 1] * corr
            tb_leakage[:, 1] /= model[0, 1] * corr

            np.savetxt(
                os.path.join(
                    prof_dir,
                    f"te_leakage_{spl_rel}_{epoch[0]}_{epoch[1]}.txt",
                ),
                te_leakage,
            )

            np.savetxt(
                os.path.join(
                    prof_dir,
                    f"tb_leakage_{spl_rel}_{epoch[0]}_{epoch[1]}.txt",
                ),
                tb_leakage,
            )

            # Save for plots
            to_plot["epoch"] += [f"{epoch[0]}_{epoch[1]}"] * len(te_leakage)
            to_plot["ell"] += te_leakage[:, 0].tolist()
            to_plot["te_leakage"] += te_leakage[:, 1].tolist()
            to_plot["tb_leakage"] += tb_leakage[:, 1].tolist()
            for sc, si in zip(split.split("+"), spl_rel.split("+")):
                to_plot[sc] += [si] * len(te_leakage)

    # Plot TB
    plt.close()
    plot = auto_relplot(to_plot, x="ell", y="tb_leakage", ignore=["te_leakage",], kind="line", estimator=None, hue=("band" if "tube_slot" in split else None))
    plot.set_axis_labels(r"$\ell$", r'Leakage ($B_{\ell}^{T \rightarrow B}/B_{0}^{T}$)')
    plot.set(xlim=(0, cfg.lmax))
    plt.suptitle(f"T->B Leakage by {split}")
    plt.subplots_adjust(top=0.85)
    plt.savefig(
        os.path.join(prof_plot_dir, f"tb_leakage_{split}.png"), bbox_inches="tight"
    )

    # Load ACT as reference
    if split == "band":
        for b in ["f090", "f150", "f220"]:
            act_te = np.genfromtxt(f"/global/cfs/cdirs/cmb/data/act_dr6/dr6.02/beams/leakage_beams/nominal/pa{4 if b == 'f220' else 6}_{b}_gamma_t2e.txt")
            act_main = np.genfromtxt(f"/global/cfs/cdirs/cmb/data/act_dr6/dr6.02/beams/main_beams/nominal/coadd_pa{4 if b == 'f220' else 6}_{b}_night_beam_tform_jitter_cmb.txt")
            to_plot["ell"] += act_te[:, 0].tolist()
            to_plot["te_leakage"] += (act_te[:, 1] * act_main[:, 1]/act_main[0, 1]).tolist()
            to_plot["epoch"] += ["ACT PA4/6"]*len(act_te)
            to_plot["band"] += [b]*len(act_te)
        del to_plot["tb_leakage"]

    # Plot TE
    plt.close()
    plot = auto_relplot(to_plot, x="ell", y="te_leakage", ignore=["tb_leakage",], kind="line", estimator=None, hue=("band" if "tube_slot" in split else None))
    plot.set_axis_labels(r"$\ell$", r'Leakage ($B_{\ell}^{T \rightarrow E}/B_{0}^{T}$)')
    plot.set(xlim=(0, cfg.lmax))
    plt.suptitle(f"T-> E Leakage by {split}")
    plt.subplots_adjust(top=0.85)
    plt.savefig(
        os.path.join(prof_plot_dir, f"te_leakage_{split}.png"), bbox_inches="tight"
    )
