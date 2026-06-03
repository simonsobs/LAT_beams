# TODO: jobdb
# TODO: speed up: mpi? asyncio?
import os
import sys
from glob import glob
from typing import cast

import astropy.units as u
import numpy as np
from pixell import enmap, reproject
from sotodlib.core import Context
from tqdm import tqdm

from lat_beams import beam_utils as bu
from lat_beams.plotting import plot_map_complete
from lat_beams.utils import get_args_cfg, make_jobdb, setup_cfg, setup_paths, init_log


def view_TQU(imap):
    padded = imap
    if len(imap) == 1:
        padded = enmap.zeros((3,) + imap.shape[1:], imap.wcs)
        padded[0][:] = imap[0][:]
    return padded


nominal_fwhm = {"f090": 2.0, "f150": 1.3, "f220": 0.95, "f280": 0.83}  # arcmin

# Setup logger
logger = init_log()

# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg(
    args,
    cfg_dict,
    {
        "map_mask_size": "mask_size",
    },
)
ctx = Context(cfg.ctx_path)
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
pixsize = 3600 * np.rad2deg(cfg.res)
op = np.ndarray.__iadd__

# Setup folders
plot_dir, data_dir = setup_paths(
    cfg.root_dir,
    "beams",
    cfg.tel,
    f"{cfg.pointing_type}{(cfg.append!="")*'_'}{cfg.append}",
)
plot_dir = os.path.join(plot_dir, "stacks")
os.makedirs(plot_dir, exist_ok=True)
fpath = os.path.join(data_dir, "beam_pars.h5")
jdb = make_jobdb(None, data_dir)

# Get jobs
mjobdict = {
    f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['band']}": job
    for job in jdb.get_jobs(jclass="beam_map", jstate="done")
}
fjobs = np.array(jdb.get_jobs(jclass="fit_map", jstate="done"))

logger.info("%d maps to add", len(fjobs))
if len(fjobs) == 0:
    sys.exit(0)

# Load fits
all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs.tolist())
snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
solid_angle = bu.get_fit_vec(all_fits, "gauss.data_solid_angle_corr")
msk = snr > 100
msk *= solid_angle > 0
all_fits = all_fits[msk]
fjobs = fjobs[msk]

# Make template map
ext_rad = np.deg2rad(cfg.mask_size)
pix_extent = 2 * int(3600 * cfg.mask_size // pixsize)
# rowmajor = True here to match sotodlib
twcs = enmap.wcsutils.build(
    [0, 0],
    res=np.rad2deg(cfg.res),
    shape=(pix_extent, pix_extent),
    system="tan",
    rowmajor=True,
)
tmap = enmap.zeros((3, pix_extent, pix_extent), twcs)

if args.plot_only:
    logger.info("Running in plot only mode!")

# Get det splits
det_split_names = [""]
if cfg.det_split_dir != "":
    det_split_names += [
        os.path.splitext(os.path.basename(fname))[0]
        for fname in glob(os.path.join(cfg.det_split_dir, "*.txt"))
    ]

# Loop through splits
map_types = ("", "resid")  # , "ml") # Skipping ML for now
for split in cfg.split_by:
    logger.info("Splitting by %s", split)
    split_vec = bu.get_split_vec(all_fits, split, ctx, metasplits=cfg.metasplits)
    for spl in np.unique(split_vec):
        if "NOMATCH!" in spl:
            continue
        data_dir_spl = os.path.join(data_dir, "stacks", split, spl)
        plot_dir_spl = os.path.join(plot_dir, split, spl)
        os.makedirs(data_dir_spl, exist_ok=True)
        os.makedirs(plot_dir_spl, exist_ok=True)

        smsk = split_vec == spl
        sfits = all_fits[smsk]
        sfjobs = fjobs[smsk]
        fwhm_exp = np.array([nominal_fwhm[band] for band in sfits["band"]]) * u.arcmin
        sang_exp = (2 * np.pi * (fwhm_exp.to(u.radian) / 2.355) ** 2).to(u.sr)
        data_fwhm = bu.get_fit_vec(sfits, "data_fwhm")
        msk = data_fwhm < 3 * fwhm_exp
        msk *= data_fwhm < np.percentile(data_fwhm[msk], 95)
        sfits = sfits[msk]
        sfjobs = sfjobs[msk]
        for epoch in cfg.epochs:
            plot_dir_epc = os.path.join(plot_dir_spl, f"{epoch[0]}_{epoch[1]}")
            os.makedirs(plot_dir_epc, exist_ok=True)
            logger.info("Running %s %s", spl, str(epoch))
            times = sfits["time"]
            tmsk = (times >= epoch[0]) * (times < epoch[1])
            if np.sum(tmsk) == 0:
                logger.warning("No maps found! Skipping...")
                continue
            # Structure here is split -> type (map/ML/resid)
            mcoadd = {
                name: {
                    map_type: enmap.zeros(tmap.shape, tmap.wcs)
                    for map_type in map_types
                }
                for name in det_split_names
            }
            wcoadd = {
                name: {
                    map_type: enmap.zeros(tmap.shape, tmap.wcs)
                    for map_type in map_types
                }
                for name in det_split_names
            }
            for fit, fjob in tqdm(zip(sfits[tmsk], sfjobs[tmsk]), total=np.sum(tmsk)):
                jobstr = f"{fjob.tags['obs_id']}-{fjob.tags['wafer_slot']}-{fjob.tags['stream_id']}-{fjob.tags['band']}"
                split = fjob.tags["split"]
                if jobstr not in mjobdict:
                    logger.debug("Map job not found for %s", jobstr)
                    continue
                mjob = mjobdict[jobstr]
                if args.plot_only:
                    continue
                # Load
                for map_type in map_types:
                    if map_type == "":
                        map_path = os.path.join(data_dir, mjob.tags["solved"])
                        ivar_path = os.path.join(data_dir, mjob.tags["weights"])
                        if split != "":
                            map_path = map_path.replace(
                                "solved.fits", f"{split}_map.fits"
                            )
                            ivar_path = map_path.replace("map.fits", "weights.fits")
                    elif map_type == "resid":
                        map_path = os.path.join(data_dir, fjob.tags["resid"])
                        ivar_path = os.path.join(data_dir, fjob.tags["resid_weights"])
                    else:
                        raise ValueError(f"Bad map type {map_type}")
                    try:
                        imap = enmap.read_map(map_path)
                        if len(imap.shape) == 2:
                            imap = imap.reshape((1,) + imap.shape)
                        ivar = enmap.read_map(ivar_path)
                        if len(ivar.shape) == 4:
                            ivar = ivar[np.diag_indices(len(ivar))]
                        ivar = ivar.reshape(imap.shape)
                    except FileNotFoundError:
                        logger.debug("Maps missing for job: %s-%s", jobstr, split)
                        continue
                    # Make everything look like TQU
                    imap = view_TQU(imap)
                    ivar = view_TQU(ivar)

                    # Crop, recenter, and normalize
                    cent = np.array(
                        (
                            fit["aman"].gauss.eta0.to(u.rad).value,
                            fit["aman"].gauss.xi0.to(u.rad).value,
                        )
                    )
                    imap = (
                        reproject.thumbnails(
                            imap,
                            r=ext_rad,
                            coords=cent,
                            oshape=(pix_extent, pix_extent),
                            owcs=twcs,
                        )
                        / fit["aman"].gauss.amp.value
                    )
                    ivar = (
                        reproject.thumbnails_ivar(
                            ivar,
                            r=ext_rad,
                            coords=cent,
                            oshape=(pix_extent, pix_extent),
                            owcs=twcs,
                        )
                        * fit["aman"].gauss.amp.value**2
                    )

                    # If the new center seems very far from the origin then lets skip
                    if map_type == "":
                        cent_est = bu.estimate_cent(imap[0], sigma=10, buf=1)
                        dist = np.linalg.norm(cent_est - imap.wcs.wcs.crpix)
                        if dist > cfg.miscenter_thresh:
                            logger.debug(
                                "%s %s (%s) seems miscentered! Skipping!", jobstr, split, mjob.tags['source']
                            )
                            break

                    # Add
                    np.nan_to_num(imap, copy=False, nan=0, posinf=0, neginf=0)
                    np.nan_to_num(ivar, copy=False, nan=0, posinf=0, neginf=0)
                    mcoadd[split][map_type].insert(imap * ivar, op=op)
                    wcoadd[split][map_type].insert(ivar, op=op)

            # Divide weights
            for split in mcoadd.keys():
                for map_type in map_types:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        mcoadd[split][map_type] /= wcoadd[split][map_type]  # type: ignore
                    np.nan_to_num(
                        mcoadd[split][map_type], copy=False, nan=0, posinf=0, neginf=0
                    )
                    # Save and plot
                    for omap, name in [(mcoadd[split][map_type], "stack"), (wcoadd[split][map_type], "stack_ivar")]:
                        path = os.path.join(
                            data_dir_spl,
                            f"{spl}_{epoch[0]}_{epoch[1]}{'_'*bool(split)}{split}{'_'*bool(map_type)}{map_type}_{name}.fits",
                        )
                        if args.plot_only:
                            if not os.path.isfile(path):
                                logger.warning("Maps do not exist!")
                                continue
                            omap = enmap.read_map(path)
                        else:
                            enmap.write_map(
                                path,
                                omap,
                                "fits",
                                allow_modify=True,
                            )
                        if "ivar" in name:
                            continue

                        omap = cast(enmap.ndmap, omap)
                        posmap = omap.posmap()
                        posmap = np.rad2deg(posmap) * 3600
                        for append, smap in [
                            ("", omap),
                            ("_smooth3pix", enmap.smooth_gauss(omap, 3 * cfg.res)),
                        ]:
                            plot_map_complete(
                                smap,
                                posmap,
                                pixsize,
                                cfg.extent,
                                (0, 0),
                                plot_dir_epc,
                                f"{spl} {epoch[0]} {epoch[1]}{' '*bool(split)}{split}{' '*bool(map_type)}{map_type}",
                                log_thresh=cfg.log_thresh,
                                append=name + append,
                                qrur=True,
                            )
