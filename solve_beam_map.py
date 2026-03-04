import logging
import os
import sys

import astropy.units as u
import numpy as np
import yaml
from mpi4py import MPI
from pixell import enmap
from sotodlib import tod_ops
from sotodlib.coords import planets as cp
from sotodlib.core import Context, metadata

import lat_beams.mapmaking as lbm
from lat_beams import beam_utils as bu
from lat_beams.plotting import plot_map_complete
from lat_beams.utils import (
    get_args_cfg,
    init_log,
    load_aman,
    make_jobdb,
    setup_cfg,
    setup_paths,
)

tod_ops.filters.logger.setLevel(logging.ERROR)
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

# Setup logger
logger = init_log()
metadata.loader.logger = logger

# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg(args, cfg_dict, {"cgiters_full": "cgiters"})
ctx = Context(cfg.ctx_path)
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
pixsize = 3600 * np.rad2deg(cfg.res)

# Setup folders
plot_dir, data_dir_root = setup_paths(
    cfg.root_dir,
    "beams",
    cfg.tel,
    f"{cfg.pointing_type}{(cfg.append!="")*'_'}{cfg.append}",
)
fpath = os.path.join(data_dir_root, "beam_pars.h5")
plot_dir = os.path.join(plot_dir, "ml_maps")
data_dir = os.path.join(data_dir_root, "ml_maps")
os.makedirs(plot_dir, exist_ok=True)
jdb = make_jobdb(None, data_dir_root)

# TODO: Loop to only load one epoch at a time
# Load and filter fits in rank 0
fjobs = None
if myrank == 0:
    fjobs = jdb.get_jobs(jclass="fit_map", jstate="done")
    logger.info(f"{len(fjobs)} sub_ids to map")
    all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs)
    snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
    fwhm_exp = (
        np.array([cfg.nominal_fwhm[band] for band in all_fits["band"]]) * u.arcmin
    )
    sang_exp = (2 * np.pi * (fwhm_exp.to(u.radian) / 2.355) ** 2).to(u.sr)
    data_fwhm = bu.get_fit_vec(all_fits, "data_fwhm")
    solid_angle = bu.get_fit_vec(all_fits, "data_solid_angle_corr")
    msk = snr > 100
    msk *= data_fwhm < 1.5 * fwhm_exp
    msk *= data_fwhm > 0.5 * fwhm_exp
    msk *= solid_angle < 2 * sang_exp
    msk *= solid_angle > 0.25 * sang_exp
    all_fits = all_fits[msk]
    fjobs = np.array(fjobs)[msk]
# Now distribute jobs
comm.scatter(np.array_split(fjobs, nproc), root=0)
fjobs = fjobs[:2]
nkept = len(fjobs)
nkept_all = np.array(comm.allgather(nkept))
if np.sum(nkept_all) == 0:
    logger.info("Nothing to map!")
    sys.exit(0)
if np.any(nkept_all == 0):
    if nkept == 0:
        logger.info("No tods assigned to this process. Pruning")
    comm = mapmaking.prune_mpi(comm, np.where(nkept_all > 0)[0])
all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs)

# Get preproc config
if cfg.preprocess_cfg is None:
    raise ValueError("Must specify a valid preprocess config!")
with open(cfg.preprocess_cfg, "r") as f:
    preprocess_cfg = yaml.safe_load(f)
preprocess_cfg["archive"]["index"] = os.path.join(
    data_dir_root, preprocess_cfg["archive"]["index"]
)
preprocess_cfg["archive"]["policy"]["filename"] = os.path.join(
    data_dir_root, preprocess_cfg["archive"]["policy"]["filename"]
)
os.makedirs(os.path.dirname(preprocess_cfg["archive"]["index"]), exist_ok=True)
os.makedirs(os.path.dirname(preprocess_cfg["archive"]["index"]), exist_ok=True)

# Make template map
ext_rad = np.deg2rad(cfg.extent / 3600)
pix_extent = 2 * int(cfg.extent // pixsize)
twcs = enmap.wcsutils.build(
    [0, 0],
    res=np.rad2deg(cfg.res),
    shape=(pix_extent, pix_extent),
    system="tan",
    rowmajor=True,
)
tmap = enmap.zeros((3, pix_extent, pix_extent), twcs)
[[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(tmap.corners(corner=False))
plt_extent = (ra_min, ra_max, dec_min, dec_max)

# Load and process TODs
amans = {}
msk = np.ones(len(all_fits), bool)
for i, (job, fit) in enumerate(zip(fjobs, all_fits)):
    obs_id = job.tags["obs_id"]
    ws = job.tags["wafer_slot"]
    band = job.tags["band"]
    sub_id = f"{obs_id}:{ws}:{band}"

    aman = load_aman(
        obs_id,
        preprocess_cfg,
        {"wafer_slot": ws, "wafer.bandpass": band},
        job,
        cfg.min_dets,
        logger,
        fp_flag=True,
        save=(nproc == 1),
    )
    if aman is None:
        logger.warning("Could not add %s", sub_id)
        msk[i] = False
        continue

    # Normalize by the fit amp
    aman.signal /= fit["aman"].gauss.amp.value

    # Make projection operator
    cent = np.array(
        (fit["aman"].gauss.xi0.to(u.rad).value, fit["aman"].gauss.eta0.to(u.rad).value)
    )
    aman.focal_plane.xi += cent[0]
    aman.focal_plane.eta -= cent[1]
    P, X = cp.get_scan_P(
        aman, job.tags["source"], res=cfg.res, comps=cfg.comps, threads="domdir"
    )
    P.geom = enmap.Geometry(shape=tmap.shape, wcs=tmap.wcs)

    logger.warning("Added %s", sub_id)
    amans[sub_id] = (aman, P)
all_ids = np.array(list(amans.keys()))
all_fits = all_fits[msk]

# Loop through splits
passes = lbm.get_passes(cfg)
for split in cfg.split_by:
    logger.info(f"Splitting by {split}")
    split_vec = bu.get_split_vec(all_fits, split, ctx)
    for spl in np.unique(split_vec):
        data_dir_spl = os.path.join(data_dir, split, spl)
        plot_dir_spl = os.path.join(plot_dir, split, spl)
        os.makedirs(data_dir_spl, exist_ok=True)
        os.makedirs(plot_dir_spl, exist_ok=True)

        smsk = split_vec == spl
        sids = all_ids[smsk]
        sfits = all_fits[smsk]
        for epoch in cfg.epochs:
            comm.barrier()
            plot_dir_epc = os.path.join(plot_dir_spl, f"{epoch[0]}_{epoch[1]}")
            data_dir_epc = os.path.join(data_dir_spl, f"{epoch[0]}_{epoch[1]}")
            os.makedirs(plot_dir_epc, exist_ok=True)
            os.makedirs(data_dir_epc, exist_ok=True)
            logger.info(f"{spl} {epoch}")
            times = sfits["time"]
            tmsk = (times >= epoch[0]) * (times < epoch[1])
            ntods = np.sum(tmsk)
            logger.normal("Have %d TODs in rank", ntods)

            # Make a comm with just the procs that have TODs loaded
            # This is somewhat innefecient, in an ideal world I can provide an optimal TOD splitting scheme
            run_comm = comm.Split(ntods > 0, myrank)
            if ntods == 0:
                continue
            amans_to_map = {sid: amans[sid] for sid in sids[tmsk]}

            # TODO: Load stack as the initial guess
            outmap, (mlmap_path, rhs_path, div_path, bin_path) = lbm.make_ml_map(
                amans_to_map,
                passes,
                tmap.shape,
                tmap.wcs,
                f"{spl}_{epoch[0]}_{epoch[1]}_",
                data_dir_epc,
                run_comm,
                logger,
                cfg,
                guess=None,
            )
            # outmap[outmap > 1] = 1

            # Plot
            posmap = np.rad2deg(outmap.posmap()) * 3600
            for append, smap in [
                ("", outmap),
                ("_smooth3pix", enmap.smooth_gauss(outmap, 3 * cfg.res)),
            ]:
                plot_map_complete(
                    smap,
                    posmap,
                    pixsize,
                    cfg.extent,
                    (0, 0),
                    plot_dir_epc,
                    f"{spl} {epoch[0]} {epoch[1]}",
                    log_thresh=cfg.log_thresh,
                    append=append,
                    qrur=True,
                )

    comm.barrier()
