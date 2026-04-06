import gc
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


def get_jobdict(jdb):
    jobdict = {
        f"{job.tags['split']}-{job.tags['split_str']}-{job.tags['epoch_start']}-{job.tags['epoch_end']}": job
        for job in jdb.get_jobs(jclass="solve_maps")
    }
    return jobdict


def get_jobit(jdb, cfg, all_fits, all_fjobs):
    jobit = []
    if comm.Get_rank == 0:
        for epoch in cfg.epochs:
            times = all_fits["time"]
            tmsk = (times >= epoch[0]) * (times < epoch[1])
            fjobs = all_fjobs[tmsk]
            fjobs = np.array_split(fjobs, nproc)
            fits = bu.load_beam_fits_from_jobs(fpath, fjobs)
            for split in cfg.split_by:
                split_vec = bu.get_split_vec(fits, split, ctx)
                for spl in np.unique(split_vec):
                    jobit += [(split, spl, epoch[0], epoch[1])]
    return jobit


def get_jobstr(info):
    return f"{info[0]}-{info[1]}-{info[2]}-{info[3]}"


def get_tags(info):
    split, split_str, epoch_start, epoch_end = info
    tags = {
        "split": split,
        "split_str": split_str,
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
        "message": "",
        "ml_map": "",
        "ml_div": "",
        "ml_rhs": "",
        "ml_bin": "",
        "comps": "",
        "config": "",
        "context": "",
        "preprocess": "",
        "obslist": "",
    }
    return tags


# Setup logger
logger = init_log()
metadata.loader.logger = logger

# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg(args, cfg_dict, {"cgiters_full": "cgiters"})
pixsize = 3600 * np.rad2deg(cfg.res)

# Get context
with open(cfg.ctx_path, "r") as f:
    ctx_str = yaml.dump(yaml.safe_load(f))
ctx = Context(cfg.ctx_path)
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")

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

# Load and filter fits in rank 0
all_fjobs = None
if myrank == 0:
    all_fjobs = jdb.get_jobs(jclass="fit_map", jstate="done")
    logger.info(f"{len(all_fjobs)} sub_ids to map")
    all_fits = bu.load_beam_fits_from_jobs(fpath, all_fjobs)
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
    all_fjobs = np.array(all_fjobs)[msk]

# Setup jobs
jdb, all_jobs = setup_jobs(
    comm,
    data_dir_root,
    "solve_maps",
    get_jobdict,
    partial(
        get_jobit,
        all_fits=all_fits,
        all_fjobs=all_fjobs,
    ),
    get_jobstr,
    get_tags,
    [],
    args.overwrite,
    args.retry_failed,
    args.job_memory,
    args.job_memory_buffer,
    args.plot_only,
    logger,
)

# Loop through epochs
passes = lbm.get_passes(cfg)
for epoch in cfg.epochs:
    comm.barrier()
    logger.info(f"Running for epoch {epoch}")

    # Find all jobs with this epoch
    ejobs = [
        job
        for job in all_jobs
        if job.tags["epoch_start"] == epoch[0] and job.tags["epoch_end"] == epoch[1]
    ]
    if len(ejobs) == 0:
        logger.info("No open jobs found!")
        continue

    # Split up fits in this epoch
    fjobs = None
    if myrank == 0:
        times = all_fits["time"]
        tmsk = (times >= epoch[0]) * (times < epoch[1])
        fjobs = all_fjobs[tmsk]
        fjobs = np.array_split(fjobs, nproc)

    # Now distribute jobs
    fjobs = comm.scatter(fjobs, root=0)
    nkept = len(fjobs)
    nkept_all = np.array(comm.allgather(nkept))
    if np.sum(nkept_all) == 0:
        logger.info("Nothing to map!")
        continue
    # if np.any(nkept_all == 0):
    #     if nkept == 0:
    #         logger.info("No tods assigned to this process. Pruning")
    #     comm = mapmaking.prune_mpi(comm, np.where(nkept_all > 0)[0])
    fits = bu.load_beam_fits_from_jobs(fpath, fjobs)

    # Load and process TODs
    amans = {}
    msk = np.ones(len(fits), bool)
    for i, (job, fit) in enumerate(zip(fjobs, fits)):
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
            (
                fit["aman"].gauss.xi0.to(u.rad).value,
                fit["aman"].gauss.eta0.to(u.rad).value,
            )
        )
        aman.focal_plane.xi += cent[0]
        aman.focal_plane.eta -= cent[1]
        planet = cp.SlowSource.for_named_source(job.tags[source], aman.timestamps[0])
        ra0, dec0 = planet.pos(aman.timestamps.mean())
        rot = ~quat.rotation_lonlat(ra0, dec0)
        P = coords.P.for_tod(
            aman, comps=cfg.comps, threads="domdir", wcs_kernel=tmap.wcs, rot=rot
        )
        # P, X = cp.get_scan_P(
        #     aman, job.tags["source"], res=cfg.res, comps=cfg.comps, threads="domdir"
        # )
        P.geom = enmap.Geometry(shape=tmap.shape, wcs=tmap.wcs)

        logger.debug("Added %s", sub_id)
        amans[sub_id] = (aman, P)
    all_ids = np.array(list(amans.keys()))
    fits = fits[msk]

    # Loop through jobs
    for split in cfg.split_by:
        logger.info(f"Splitting by {split}")
        sjobs = [job for job in ejobs if job.tags["split"] == split]
        split_vec = bu.get_split_vec(fits, split, ctx)
        for j in sjobs:
            comm.barrier()

            data_dir_spl = os.path.join(data_dir, split, spl)
            plot_dir_spl = os.path.join(plot_dir, split, spl)
            plot_dir_epc = os.path.join(plot_dir_spl, f"{epoch[0]}_{epoch[1]}")
            data_dir_epc = os.path.join(data_dir_spl, f"{epoch[0]}_{epoch[1]}")
            os.makedirs(plot_dir_epc, exist_ok=True)
            os.makedirs(data_dir_epc, exist_ok=True)
            logger.info(f"Mapping {spl} {epoch}")

            smsk = split_vec == spl
            sids = all_ids[smsk]
            logger.normal("Have %d TODs in rank", len(sids))
            all_sids = comm.reduce(sids)

            # Have rank 0 handle the jobdb
            if myrank == 0:
                with jdb.session_scope() as session:
                    job = session.get(Job, j.id)
                    session.expunge(job)
                job.mark_visited()
                # Save metadata and config info
                set_tag(job, "config", cfg_str)
                set_tag(job, "context", ctx_str)
                set_tag(job, "preprocess", preprocess_str)
                set_tag(job, "comps", cfg.comps)
                set_tag(job, "obslist", ",".join(all_sids))

            # Make a comm with just the procs that have TODs loaded
            # This is somewhat innefecient, in an ideal world I can provide an optimal TOD splitting scheme
            run_comm = comm.Split(len(sids) > 0, myrank)
            jobdat = ("", "", "", "", "")
            if len(sids) > 0:
                amans_to_map = {sid: amans[sid] for sid in sids}

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
                message = "Success!"

                # Plot
                if run_comm.Get_rank() == 0:
                    try:
                        posmap = np.rad2deg(outmap.posmap()) * 3600
                        for append, smap in [
                            ("ML", outmap),
                            ("ML_smooth3pix", enmap.smooth_gauss(outmap, 3 * cfg.res)),
                        ]:
                            plot_map_complete(
                                smap,
                                posmap,
                                pixsize,
                                cfg.extent,
                                (0, 0),
                                plot_dir_epc,
                                f"{spl} {epoch[0]} {epoch[1]} ML",
                                log_thresh=cfg.log_thresh,
                                append=append,
                                qrur=True,
                            )
                    except:
                        message = "Plotting failed!"
                    jobdat = comm.bcast(jobdat)

            if myrank == 0:
                for m, d in zip(
                    ("message", "ml_map", "ml_div", "ml_rhs", "ml_bin"), jobdat
                ):
                    set_tag(job, m, d)
                job.jstate = "done"
                with jdb.session_scope() as session:
                    session.merge(job)
                    session.commit()

        del amans
        gc.collect()
        comm.barrier()
logger.flush()
comm.barrier()
