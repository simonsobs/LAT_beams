import logging
import os
import sys
from functools import cache, partial, reduce
from glob import glob
from typing import cast

import numpy as np
import sqlalchemy as sqy
import yaml
from mpi4py import MPI
from pixell import enmap
from pshmem.locking import MPILock
from so3g.proj import RangesMatrix
from sotodlib import tod_ops
from sotodlib.coords import planets as cp
from sotodlib.core import Context, metadata
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.jobdb import Job

import lat_beams.mapmaking as lbm
from lat_beams.beam_utils import estimate_cent
from lat_beams.plotting import plot_map_complete
from lat_beams.utils import (
    ErrCode,
    fail,
    get_args_cfg,
    init_log,
    load_aman,
    log_lvl,
    set_tag,
    setup_cfg,
    setup_jobs,
    setup_paths,
)

tod_ops.filters.logger.setLevel(logging.ERROR)
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()


def get_jobdict(jdb):
    jobdict = {
        f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['array']}-{job.tags['band']}": job
        for job in jdb.get_jobs(jclass="beam_map")
    }
    return jobdict


def get_jobit(
    jdb,
    obs_ids,
    ctx,
    start_time,
    stop_time,
    source_list,
    pointing_type,
    logger,
    forced_ws,
):
    _ = jdb
    with log_lvl(logger, 25):
        if obs_ids is not None:
            obslist = [ctx.obsdb.get(obs_id) for obs_id in obs_ids]
        else:
            # src_str = "==1 or ".join(source_list) + "==1"
            obslists = [
                ctx.obsdb.query(
                    f"type=='obs' and subtype=='cal' and start_time > {start_time} and stop_time < {stop_time}",
                    tags=source_list[:i] + [f"{source}=1"] + source_list[(i + 1) :],
                )
                for i, source in enumerate(source_list)
            ]
            obslist = reduce(lambda q, p: p + q, obslists)
        if pointing_type != "pointing_model":
            dbs = [
                md["db"]
                for md in ctx["metadata"]
                if "focal_plane" in md.get("name", "")
            ]
            if len(dbs) > 1:
                if myrank == 0:
                    logger.warning(
                        "Multiple pointing metadata entries found, using the first one"
                    )
            elif len(dbs) == 0:
                if myrank == 0:
                    logger.error("No pointing metadata entries found")
                sys.exit()
            logger.info("Using ManifestDb at %s", dbs[0])
            db = metadata.ManifestDb(dbs[0])
            obs_ids = np.array([entry["obs:obs_id"] for entry in db.inspect()])
            obslist = [obs for obs in obslist if obs["obs_id"] in obs_ids]
            logger.info("Only %s observations with pointing metadata", len(obslist))

        obslist = np.array_split(obslist, nproc)[myrank]
        obsit = []
        for obs in obslist:
            try:
                det_info = ctx.get_det_info(obs["obs_id"])
            except:
                continue
            obs = ctx.obsdb.get(obs["obs_id"], tags=True)
            wafers = np.unique(
                [t[3:] for t in obs["tags"] if t[:2] == obs["tube_slot"]] + forced_ws
            )
            ws = det_info["wafer_slot"]
            if "wafer.wafer_slot" in det_info.keys:
                ws = np.array(
                    [
                        ws1 if ws1 != "ws." else ws2
                        for ws1, ws2 in zip(
                            det_info["wafer_slot"], det_info["wafer.wafer_slot"]
                        )
                    ]
                )
            wsufmsband = np.unique(
                np.column_stack(
                    [
                        ws,
                        det_info["stream_id"],
                        det_info["wafer.array"],
                        det_info["wafer.bandpass"],
                    ]
                ),
                axis=0,
            )
            for ws, sid, ufm, band in wsufmsband:
                if band[0] != "f":
                    continue
                if ws == "ws.":
                    continue
                if ws not in wafers and "all" not in obs["tags"]:
                    continue
                obsit += [(obs, ws, sid, ufm, band)]
    return obsit


def get_jobstr(info):
    obs, ws, sid, ufm, band = info
    job_str = f"{obs['obs_id']}-{ws}-{sid}-{ufm}-{band}"
    return job_str


def get_tags(info):
    obs, ws, sid, ufm, band = info
    tags = {
        "obs_id": obs["obs_id"],
        "wafer_slot": ws,
        "stream_id": sid,
        "array": ufm,
        "band": band,
        "errcode": 0,
        "message": "",
        "binned": "",
        "detweights": "",
        "solved": "",
        "weights": "",
        "ml_map": "",
        "ml_div": "",
        "ml_rhs": "",
        "ml_bin": "",
        "comps": "",
        "source": "",
        "config": "",
        "context": "",
        "preprocess": "",
        "splits": "",
        "X": "",
    }
    return tags


@cache
def load_det_splits(split_dir):
    det_splits = {}
    for fname in glob(os.path.join(split_dir, "*.txt")):
        name = os.path.splitext(os.path.basename(fname))[0]
        dets = np.genfromtxt(
            fname,
            dtype=str,
            usecols=[
                0,
            ],
        )
        det_splits[name] = dets
    return det_splits


def make_det_splits(aman, split_dir, min_dets, det_split_cfg, single_det=False):
    det_splits = {"full": RangesMatrix.zeros(aman.signal.shape)}
    if single_det:
        for i, det_id in enumerate(aman.det_info.det_id):
            rmat = RangesMatrix.ones(aman.signal.shape)
            rmat.ranges[i] = rmat.ranges[i].complement()
            det_splits = {det_id: rmat}
        return det_splits
    if "det_id" not in aman.det_info:
        return det_splits
    det_split_files = load_det_splits(split_dir)
    for det_split in det_split_cfg:
        if det_split in det_split_files:
            dets = det_split_files[det_split]
            msk = np.isin(aman.det_info.det_id, dets)
            if np.sum(msk) < min_dets / 2:
                continue
            rmat = RangesMatrix.from_mask(
                np.broadcast_to(~msk[..., None], aman.signal.shape)
            )
            det_splits[det_split] = rmat
        elif det_split == "left":
            det_splits["left"] = aman.preprocess.turnaround_flags.left_scan
        elif det_split == "right":
            det_splits["right"] = aman.preprocess.turnaround_flags.right_scan
        else:
            raise ValueError(f"Unknown det_split: {det_split}")

    return det_splits


# Setup logger
logger = init_log()
metadata.loader.logger = logger
cp.logger = logger
if logger.extra is None:
    raise ValueError("Logger doesn't have adapter set up!")
logger.extra = cast(dict, logger.extra)

# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg(
    args,
    cfg_dict,
    {
        "map_source_list": "source_list",
        "map_mask_size": "mask_size",
        "cgiters_single": "cgiters",
    },
)

if args.plot_only:
    logger.info("Running in plot_only mode!")

if cfg.preprocess_cfg is None:
    raise ValueError("Must specify a valid preprocess config!")
with open(cfg.preprocess_cfg) as f:
    preprocess_cfg = yaml.safe_load(f)
    preprocess_str = yaml.dump(preprocess_cfg)
if cfg.single_det:
    logger.info("Running in single det mode, changing comps from %s to T", cfg.comps)
    cfg.comps = "T"

# Check pointing_type
if cfg.pointing_type not in ["pointing_model", "per_obs", "raw"]:
    raise ValueError(f"Invalid pointing_type {cfg.pointing_type}")
if cfg.pointing_type == "raw" and cfg.comps != "T":
    logger.info("Running with raw pointing, changing comps from %s to T", cfg.comps)
    cfg.comps = "T"

# Setup folders
plot_dir, data_dir = setup_paths(
    cfg.root_dir,
    "beams",
    cfg.tel,
    f"{cfg.pointing_type}{(cfg.append!='')*'_'}{cfg.append}{(cfg.single_det)*'_single_det'}",
)

# Get context
with open(cfg.ctx_path) as f:
    ctx_str = yaml.dump(yaml.safe_load(f))
ctx = Context(cfg.ctx_path)
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")

# Modify preproc with our paths
preprocess_cfg["archive"]["index"] = os.path.join(
    data_dir, preprocess_cfg["archive"]["index"]
)
preprocess_cfg["archive"]["policy"]["filename"] = os.path.join(
    data_dir, preprocess_cfg["archive"]["policy"]["filename"]
)
preprocess_cfg["context_file"] = cfg.ctx_path
os.makedirs(os.path.dirname(preprocess_cfg["archive"]["index"]), exist_ok=True)
os.makedirs(os.path.dirname(preprocess_cfg["archive"]["index"]), exist_ok=True)

# Setup jobs
jdb, all_jobs = setup_jobs(
    comm,
    data_dir,
    "beam_map",
    get_jobdict,
    partial(
        get_jobit,
        obs_ids=args.obs_ids,
        ctx=ctx,
        start_time=cfg.start_time,
        stop_time=cfg.stop_time,
        source_list=cfg.source_list,
        pointing_type=cfg.pointing_type,
        logger=logger,
        forced_ws=cfg.forced_ws,
    ),
    get_jobstr,
    get_tags,
    cfg.source_list,
    args.overwrite,
    args.retry_failed,
    args.job_memory,
    args.job_memory_buffer,
    args.plot_only,
    logger,
)

# Even things out
joblist = np.array_split(all_jobs, nproc)[myrank].tolist()
n_maps = comm.allgather(len(joblist))
max_maps = np.max(n_maps)
if n_maps[0] != max_maps:
    raise ValueError("Root doesn't have max maps!")
joblist += [None] * (1 + max_maps - len(joblist))

# Get settings for source mask
pixsize = 3600 * np.rad2deg(cfg.res)
mask_fac = cfg.search_mask["xyr"][-1] / cfg.mask_size

# Setup passes
passes = lbm.get_passes(cfg)

# Local comm for ML map
l_comm = comm.Split(myrank, myrank)

# Profiler setup
profiler = None
if args.profile:
    from pyinstrument import Profiler

    profiler = Profiler()
    logger.info("Running in profiler mode! Only one job will be run per process")
    joblist = [joblist[0]]
    profiler.start()

# Mapping loop
source_list = set(cfg.source_list)
job = None
mpilock = MPILock(comm)
for i, j in enumerate(joblist):
    # To avoid multiproc issues where the database is locked we lock and unlock serially
    mpilock.lock()
    if job is not None:
        with jdb.session_scope() as session:
            session.merge(job)
            session.commit()
    job = None
    if j is not None:
        with jdb.session_scope() as session:
            job = session.get(Job, j.id)
            session.expunge(job)
    mpilock.unlock()

    if job is None:
        continue

    job.mark_visited()
    obs_id = job.tags["obs_id"]
    sid = job.tags["stream_id"]
    ufm = job.tags["array"]
    ws = job.tags["wafer_slot"]
    band = job.tags["band"]
    sub_id = f"{obs_id}:{ws}:{band}"
    obs = ctx.obsdb.get(obs_id, tags=True)
    logger.extra["extra"] = f" [{obs_id} {ufm} {band} ({i+1}/{n_maps[myrank]})]"

    if args.plot_only:
        logger.log(25, "Replotting")
        try:
            solved = enmap.read_map(os.path.join(data_dir, job.tags["solved"]))
            solved = cast(enmap.ndmap, solved)
        except FileNotFoundError:
            fail(
                job, ErrCode.MAP_MISSING, "Missing map files in plot_only mode", logger
            )
            continue

        obs_plot_dir = os.path.join(
            plot_dir, job.tags["source"], str(obs["timestamp"])[:5], obs_id
        )
        cent = estimate_cent(solved[0], cfg.smooth_kern / pixsize, cfg.buf)
        posmap = solved.posmap()
        posmap = np.rad2deg(posmap) * 3600
        if solved.wcs is None:
            raise ValueError("WCS is None")
        plot_map_complete(
            solved,
            posmap,
            solved.wcs.wcs.cdelt[1] * (60 * 60),
            cfg.extent,
            (posmap[1][cent], posmap[0][cent]),
            os.path.join(obs_plot_dir, ufm),
            f"{obs_id} {ufm} {band}",
            comps=job.tags["comps"],
            log_thresh=cfg.log_thresh,
            lognorm=float(1.0 / solved[0][cent]),
        )
        continue

    logger.log(25, "Mapping")

    # Save metadata and config info
    set_tag(job, "config", cfg_str)
    set_tag(job, "context", ctx_str)
    set_tag(job, "preprocess", preprocess_str)
    set_tag(job, "comps", cfg.comps)

    # Get metadata
    with log_lvl(logger, logging.ERROR):
        try:
            meta = ctx.get_meta(obs_id)
        except Exception as e:
            msg = f"Failed to load metadata with error {e}"
            fail(job, ErrCode.META, msg, logger)
            continue
    if meta.dets.count == 0:
        msg = "Looks like we don't have real metadata for this observation!"
        fail(job, ErrCode.META, msg, logger)
        continue
    fscale_fac = 90.0 / float(band[1:]) if cfg.apply_fscale else 1

    src_names = list(source_list & set(obs["tags"]))
    if len(src_names) > 1:
        logger.warning("\tObservation tagged for multiple sources!")
    elif len(src_names) == 0:
        msg = "Observation somehow not tagged for any sources in source_list! Skipping!"
        fail(job, ErrCode.SRC_TAG, msg, logger)
        logger.debug("Tags were: %s", obs["tags"])
        continue
    src_name = "_".join(src_names)
    logger.debug("Mapping %s", src_name)

    if "hits" in meta.focal_plane:
        meta.restrict("dets", meta.focal_plane.hits >= cfg.min_hits)
        if meta.dets.count < cfg.min_dets:
            msg = f"Only {meta.dets.count} detectors with good pointing fits!"
            fail(job, ErrCode.MIN_DETS, msg, logger)
            continue

    obs_plot_dir = os.path.join(
        plot_dir, src_name, str(obs["timestamp"])[:5], obs["obs_id"]
    )
    obs_data_dir = os.path.join(
        data_dir, src_name, str(obs["timestamp"])[:5], obs["obs_id"]
    )

    os.makedirs(obs_data_dir, exist_ok=True)

    src_to_map = src_name.split("_")[0]
    set_tag(job, "source", src_to_map)
    if src_to_map == "taua":
        src_to_map = ("tauA", 83.6272579, 22.02159891)
    elif src_to_map == "3c279":
        src_to_map = "J194.0409868m5.79174024"

    # Load and process the TOD
    aman = load_aman(
        obs["obs_id"],
        preprocess_cfg,
        {"wafer.array": ufm, "wafer.bandpass": band},
        job,
        cfg.min_dets,
        logger,
        fp_flag=True,
        save=(nproc == 1),
    )
    if aman is None:
        continue

    # Relcal cut
    if "relcal" in aman._fields:
        aman.restrict(
            "dets",
            (aman.relcal.rel_factor >= cfg.relcal_range[0])
            * (aman.relcal.rel_factor <= cfg.relcal_range[1]),
        )

    # Get initial source_flags
    with log_lvl(logger, logging.WARNING):
        source_flags = cp.compute_source_flags(
            tod=aman,
            P=None,
            mask=cfg.search_mask,
            center_on=src_to_map,
            res=cfg.res,
            max_pix=4e8,
            wrap=None,
            unroll=False,
        )

    # Do an aggressive filter and flag dets without the source
    cuts = lbm.make_cuts(
        aman, source_flags, min(len(aman.signal), 2 * cfg.n_modes), job, logger, cfg
    )
    if cuts is None:
        continue

    # Initial map
    info = {"obs_id": obs["obs_id"], "ufm": ufm, "band": band}
    out, cent, _ = lbm.make_map(
        aman,
        src_to_map,
        cfg.res,
        cuts,
        source_flags,
        "T",
        min(len(aman.signal), cfg.n_modes),
        pixsize,
        cfg.nominal_fwhm[band] * 60,
        None,
        cfg.min_det_secs * mask_fac * (fscale_fac**2),
        info,
        job,
        "initial",
        logger,
        cfg,
    )
    if out is None or cent is None:
        continue

    # Make a new mask with this center and the correct map size
    [[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(
        out["solved"].corners(corner=False)
    )
    mask = {
        "shape": "circle",
        "xyr": (
            (ra_min - pixsize * cent[1]) / 3600,
            (dec_min + pixsize * cent[0]) / 3600,
            cfg.mask_size * fscale_fac,
        ),
    }
    with log_lvl(logger, logging.WARNING):
        source_flags = cp.compute_source_flags(
            tod=aman,
            P=None,
            mask=mask,
            center_on=src_to_map,
            res=cfg.res,
            max_pix=4e8,
            wrap=None,
            unroll=False,
        )

    # Make splits
    det_splits = make_det_splits(aman, cfg.det_split_dir, cfg.min_dets, cfg.det_splits)
    set_tag(job, "splits", ",".join(det_splits.keys()))

    # Make final map
    out, cent, X = lbm.make_map(
        aman,
        src_to_map,
        cfg.res,
        cuts,
        source_flags,
        cfg.comps,
        min(len(aman.signal), cfg.n_modes),
        pixsize,
        cfg.nominal_fwhm[band] * 60,
        os.path.join(obs_data_dir, "{obs_id}_{ufm}_{band}_{map}.fits"),
        cfg.min_det_secs * (fscale_fac**2),
        info,
        job,
        "final",
        logger,
        cfg,
        det_splits,
    )
    set_tag(job, "X", X)
    if out is None or cent is None:
        continue

    # Add paths to job
    for name, fname, ext in [
        ("detweights", "detweights", "h5"),
        ("solved", "{split}_map", "fits"),
        ("weights", "{split}_weights", "fits"),
    ]:
        set_tag(
            job,
            name,
            os.path.relpath(
                os.path.join(obs_data_dir, f"{obs_id}_{ufm}_{band}_{fname}.{ext}"),
                data_dir,
            ),
        )

    # Plot
    for spl in out["splits"].keys():
        try:
            omap = out["splits"][spl]["solved"]
            if omap[0][cent] == 0:
                continue
            os.makedirs(os.path.join(obs_plot_dir, ufm, spl), exist_ok=True)
            posmap = omap.posmap()
            posmap = np.rad2deg(posmap) * 3600
            plot_map_complete(
                omap,
                posmap,
                omap.wcs.wcs.cdelt[1] * (60 * 60),
                cfg.extent,
                (posmap[1][cent], posmap[0][cent]),
                os.path.join(obs_plot_dir, ufm, spl),
                f"{obs_id} {ufm} {band} {spl}",
                comps=cfg.comps,
                log_thresh=cfg.log_thresh,
                lognorm=1.0 / omap[0][cent],
            )
        except Exception as e:
            logger.warning("Plotting for %s failed with error: %s", spl, e)
    omap = out["splits"]["full"]["solved"]

    # In case we don't want to make ML maps
    if cfg.mlpass < 1 or cfg.cgiters < 1:
        set_tag(job, "message", "Success")
        job.jstate = cast(sqy.Column[str], jobdb.JState.done)
        continue

    # Now make the ML map
    outmap, (mlmap_path, rhs_path, div_path, bin_path) = lbm.make_ml_map(
        {sub_id: (aman, out["P"])},
        passes,
        omap.shape,
        omap.wcs,
        f"{obs_id}_{ufm}_{band}_",
        obs_data_dir,
        l_comm,
        logger,
        cfg,
        guess=omap,
    )
    if mlmap_path == "" or outmap is None:
        msg = "Failed to make ML map"
        fail(job, ErrCode.ML_MAP, msg, logger)
        continue

    # Add paths to job
    for name, path in [
        ("ml_map", mlmap_path),
        ("ml_rhs", rhs_path),
        ("ml_div", div_path),
        ("ml_bin", bin_path),
    ]:
        set_tag(
            job,
            name,
            path,
        )

    # Plot
    try:
        if outmap.wcs is None:
            raise ValueError("WCS is None")
        posmap = outmap.posmap()
        posmap = np.rad2deg(posmap) * 3600
        plot_map_complete(
            outmap,
            posmap,
            outmap.wcs.wcs.cdelt[1] * (60 * 60),
            cfg.extent,
            (posmap[1][cent], posmap[0][cent]),
            os.path.join(obs_plot_dir, ufm),
            f"{obs_id} {ufm} {band} MLmap",
            log_thresh=cfg.log_thresh,
            lognorm=float(1.0 / outmap[0][cent]),
        )
    except Exception as e:
        logger.warning("Plotting failed with error: %s", e)

    set_tag(job, "message", "Success")
    job.jstate = cast(sqy.Column[str], jobdb.JState.done)

if args.profile and profiler is not None:
    profiler.stop()
    profiler.write_html(f"profile_{myrank}.html")

logger.extra["extra"] = ""
comm.barrier()
mpilock.close()
