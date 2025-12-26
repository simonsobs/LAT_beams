import glob
import logging
import os
import sys
import time
from functools import partial

import mpi4py.rc
import numpy as np
import yaml
from pixell import enmap
from so3g.proj import RangesMatrix
from sotodlib import tod_ops
from sotodlib.coords import planets as cp
from sotodlib.core import Context, metadata

from lat_beams.beam_utils import estimate_cent
from lat_beams.plotting import plot_map
from lat_beams.utils import get_args_cfg, init_log, load_aman, set_tag, setup_jobs

mpi4py.rc.threads = False
from mpi4py import MPI

mpi4py.rc.threads = False
from mpi4py import MPI

tod_ops.filters.logger.setLevel(logging.ERROR)
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

band_names = {"m": ["f090", "f150"], "u": ["f220", "f280"]}
comps = "TQU"


def get_jobdict(jdb):
    jobdict = {
        f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['band']}": job
        for job in jdb.get_jobs(jclass="beam_map")
    }
    return jobdict


def get_jobit(jdb, obs_ids, ctx, start_time, stop_time, source_list, pointing_type, L):
    lvl = metadata.loader.logger.level
    metadata.loader.logger.setLevel(25)
    if obs_ids is not None:
        obslist = [ctx.obsdb.get(obs_id) for obs_id in obs_ids]
    else:
        src_str = "==1 or ".join(source_list) + "==1"
        obslist = ctx.obsdb.query(
            f"type=='obs' and subtype=='cal' and start_time > {start_time} and stop_time < {stop_time} and {src_str}",
            tags=source_list,
        )
    if pointing_type != "pointing_model":
        dbs = [
            md["db"] for md in ctx["metadata"] if "focal_plane" in md.get("name", "")
        ]
        if len(dbs) > 1:
            if myrank == 0:
                L.warning(
                    "Multiple pointing metadata entries found, using the first one"
                )
        elif len(dbs) == 0:
            if myrank == 0:
                L.error("No pointing metadata entries found")
            sys.exit()
        L.info(f"Using ManifestDb at {dbs[0]}")
        db = metadata.ManifestDb(dbs[0])
        obs_ids = np.array([entry["obs:obs_id"] for entry in db.inspect()])
        obslist = [obs for obs in obslist if obs["obs_id"] in obs_ids]
        L.info(f"Only {len(obslist)} observations with pointing metadata")

    obslist = np.array_split(obslist, nproc)[myrank]
    obsit = []
    for obs in obslist:
        try:
            det_info = ctx.get_det_info(obs["obs_id"])
        except:
            continue
        wsufmsband = np.unique(
            np.column_stack(
                [
                    det_info["wafer_slot"],
                    det_info["stream_id"],
                    det_info["wafer.bandpass"],
                ]
            ),
            axis=0,
        )
        for ws, ufm, band in wsufmsband:
            if band[0] != "f":
                continue
            obsit += [(obs, ws, ufm, band)]
    metadata.loader.logger.setLevel(lvl)
    return obsit


def get_jobstr(info):
    obs, ws, ufm, band = info
    job_str = f"{obs['obs_id']}-{ws}-{ufm}-{band}"
    return job_str


def get_tags(info):
    obs, ws, ufm, band = info
    tags = {
        "obs_id": obs["obs_id"],
        "wafer_slot": ws,
        "stream_id": ufm,
        "band": band,
        "message": "",
        "binned": "",
        "detweights": "",
        "solved": "",
        "weights": "",
        "comps": "",
        "source": "",
        "config": "",
        "context": "",
        "preprocess": "",
    }
    return tags


def make_plots(solved, cent, extent, obs_plot_dir, obs_id, ufm, band, zoom, log_thresh):
    pixsize = solved.wcs.wcs.cdelt[1] * (60 * 60)
    ufm_plot_dir = os.path.join(obs_plot_dir, ufm)
    os.makedirs(ufm_plot_dir, exist_ok=True)
    [[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(
        solved.corners(corner=False)
    )
    plt_extent = [ra_min, ra_max, dec_min, dec_max]
    plt_cent = (ra_min - pixsize * cent[1], dec_min + pixsize * cent[0])
    map_norm = 1.0 / solved[0][cent]
    if not np.isfinite(map_norm):
        map_norm = 0.0
    for i, comp in enumerate(comps):
        plot_map(
            solved[i],
            pixsize,
            extent,
            plt_extent,
            cent,
            plt_cent,
            zoom,
            ufm_plot_dir,
            obs_id,
            ufm,
            band,
            comp,
            False,
            log_thresh,
        )
        plot_map(
            map_norm * solved[i],
            pixsize,
            extent,
            plt_extent,
            cent,
            plt_cent,
            zoom,
            ufm_plot_dir,
            obs_id,
            ufm,
            band,
            comp,
            True,
            log_thresh,
        )


def make_cuts(aman, source_flags, n_modes, job, L):
    sig_filt = cp.filter_for_sources(
        tod=aman,
        signal=aman.signal.copy(),
        source_flags=source_flags,
        n_modes=n_modes,
    )
    smsk = source_flags.mask()
    sig_filt_src = sig_filt.copy()
    sig_filt_src[~smsk] = np.nan
    sig_filt[smsk] = np.nan
    all_src = np.all(smsk, axis=-1)
    no_src = ~np.any(smsk, axis=-1)
    sdets = ~(all_src + no_src)
    peak_snr = np.zeros(len(sig_filt))
    if np.sum(sdets) > 0:
        with np.errstate(divide="ignore"):
            peak_snr[sdets] = np.nanmax(sig_filt_src[sdets], axis=-1) / np.nanstd(
                np.diff(sig_filt[sdets], axis=-1)
            )
    to_cut = peak_snr < min_snr  # + ~np.isfinite(peak_snr)
    to_cut[~sdets] = False
    cuts = RangesMatrix.from_mask(np.zeros_like(aman.signal, bool) + to_cut[..., None])
    L.debug(f"\tCutting {np.sum(to_cut)} detectors from map")
    if np.sum(~to_cut) < min_dets:
        msg = f"Not enough detectors after source flag cuts!"
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        return None
    return cuts


def make_map(
    aman,
    src_to_map,
    res,
    cuts,
    source_flags,
    comps,
    n_modes,
    filename,
    min_det_secs,
    job,
    map_str,
    L,
):
    # Get time on source
    det_secs = np.sum((source_flags * ~cuts).get_stats()["samples"]) * np.mean(
        np.diff(aman.timestamps)
    )
    L.debug(f"\t{det_secs} detector seconds on source in {map_str} mask")
    if det_secs < min_det_secs:
        msg = f"\tNot enough time on source in {map_str} mask."
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        return None, None

    # Initial map
    lvl = L.level
    L.setLevel(logging.WARNING)
    out = cp.make_map(
        aman,
        center_on=src_to_map,
        res=res,
        cuts=cuts,
        source_flags=source_flags,
        comps=comps,
        filename=filename,
        n_modes=n_modes,
        info={"obs_id": obs["obs_id"], "ufm": ufm, "band": band},
    )
    L.setLevel(lvl)

    # Smooth and find the center
    cent = estimate_cent(out["solved"][0], smooth_kern / pixsize, buf)

    # Estimate SNR
    peak = out["solved"][0][cent]
    snr = peak / tod_ops.jumps.std_est(np.atleast_2d(out["solved"][0].ravel()), ds=1)[0]
    ndets = np.sum(np.all(~cuts.mask(), axis=-1))
    L.debug(f"\t{map_str.title()} map SNR approximately {snr}")
    if snr < min_snr * np.sqrt(ndets) / 2:
        msg = f"{map_str.title()} map SNR too low."
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        if del_map and filename is not None:
            L.debug("\tDeleting map files")
            glob_path = os.path.splitext(filename)[0] + "*.*"
            flist = glob.glob(glob_path)
            for fname in flist:
                if os.path.isfile(fname):
                    os.remove(fname)
            for name in ["binned", "detweights", "solved", "weights"]:
                set_tag(job, name, "")
        return None, None
    return out, cent


args, cfg = get_args_cfg()

# Setup logger
L = init_log()
metadata.loader.logger = L
cp.logger = L

if args.plot_only:
    L.info("Running in plot_only mode!")

# Get some global settings
source_list = cfg["source_list"] = cfg.get("map_source_list", ["mars", "saturn"])
min_dets = cfg["min_dets"] = cfg.get("min_dets", 50)
min_hits = cfg["min_hits"] = cfg.get("min_hits", 1)
min_det_secs = cfg["min_det_secs"] = cfg.get("min_det_secs", 600)
min_snr = cfg["min_snr"] = cfg.get("min_snr", 5)
n_modes = cfg["n_modes"] = cfg.get("n_modes", 10)
del_map = cfg["del_map"] = cfg.get("del_map", True)
extent = cfg["extent"] = cfg.get("extent", 1800)
zoom = cfg["zoom"] = cfg.get("zoom", 5)
buf = cfg["buf"] = cfg.get("buffer", 30)
log_thresh = cfg["log_thresh"] = cfg.get("log_thresh", 1e-3)
smooth_kern = cfg["smooth_kern"] = cfg.get("smooth_kern", 60)
pointing_type = cfg["pointing_type"] = cfg.get("pointing_type", "pointing_model")
append = cfg["append"] = cfg.get("append", "")
preprocess_cfg = cfg.get("preprocess", None)
cfg_str = yaml.dump(cfg)

if preprocess_cfg is None:
    raise ValueError("Must specify a valid preprocess config!")
with open(preprocess_cfg, "r") as f:
    preprocess_str = yaml.dump(yaml.safe_load(preprocess_cfg))

# Check pointing_type
if pointing_type not in ["pointing_model", "per_obs", "raw"]:
    raise ValueError(f"Invalid pointing_type {pointing_type}")
if pointing_type == "raw" and comps != "T":
    L.info(f"Running with raw pointing, changing comps from {comps} to T")
    comps = "T"

# Setup folders
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "beams/lat")
plot_dir = os.path.join(
    root_dir,
    "plots",
    project_dir,
    f"{pointing_type}{(append!="")*'_'}{append}",
)
data_dir = os.path.join(
    root_dir,
    "data",
    project_dir,
    f"{pointing_type}{(append!="")*'_'}{append}",
)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


# Get context
ctx_path = cfg["context"] = cfg.get(
    "context",
    f"/global/cfs/cdirs/sobs/metadata/lat/contexts/use_this_local.yaml",
)
with open(ctx_path, "r") as f:
    ctx_str = yaml.dump(yaml.safe_load(f))
ctx = Context(ctx_path)
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")

# Setup jobs
start_time = cfg["start_time"]
if args.lookback is not None:
    start_time = time.time() - 3600 * args.lookback
stop_time = cfg["stop_time"]
jdb, all_jobs = setup_jobs(
    comm,
    data_dir,
    "beam_map",
    get_jobdict,
    partial(
        get_jobit,
        obs_ids=args.obs_ids,
        ctx=ctx,
        start_time=start_time,
        stop_time=stop_time,
        source_list=source_list,
        pointing_type=pointing_type,
        L=L,
    ),
    get_jobstr,
    get_tags,
    source_list,
    args.overwrite,
    args.retry_failed,
    args.job_memory,
    args.job_memory_buffer,
    args.plot_only,
    L,
)

# Even things out
joblist = np.array_split(all_jobs, nproc)[myrank].tolist()
n_maps = comm.allgather(len(joblist))
max_maps = np.max(n_maps)
if n_maps[0] != max_maps:
    raise ValueError("Root doesn't have max maps!")
joblist += [None] * (1 + max_maps - len(joblist))

# Get settings for source mask
res = cfg.get("res", (10.0 / 3600.0) * np.pi / 180.0)
pixsize = 3600 * np.rad2deg(res)
mask_size = cfg.get("mask_size", 0.1)
search_mask = cfg.get("search_mask", {"shape": "circle", "xyr": (0, 0, 0.5)})
mask_fac = search_mask["xyr"][-1] / mask_size

# Mapping loop
source_list = set(source_list)
job = None
L.flush()
for i, j in enumerate(joblist):
    sys.stdout.flush()
    comm.barrier()
    # To avoid multiproc issues where the database is locked we lock and unlock serially
    # with jdb.locked(j) as job:
    for r in range(nproc):
        if r == myrank:
            L.flush()
            if job is not None:
                jdb.unlock(job)
            job = None
            if j is not None:
                job = jdb.lock(j.id)
        comm.barrier()
    if job is None:
        continue

    job.mark_visited()
    obs_id = job.tags["obs_id"]
    ufm = job.tags["stream_id"]
    ws = job.tags["wafer_slot"]
    band = job.tags["band"]
    obs = ctx.obsdb.get(obs_id, tags=True)

    if args.plot_only:
        L.normal(f"Replotting {obs_id} {ufm} {band}({i+1}/{n_maps[myrank]})")
        try:
            solved = enmap.read_map(os.path.join(data_dir, job.tags["solved"]))
        except FileNotFoundError:
            msg = "Missing map files in plot_only mode"
            L.error(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
            continue

        obs_plot_dir = os.path.join(
            plot_dir, job.tags["source"], str(obs["timestamp"])[:5], obs_id
        )
        cent = estimate_cent(solved[0], smooth_kern / pixsize, buf)
        make_plots(
            solved, cent, extent, obs_plot_dir, obs_id, ufm, band, zoom, log_thresh
        )
        continue

    L.normal(f"Mapping {obs_id} {ufm} {band}({i+1}/{n_maps[myrank]})")

    # Save metadata and config info
    set_tag(job, "config", cfg_str)
    set_tag(job, "context", ctx_str)
    set_tag(job, "preprocess", preprocess_str)
    set_tag(job, "comps", comps)

    # Get metadata
    lvl = L.level
    L.setLevel(logging.ERROR)
    meta = ctx.get_meta(obs_id)
    L.setLevel(lvl)
    if meta.dets.count == 0:
        msg = "Looks like we don't have real metadata for this observation!"
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        continue
    fscale_fac = 90.0 / float(band[1:])

    src_names = list(source_list & set(obs["tags"]))
    if len(src_names) > 1:
        L.warning("\tObservation tagged for multiple sources!")
    elif len(src_names) == 0:
        msg = "Observation somehow not tagged for any sources in source_list! Skipping!"
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        L.debug(f"\t\tTags were: {obs['tags']}")
        continue
    src_name = "_".join(src_names)
    L.debug(f"\tMapping {src_name}")

    if "hits" in meta.focal_plane:
        meta.restrict("dets", meta.focal_plane.hits >= min_hits)
        if meta.dets.count < min_dets:
            msg = f"Only {meta.dets.count} detectors with good pointing fits!"
            L.error(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
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

    # Load and process the TOD
    aman = load_aman(
        obs["obs_id"],
        preprocess_cfg,
        {"wafer_slot": ws, "wafer.bandpass": band},
        job,
        min_dets,
        L,
        fp_flag=True,
    )
    if aman is None:
        continue

    # Get initial source_flags
    lvl = L.level
    L.setLevel(logging.WARNING)
    source_flags = cp.compute_source_flags(
        tod=aman,
        P=None,
        mask=search_mask,
        center_on=src_to_map,
        res=res,
        max_pix=4e8,
        wrap=None,
    )
    L.setLevel(lvl)

    # Do an aggressive filter and flag dets without the source
    cuts = make_cuts(aman, source_flags, 2 * n_modes, job, L)
    if cuts is None:
        continue

    # Initial map
    out, cent = make_map(
        aman,
        src_to_map,
        res,
        cuts,
        source_flags,
        "T",
        n_modes,
        None,
        min_det_secs * mask_fac * (fscale_fac**2),
        job,
        "initial",
        L,
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
            mask_size * fscale_fac,
        ),
    }
    lvl = L.level
    L.setLevel(logging.WARNING)
    source_flags = cp.compute_source_flags(
        tod=aman,
        P=None,
        mask=mask,
        center_on=src_to_map,
        res=res,
        max_pix=4e8,
        wrap=None,
    )
    L.setLevel(lvl)

    # Make final map
    out, cent = make_map(
        aman,
        src_to_map,
        res,
        cuts,
        source_flags,
        comps,
        n_modes,
        os.path.join(obs_data_dir, "{obs_id}_{ufm}_{band}_{map}.fits"),
        min_det_secs * (fscale_fac**2),
        job,
        "final",
        L,
    )
    if out is None or cent is None:
        continue

    # Add paths to job
    for name, ext in [
        ("binned", "fits"),
        ("detweights", "h5"),
        ("solved", "fits"),
        ("weights", "fits"),
    ]:
        set_tag(
            job,
            name,
            os.path.relpath(
                os.path.join(obs_data_dir, f"{obs_id}_{ufm}_{band}_{name}.{ext}"),
                data_dir,
            ),
        )

    # Plot
    try:
        make_plots(
            out["solved"],
            cent,
            extent,
            obs_plot_dir,
            obs_id,
            ufm,
            band,
            zoom,
            log_thresh,
        )
    except Exception as e:
        L.warning(f"Plotting failed with error: {e}")

    set_tag(job, "message", "Success")
    job.jstate = "done"

L.flush()

# Splits stuff to implement later
# TODO: Bin in annuli
# TODO: Per det maps?
