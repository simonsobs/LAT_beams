import glob
import logging
import os
import sys
import time
from functools import partial

import mpi4py.rc
import numpy as np
import yaml
from pixell import enmap, utils
from so3g.proj import RangesMatrix
from sotodlib import mapmaking, tod_ops
from sotodlib.coords import planets as cp
from sotodlib.core import Context, metadata

from lat_beams.beam_utils import estimate_cent
from lat_beams.plotting import plot_map_complete
from lat_beams.utils import (
    get_args_cfg,
    init_log,
    load_aman,
    log_lvl,
    set_tag,
    setup_jobs,
)

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
    with log_lvl(L, 25):
        if obs_ids is not None:
            obslist = [ctx.obsdb.get(obs_id) for obs_id in obs_ids]
        else:
            src_str = "==1 or ".join(source_list) + "==1"
            obslist = ctx.obsdb.query(
                f"type=='obs' and subtype=='cal' and start_time > {start_time} and stop_time < {stop_time} and ({src_str})",
                tags=source_list,
            )
        if pointing_type != "pointing_model":
            dbs = [
                md["db"]
                for md in ctx["metadata"]
                if "focal_plane" in md.get("name", "")
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
        "ml_map": "",
        "ml_div": "",
        "ml_rhs": "",
        "ml_bin": "",
        "comps": "",
        "source": "",
        "config": "",
        "context": "",
        "preprocess": "",
    }
    return tags


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
    with log_lvl(L, logging.WARNING):
        out = cp.make_map(
            aman.copy(),
            thread_algo="domdir",
            center_on=src_to_map,
            res=res,
            cuts=cuts,
            source_flags=source_flags,
            comps=comps,
            filename=filename,
            n_modes=n_modes,
            info={"obs_id": obs["obs_id"], "ufm": ufm, "band": band},
        )

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
extent = cfg["extent"] = cfg.get("extent", 600)
buf = cfg["buf"] = cfg.get("buffer", 30)
log_thresh = cfg["log_thresh"] = cfg.get("log_thresh", 1e-3)
smooth_kern = cfg["smooth_kern"] = cfg.get("smooth_kern", 60)
pointing_type = cfg["pointing_type"] = cfg.get("pointing_type", "pointing_model")
append = cfg["append"] = cfg.get("append", "")
preprocess_cfg = cfg.get("preprocess", None)
# ds = cfg["ds"] = cfg.get("ds", 5)
cgiters = cfg["cgiters"] = cfg.get("cgiters", 30)
mlpass = cfg["cgpass"] = cfg.get("cgpass", 3)
relcal_range = cfg["relcal_range"] = cfg.get("relcal_range", [0.3, 2])
cfg_str = yaml.dump(cfg)

if preprocess_cfg is None:
    raise ValueError("Must specify a valid preprocess config!")
with open(preprocess_cfg, "r") as f:
    preprocess_cfg = yaml.safe_load(f)
    preprocess_str = yaml.dump(preprocess_cfg)

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

# Modify preproc with our paths
preprocess_cfg["archive"]["index"] = os.path.join(
    data_dir, preprocess_cfg["archive"]["index"]
)
preprocess_cfg["archive"]["policy"]["filename"] = os.path.join(
    data_dir, preprocess_cfg["archive"]["policy"]["filename"]
)
os.makedirs(os.path.dirname(preprocess_cfg["archive"]["index"]), exist_ok=True)
os.makedirs(os.path.dirname(preprocess_cfg["archive"]["index"]), exist_ok=True)

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

# Setup passes
dsstr = "1"
maxiter = str(cgiters)
interpol = "bilinear"
for i in range(1, mlpass):
    interpol = "nearest," + interpol
    maxiter = f"{max(1, max(cgiters//2, cgiters//(i + 1)))}," + maxiter
passes = mapmaking.setup_passes(downsample=dsstr, maxiter=maxiter, interpol=interpol)

# Local comm for ML map
l_comm = comm.Split(myrank, myrank)

# Profiler setup
if args.profile:
    from pyinstrument import Profiler

    profiler = Profiler()
    L.info("Running in profiler mode! Only one job will be run per process")
    joblist = [joblist[0]]
    profiler.start()

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
    sub_id = f"{obs_id}:{ws}:{band}"
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
        posmap = solved.posmap()
        posmap = np.rad2deg(posmap) * 3600
        plot_map_complete(
            solved,
            posmap,
            solved.wcs.wcs.cdelt[1] * (60 * 60),
            extent,
            (posmap[1][cent], posmap[0][cent]),
            os.path.join(obs_plot_dir, ufm),
            f"{obs_id} {ufm} {band}",
            log_thresh=log_thresh,
            lognorm=1.0 / solved[0][cent],
        )
        continue

    L.normal(f"Mapping {obs_id} {ufm} {band}({i+1}/{n_maps[myrank]})")

    # Save metadata and config info
    set_tag(job, "config", cfg_str)
    set_tag(job, "context", ctx_str)
    set_tag(job, "preprocess", preprocess_str)
    set_tag(job, "comps", comps)

    # Get metadata
    with log_lvl(L, logging.ERROR):
        meta = ctx.get_meta(obs_id)
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
        save=(nproc == 1),
    )
    if aman is None:
        continue

    # Relcal cut
    if "relcal" in aman._fields:
        aman.restrict(
            "dets",
            (aman.relcal.relcal >= relcal_range[0])
            * (aman.relcal.relcal <= relcal_range[1]),
        )

    # Get initial source_flags
    with log_lvl(L, logging.WARNING):
        source_flags = cp.compute_source_flags(
            tod=aman,
            P=None,
            mask=search_mask,
            center_on=src_to_map,
            res=res,
            max_pix=4e8,
            wrap=None,
        )

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
    with log_lvl(L, logging.WARNING):
        source_flags = cp.compute_source_flags(
            tod=aman,
            P=None,
            mask=mask,
            center_on=src_to_map,
            res=res,
            max_pix=4e8,
            wrap=None,
        )

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
    os.makedirs(os.path.join(obs_plot_dir, ufm), exist_ok=True)
    try:
        posmap = out["solved"].posmap()
        posmap = np.rad2deg(posmap) * 3600
        plot_map_complete(
            out["solved"],
            posmap,
            out["solved"].wcs.wcs.cdelt[1] * (60 * 60),
            extent,
            (posmap[1][cent], posmap[0][cent]),
            os.path.join(obs_plot_dir, ufm),
            f"{obs_id} {ufm} {band}",
            log_thresh=log_thresh,
            lognorm=1.0 / out["solved"][0][cent],
        )
    except Exception as e:
        L.warning(f"Plotting failed with error: {e}")

    # Now make the ML map
    P = out["P"]
    aman_clean = aman
    utils.deslope(aman_clean.signal, w=5, inplace=True)
    aman_clean.wrap("weather", np.full(1, "typical"))
    aman_clean.wrap("site", np.full(1, "so_lat"))
    mlmap_path = ""
    rhs_path = ""
    div_path = ""
    bin_path = ""
    outmap = None
    for ipass, passinfo in enumerate(passes):
        L.debug(
            "Starting pass %d/%d maxit %d down %d interp %s"
            % (
                ipass + 1,
                len(passes),
                passinfo.maxiter,
                passinfo.downsample,
                passinfo.interpol,
            )
        )
        pass_prefix = os.path.join(
            obs_data_dir, f"{obs_id}_{ufm}_{band}_pass{ipass+1}_"
        )
        noise_model = mapmaking.NmatDetvecs(verbose=True)
        signal_cut = mapmaking.SignalCut(l_comm, dtype=np.float32)
        signal_map = mapmaking.SignalMap(
            out["solved"].shape,
            out["solved"].wcs,
            l_comm,
            comps=comps,
            dtype=np.float64,
            tiled=False,
            interpol=passinfo.interpol,
        )
        signals = [signal_cut, signal_map]
        mapmaker = mapmaking.MLMapmaker(
            signals, noise_model=None, dtype=np.float32, verbose=True
        )

        if passinfo.downsample != 1:
            aman = mapmaking.downsample_obs(aman_clean, passinfo.downsample)
        else:
            aman = aman_clean.copy()
        aman.signal = aman.signal.astype(np.float32)

        # Estimate noise
        if ipass == 0:
            signal_estimate = P.from_map(out["solved"])
        else:
            signal_estimate = eval_prev.evaluate(mapmaker_prev.data[len(mapmaker.data)])
        signal_estimate = mapmaking.resample.resample_fft_simple(
            signal_estimate, aman.samps.count
        )
        mapmaker.add_obs(
            sub_id, aman, noise_model=None, signal_estimate=signal_estimate, pmap=P
        )
        del signal_estimate

        # Write the starting maps
        mapmaker.prepare()
        rhs_path = signal_map.write(
            obs_data_dir + "/", "rhs", signal_map.rhs, unit="pW^-1"
        )
        div_path = signal_map.write(
            obs_data_dir + "/", "div", signal_map.div, unit="pW^-2"
        )
        bin_path = signal_map.write(
            obs_data_dir + "/",
            "bin",
            enmap.map_mul(signal_map.idiv, signal_map.rhs),
            unit="pW",
        )
        L.debug("\tWrote rhs, div, bin")

        # Set up initial condition
        x0 = None  # if ipass == 0 else mapmaker.translate(mapmaker_prev, eval_prev.x_zip)

        # Solve
        t1 = time.time()
        for step in mapmaker.solve(maxiter=passinfo.maxiter, x0=x0):
            t2 = time.time()
            dump = step.i % 10 == 0
            L.debug(
                "\tCG step %4d %15.7e %8.3f %s"
                % (step.i, step.err, t2 - t1, "" if not dump else "(write)")
            )
            if dump:
                for signal, val in zip(signals, step.x):
                    if signal.output:
                        mlmap_path = signal.write(pass_prefix, "map%04d" % step.i, val)
            L.flush()
            t1 = time.time()

        L.debug("Done with ML map")
        for signal, val in zip(signals, step.x):
            if signal.output:
                outmap = val
                mlmap_path = signal.write(pass_prefix, "map", val, unit="pW")

        mapmaker_prev = mapmaker
        eval_prev = mapmaker.evaluator(step.x_zip)

    if mlmap_path == "" or outmap is None:
        msg = "Failed to make ML map"
        L.error(msg)
        set_tag(job, "message", msg)
        job.jstate = "failed"
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
    cent = estimate_cent(outmap[0], smooth_kern / pixsize, buf)
    try:
        posmap = outmap.posmap()
        posmap = np.rad2deg(posmap) * 3600
        plot_map_complete(
            outmap,
            posmap,
            outmap.wcs.wcs.cdelt[1] * (60 * 60),
            extent,
            (posmap[1][cent], posmap[0][cent]),
            os.path.join(obs_plot_dir, ufm),
            f"{obs_id} {ufm} {band} MLmap",
            log_thresh=log_thresh,
            lognorm=1.0 / outmap[0][cent],
        )
    except Exception as e:
        L.warning(f"Plotting failed with error: {e}")

    set_tag(job, "message", "Success")
    job.jstate = "done"

if args.profile:
    profiler.stop()
    profiler.write_html(f"profile_{myrank}.html")

L.flush()

# Splits stuff to implement later
# TODO: Bin in annuli
# TODO: Per det maps?
