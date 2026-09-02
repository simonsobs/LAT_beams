"""
More generic fitting script.
Still somewhat LAT specific but could be genralized if desired.
"""

#### TODO ####
# more plots:
# - fit for each detector [on demand]
# - input priors
# - coverage showing planet trajectory through focal plane [just use/modify the matthew functions]
# - should develop tests for this
# - allow a list of sources

import logging
import os
import re
import sys
import time
from functools import partial, reduce

import h5py
import mpi4py.rc
import numpy as np
import sqlalchemy as sqy
import yaml
from pshmem.locking import MPILock
from scipy.sparse.linalg import svds
from scipy.special import ndtri
from so3g.proj import Ranges, RangesMatrix
from sotodlib import tod_ops
from sotodlib.coords import planets as cp
from sotodlib.core import AxisManager, Context, metadata
from sotodlib.io.metadata import write_dataset
from sotodlib.mapmaking import downsample_obs
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.jobdb import Job
from typing_extensions import cast

from lat_beams.fitting.tod import fit_tod_pointing
from lat_beams.plotting import plot_focal_plane, plot_tod
from lat_beams.utils import (
    get_args_cfg,
    init_log,
    load_aman,
    log_lvl,
    set_tag,
    setup_cfg,
    setup_jobs,
    setup_paths,
)

mpi4py.rc.threads = False
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor, wait

tod_ops.filters.logger.setLevel(logging.ERROR)

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

band_names = {"l": ["f030", "f040"], "m": ["f090", "f150"], "u": ["f220", "f280"]}
ufm_rad_cache = {}

WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[_.-][A-Za-z0-9]+)*")


def get_jobdict(jdb):
    jobdict = {
        f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}": job
        for job in jdb.get_jobs(jclass="fit_pointing")
    }
    return jobdict


def get_jobit(jdb, obs_ids, ctx, start_time, stop_time, source_list, max_dur, logger):
    with log_lvl(logger, 25):
        if obs_ids is not None:
            sub_ids = [obs_id.split(":") for obs_id in obs_ids]
            obslist = [ctx.obsdb.get(obs_id[0]) for obs_id in sub_ids]
            ws_list = [sub_id[1] if len(sub_id) > 1 else None for sub_id in sub_ids]
        else:
            # src_str = "==1 or ".join(source_list) + "==1"
            obslists = [
                ctx.obsdb.query(
                    f"type=='obs' and subtype=='cal' and start_time > {start_time} and stop_time < {stop_time} and duration < {max_dur * 3600}",
                    tags=source_list[:i] + [f"{source}=1"] + source_list[(i + 1) :],
                )
                for i, source in enumerate(source_list)
            ]
            obslist = reduce(lambda q, p: p + q, obslists)
            ws_list = [None] * len(obslist)

        obslist = np.array_split(obslist, nproc)[myrank]
        obsit = []
        for obs, wsl in zip(obslist, ws_list):
            try:
                det_info = ctx.get_det_info(obs["obs_id"])
            except:
                continue
            wsufms = np.unique(
                np.column_stack(
                    [
                        det_info["wafer_slot"],
                        det_info["stream_id"],
                    ]
                ),
                axis=0,
            )
            for (
                ws,
                ufm,
            ) in wsufms:
                if wsl is not None and ws not in wsl:
                    continue
                obsit += [(obs, ws, ufm)]
    return obsit


def get_jobstr(info):
    obs, ws, ufm = info
    job_str = f"{obs['obs_id']}-{ws}-{ufm}"
    return job_str


def get_tags(info):
    obs, ws, ufm = info
    tags = {
        "obs_id": obs["obs_id"],
        "wafer_slot": ws,
        "stream_id": ufm,
        "message": "",
        "source": "",
        "config": "",
        "context": "",
        "preprocess": "",
    }
    return tags


def get_ufm_rad(nominal, ufm):
    if ufm in ufm_rad_cache:
        return ufm_rad_cache[ufm]
    xi0 = np.nanmean(np.array(nominal[ufm]["xi"][:]))
    eta0 = np.nanmean(np.array(nominal[ufm]["eta"][:]))
    r = np.sqrt(
        (np.array(nominal[ufm]["xi"][:]) - xi0) ** 2
        + (np.array(nominal[ufm]["eta"][:]) - eta0) ** 2
    )
    return np.nanmax(r)


def iter_svd(signal, n_modes=5, n_iter=5, n_std=5, positive_src=True):
    data = np.asarray(signal, float).copy()
    src_msk = np.zeros_like(data, dtype=bool)
    n_det, n_t = data.shape
    k = min(n_modes, min(n_det, n_t) - 1)
    x = np.arange(n_t)
    common_mode = data.copy()
    mad_to_sigma = 1 / ndtri(0.75)

    for _ in range(n_iter):
        filled = common_mode.copy()
        for i in range(n_det):
            good = ~src_msk[i]
            if good.sum() > 1:
                r = data[i] - common_mode[i]
                filled[i, ~good] += np.interp(x[~good], x[good], r[good])

        U, S, Vt = svds(filled, k=k)
        order = np.argsort(S)[::-1]
        U, S, Vt = U[:, order], S[order], Vt[order]
        common_mode = (U * S) @ Vt

        r = data - common_mode
        rr = np.where(src_msk, np.nan, r)
        med = np.nanmedian(rr, axis=1, keepdims=True)
        sigma = mad_to_sigma * np.nanmedian(np.abs(rr - med), axis=1, keepdims=True)
        threshold = n_std * np.maximum(sigma, np.finfo(float).eps)
        new_msk = r - med > threshold if positive_src else np.abs(r - med) > threshold
        src_msk = (src_msk + new_msk).astype(bool)

    return common_mode, src_msk


def match_shape(flag, target_shape):
    if isinstance(flag, RangesMatrix) and flag.shape == target_shape:
        return flag
    if isinstance(flag, Ranges) and flag.shape == target_shape:
        return RangesMatrix([flag])

    if isinstance(flag, (Ranges, RangesMatrix)):
        flag = flag.mask()
    try:
        np.broadcast_shapes(flag.shape, target_shape)
        broadcastable = True
    except ValueError:
        broadcastable = False
    if broadcastable:
        flag = np.broadcast_to(flag, target_shape)
        return RangesMatrix.from_mask(flag)
    if flag.ndim == len(target_shape):
        if all(t % f == 0 for f, t in zip(flag.shape, target_shape)):
            repeats = tuple(t // f for f, t in zip(flag.shape, target_shape))
            flag = np.tile(flag, repeats)
            return RangesMatrix.from_mask(flag)

    raise ValueError(f"Cannot match shape {flag.shape} to {target_shape}")


def cent_source_flag(aman, cfg, logger, info):
    source_name, nominal, ufm, res, mask = (
        info["source_name"],
        info["nominal"],
        info["ufm"],
        info["res"],
        info["mask"],
    )
    # See how much of the source we saw...
    # Mask is made massive. This ONLY helps if you have prior knowledge of where source is.
    aman_dummy = aman.restrict("dets", [aman.dets.vals[0]], in_place=False)
    fp = AxisManager(aman_dummy.dets)
    fp.wrap(
        "xi",
        np.zeros(1) + np.nanmean(np.array(nominal[ufm]["xi"][:])),
        [(0, "dets")],
    )
    fp.wrap(
        "eta",
        np.zeros(1) + np.nanmean(np.array(nominal[ufm]["eta"][:])),
        [(0, "dets")],
    )
    fp.wrap(
        "gamma",
        np.zeros(1) + np.nanmean(np.array(nominal[ufm]["gamma"][:])),
        [(0, "dets")],
    )
    aman_dummy.wrap("focal_plane", fp)
    with log_lvl(logger, logging.WARNING):
        source_flags = cp.compute_source_flags(
            tod=aman_dummy,
            P=None,
            mask=mask,
            center_on=source_name,
            res=res * 10,
            max_pix=4e8,
            wrap=None,
        )
    return source_flags


def blind_source_flag(aman, cfg, logger, info):
    flagged = aman.sig_filt > cfg.n_std * aman.std_est
    samp_idx = np.where(np.any(flagged, 0))[0]

    # TODO: Keep the block with the highest sum?
    # Lets kill spurs by only keeping chunks that are mostly continous
    # Spur definition: Glitch leftovers effectively (ie. samples with high signal randomly that are not sources).
    # GLitch + fast jumps finder is NOT run on planet data for lat because sources look like glitches!
    if len(samp_idx) > 2 * cfg.block_size:
        diff_idx = np.diff(samp_idx, prepend=1)
        m = np.r_[False, diff_idx < cfg.block_size // 2, False]
        idx = np.flatnonzero(m[:-1] != m[1:])
        max_idx = (idx[1::2] - idx[::2]).argmax()
        samp_idx = samp_idx[idx[2 * max_idx] : idx[2 * max_idx + 1]]
    flagged = np.zeros(cast(int, aman.samps.count), dtype=bool)
    flagged[samp_idx] = True
    source_flag = Ranges.from_mask(flagged).buffer(block_size)
    return source_flag


def svd_source_flag(aman, cfg, logger, info):
    _ = info
    common_mode, source_flag = iter_svd(
        aman.signal, cfg.svd_modes, cfg.n_std, cfg.std_iters, True
    )
    if cfg.iter_svd_sub:
        aman.signal -= common_mode
    return source_flag


flag_funcs = {
    "cent": cent_source_flag,
    "blind": blind_source_flag,
    "svd": svd_source_flag,
}


def get_source_flag(aman, source_flag_exp, cfg, logger, info, buffer):
    flags = {}
    words = WORD_PATTERN.findall(source_flag_exp)
    if len(words) == 0:
        return np.ones(len(aman.signal), dtype=bool)
    for word in words:
        if word in aman:
            flags[word] = aman[word]
        elif word in flag_funcs:
            flags[word] = flag_funcs[word](aman, cfg, logger, info)
        else:
            raise ValueError(f"Unknown source flag: {word}")
    flags = {
        k: match_shape(v, aman.signal.shape).buffer(buffer // 2)
        for k, v in flags.items()
    }
    expression_for_eval = WORD_PATTERN.sub(
        lambda match: f"values[{match.group()!r}]", source_flag_exp
    )
    return eval(expression_for_eval, {"__builtins__": {}}, {"flags": flags}).buffer(
        buffer // 2
    )


def main():
    # Setup logger
    logger = init_log()
    if logger.extra is None:
        raise ValueError("Logger doesn't have adapter set up!")
    logger.extra = cast(dict, logger.extra)
    metadata.loader.logger = logger
    cp.logger = logger

    # Get settings
    args, cfg_dict = get_args_cfg()
    cfg, cfg_str = setup_cfg(
        args,
        cfg_dict,
        {"fit_source_list": "source_list", "fwhm_tol_pointing": "fwhm_tol"},
        True,
    )

    if args.plot_only:
        logger.info(
            "Running in 'plot_only' mode. TOD plots will be made but pointing will not be fit"
        )

    profiler = None
    if args.profile:
        logger.info("Running in profiler mode! Only a few dets will be kept")

    if cfg.preprocess_cfg is None:
        raise ValueError("Must specify a valid preprocess config!")
    with open(cfg.preprocess_cfg) as f:
        preprocess_cfg = yaml.safe_load(f)
        preprocess_str = yaml.dump(preprocess_cfg)

    if cfg.nominal_fwhm is None:
        raise ValueError("FWHM not found in config file.")

    # Setup folders
    plot_dir, data_dir = setup_paths(cfg.root_dir, "pointing", cfg.tel, "source_fits")
    if myrank == 0:
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

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

    # Output metadata setup
    h5_path = os.path.join(data_dir, "tod_fits.h5")
    h5_file = None
    # Only rank 0 does any writing to file.
    if myrank == 0:
        h5_file = h5py.File(h5_path, "a")
    db_path = os.path.join(data_dir, "db.sqlite")
    if not os.path.isfile(db_path) and myrank == 0:
        scheme = metadata.ManifestScheme()
        scheme.add_exact_match("obs:obs_id")
        scheme.add_exact_match("dets:stream_id")
        scheme.add_data_field("dataset")
        metadata.ManifestDb(scheme=scheme).to_file(db_path)
    db = None
    if myrank == 0:
        db = metadata.ManifestDb(db_path)
    # outdt are columns of output file.
    outdt = [
        ("dets:readout_id", None),
        ("xi", np.float32),
        ("eta", np.float32),
        ("gamma", np.float32),
        ("fwhm", np.float32),
        ("amp", np.float32),
        ("prior_dist", np.float32),
        ("hits", np.int32),
        ("az", np.float32),
        ("el", np.float32),
        ("roll", np.float32),
        ("reduced_chisq", np.float32),
        ("R2", np.float32),
    ]

    # Load nominal pointing [i.e. template pointing from the zemax model
    nominal = h5py.File(cfg.nominal_path)
    # JobDB stuff
    jdb, all_jobs = setup_jobs(
        comm,
        data_dir,
        "fit_pointing",
        get_jobdict,
        partial(
            get_jobit,
            obs_ids=args.obs_ids,
            ctx=ctx,
            start_time=cfg.start_time,
            stop_time=cfg.stop_time,
            source_list=cfg.source_list,
            max_dur=cfg.max_dur,
            logger=logger,
        ),
        get_jobstr,
        get_tags,
        cfg.source_list,
        args.overwrite,
        args.retry_failed,
        args.job_memory,
        args.job_memory_buffer,
        False,
        logger,
    )

    # MPI Splitting
    if (
        args.parallel_factor > nproc
        or args.parallel_factor < 2
        or nproc % args.parallel_factor != 0
    ):
        raise ValueError(
            "Bad parallelization scheme! Should be at least 2, less then or equal to nproc, and evenly divide nproc"
        )
    ismaster = myrank % args.parallel_factor == 0
    mygroup = myrank // args.parallel_factor
    local_comm = comm.Split(mygroup, myrank)
    master_comm = comm.Split(ismaster, myrank)
    P = local_comm.Get_size()
    if ismaster:
        joblist = np.array_split(all_jobs, master_comm.Get_size())[
            master_comm.Get_rank()
        ].tolist()
        n_fits = master_comm.allgather(len(joblist))
        max_fits = np.max(n_fits)
        if n_fits[0] != max_fits:
            raise ValueError("Root doesn't have max fits!")
        if len(joblist) < max_fits:
            joblist += [None] * (max_fits - len(joblist))
    else:
        joblist = []

    # Get settings for source mask
    if args.profile and ismaster:
        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        logger.info("Restricting joblist to just 1 entry per process for profiling!")
        joblist = [joblist[0]]
    to_save = (None, None, None)
    source_list = set(cfg.source_list)

    # Run from the masters
    job = None
    mpilock = MPILock(master_comm)
    with MPICommExecutor(local_comm, 0) as executor:
        if executor is not None:
            joblist += [None]
            for i, j in enumerate(joblist):
                logger.extra["extra"] = ""
                sys.stdout.flush()
                master_comm.barrier()
                to_save = master_comm.gather(to_save, root=0)
                if (
                    myrank == 0
                    and to_save is not None
                    and h5_file is not None
                    and db is not None
                ):
                    for ts in to_save:
                        if ts is None:
                            continue
                        rset, obs_id, ufm = ts
                        if rset is None:
                            continue
                        path = f"{obs_id}/{ufm}"
                        write_dataset(rset, h5_file, path, True)
                        db.add_entry(
                            params={
                                "obs:obs_id": obs_id,
                                "dets:stream_id": ufm,
                                "dataset": path,
                            },
                            filename="tod_fits.h5",
                            replace=True,
                        )
                    h5_file.flush()

                # Just to be safe
                if i % 5 == 0 and i > 0 and h5_file is not None:
                    logger.info("Reloading h5 file to be safe!")
                    h5_file.close()
                    h5_file = h5py.File(h5_path, "a")

                master_comm.barrier()
                # To avoid multiproc issues where the database is locked we lock and unlock serially
                to_save = (None, None, None)
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
                ufm = job.tags["stream_id"]
                ws = job.tags["wafer_slot"]
                logger.extra["extra"] = f" [{obs_id} {ufm} ({i+1}/{len(joblist) - 1})]"
                logger.log(25, "Loading and processing")
                sys.stdout.flush()

                # Save metadata and config info
                set_tag(job, "config", cfg_str)
                set_tag(job, "context", ctx_str)
                set_tag(job, "preprocess", preprocess_str)

                # Get metadata
                with log_lvl(logger, logging.ERROR):
                    obs = ctx.obsdb.get(obs_id, tags=True)
                    try:
                        meta = ctx.get_meta(obs_id)
                    except:
                        meta = None
                if meta is None or meta.dets.count == 0:
                    msg = (
                        f"Looks like we don't have real metadata for this observation!"
                    )
                    logger.error("%s", msg)
                    set_tag(job, "message", msg)
                    job.jstate = cast(sqy.Column[str], jobdb.JState.failed)
                    continue

                # Check source
                src_names = list(source_list & set(obs["tags"]))
                if len(src_names) > 1:
                    logger.warning(
                        "Observation tagged for multiple sources! Only fitting the first"
                    )
                elif len(src_names) == 0:
                    msg = "Observation somehow not tagged for any sources in source_list! Skipping!"
                    logger.error("%s", msg)
                    set_tag(job, "message", msg)
                    job.jstate = cast(sqy.Column[str], jobdb.JState.failed)
                    logger.debug("Tags were: %s", obs["tags"])
                    continue
                source = src_names[0]
                set_tag(job, "source", source)

                wafers = np.unique(
                    [t[3:] for t in obs["tags"] if t[:2] == obs["tube_slot"]]
                    + cfg.forced_ws
                )

                # Generally want to force because you dont know if youre actually scanning the wafer slot you think you are
                if ws not in wafers:
                    msg = "Wafer not targetting or forced to be fit!"
                    logger.error("%s", msg)
                    set_tag(job, "message", msg)
                    job.jstate = cast(sqy.Column[str], jobdb.JState.failed)
                    continue

                if h5_file is not None and myrank == 0 and obs["obs_id"] not in h5_file:
                    h5_file.create_group(obs["obs_id"])

                # Load and process the TOD
                aman = load_aman(
                    obs["obs_id"],
                    preprocess_cfg,
                    {"wafer_slot": ws},
                    job,
                    cfg.min_dets,
                    logger,
                    fp_flag=False,
                    save=(nproc == 1),
                )
                if aman is None:
                    continue
                bp = (aman.det_cal.bg % 4) // 2
                aman = aman.wrap("bp", bp, [(0, "dets")])

                # Downsample
                aman.signal = aman.signal.astype(np.float32)
                aman = downsample_obs(aman, cfg.ds)

                # Filter
                filt = tod_ops.filters.identity_filter()
                if cfg.hp_fc is not None:
                    filt *= tod_ops.filters.high_pass_sine2(cfg.hp_fc)
                if cfg.lp_fc is not None:
                    filt *= tod_ops.filters.low_pass_sine2(cfg.lp_fc)
                sig_filt = tod_ops.filters.fourier_filter(aman, filt)
                aman.wrap("sig_filt", [(0, "dets"), (1, "samps")])

                # Trim edges in case of FFT ringing
                aman = aman.restrict(
                    "samps",
                    slice(cfg.trim_samps + aman.samps.offset, -1 * cfg.trim_samps),
                )

                # Source flags
                source_name = source
                # This is just to account for gap in sotodlib.
                # TODO: Add it to sotodlib...
                if source == "rcw38":
                    source_name = "J134.78-47.509"
                elif source == "taua":
                    source_name = ("tauA", 83.6272579, 22.02159891)
                elif source == "3c279":
                    source_name = "J194.0409868m5.79174024"

                std_est = tod_ops.jumps.std_est(aman.sig_filt, ds=1)
                aman.wrap("std_est", std_est, [(0, "dets")])
                if cfg.source_flag_exp != "" and (
                    cfg.src_msk
                    or cfg.filter_for_sources
                    or (cfg.iter_svd_sub and "svd" in cfg.source_flag_exp)
                ):
                    to_skip = False
                    (
                        info["source_name"],
                        info["nominal"],
                        info["ufm"],
                        info["res"],
                        info["mask"],
                    ) = (source_name, nominal, ufm, res, mask)
                    source_flag = get_source_flag(
                        aman, source_flag_exp, cfg, logger, info, cfg.block_size
                    )
                    if cfg.filter_for_sources:
                        if cfg.cvd_modes <= 0:
                            logger.warning(
                                "filter_for_sources is True but svd_modes is <=0 so not running!"
                            )
                        for b in np.unique(aman.bp):
                            bmsk = aman.bp == b
                            aman.signal[b] = cp.filter_for_sources(
                                tod=None,
                                signal=aman.signal,
                                source_flags=source_flag,
                                n_modes=cfg.svd_modes,
                            )

                    if cfg.src_msk:
                        src_msk = source_flag.mask()
                        start, stop = np.percentile(
                            np.where(np.any(src_msk, 0))[0], [5, 95]
                        )
                        start = np.max(start - 3 * cfg.block_size, 0)
                        stop = np.min(
                            stop + 3 * cfg.block_size, cast(int, aman.samps.count)
                        )
                        det_msk = np.sum(src_msk, axis=1) < cfg.min_samps / 2
                        msg = ""
                        if np.sum(src_msk) == 0:
                            if not args.plot_only:
                                msg = "No samples flagged in source flags!"
                                to_skip = True
                            else:
                                logger.warning(
                                    "No samples flagged! But running in plot_only mode so will continue with all samples"
                                )
                                start = 0
                                stop = int(cast(int, aman.samps.count))
                        if stop - start < cfg.min_samps:
                            if not args.plot_only:
                                msg = f"Too few samples flagged in source flags! {start} to {stop}"
                                to_skip = True
                            else:
                                logger.debug(
                                    "Only %s flagged samples! But running in plot_only mode so will continue",
                                    stop - start,
                                )
                        if to_skip:
                            logger.error("%s", msg)
                            set_tag(job, "message", msg)
                            job.jstate = cast(sqy.Column[str], jobdb.JState.failed)
                            continue
                        logger.debug(
                            "%s samps flagged in the source range", stop - start
                        )
                        logger.debug("%s dets after source_flags", np.sum(det_msk))
                        aman = aman.restrict(
                            "samps",
                            slice(
                                start + cast(int, aman.samps.offset),
                                stop + cast(int, aman.samps.offset),
                            ),
                        )
                        aman = aman.restrict("dets", det_msk)

                # Setup plot dirs
                tod_plot_dir = os.path.join(
                    plot_dir, source, "tods", str(obs["timestamp"])[:5], obs["obs_id"]
                )
                fit_plot_dir = os.path.join(
                    plot_dir, source, "fits", str(obs["timestamp"])[:5], obs["obs_id"]
                )
                os.makedirs(tod_plot_dir, exist_ok=True)
                os.makedirs(fit_plot_dir, exist_ok=True)

                # Now loop by band
                # We do this because noise properties and source responce will be band dependant
                aman_full = aman
                # TODO: This variable name needs to be updated to something more global. Also should check if there is better way than grabbing a hard coded character in string.
                # This just extracts if it's m or u for mf or uhf
                tube_band = ufm[4]
                outdt[0] = ("dets:readout_id", np.array(aman_full.dets.vals).dtype)
                rsets = []
                msg = ""
                for band in np.unique(aman_full.bp):
                    if msg != "":
                        msg += " "
                    band_name = band_names[tube_band][band]
                    logger.extra["extra"] = (
                        f" [{obs_id} {ufm} {band_name} ({i+1}/{len(joblist) - 1})]"
                    )
                    logger.log(25, "Fitting")
                    aman = aman_full.restrict("dets", bp == band, in_place=False)
                    logger.log(25, "%s detectors in band", aman.dets.count)

                    # Kill dets with really high noise
                    thresh = cfg.n_med * np.median(aman.std_est[aman.std_est > 0])
                    aman.restrict("dets", aman.std_est < thresh)
                    if aman.dets.count < cfg.min_dets:
                        _msg = f"{band_name} Noise too high."
                        logger.error("%s", _msg)
                        msg += _msg
                        continue
                    logger.log(25, "%s detectors after noise cuts", aman.dets.count)

                    # Get median std of all dets after cuts
                    std_all = np.median(
                        aman.std_est[(aman.std_est < thresh) * (aman.std_est > 0)]
                    )

                    # Make a p2p cut
                    # Do some final cuts to kill dets that didn't see the source
                    ptp = np.ptp(aman.sig_filt, axis=-1)
                    std = np.std(aman.sig_filt, axis=-1)
                    thresh = 0.01 * np.percentile(ptp, 90)
                    msk = (ptp > thresh) * (std > 0)
                    aman = aman.restrict("dets", msk)
                    if aman.dets.count < cfg.min_dets:
                        _msg = (
                            f"{band_name} Too few detectors after final sanity check."
                        )
                        logger.error("%s", _msg)
                        msg += _msg
                        continue
                    logger.log(25, "%s detectors after ptp ccuts", aman.dets.count)

                    # Plot the TOD
                    plot_tod(
                        aman,
                        aman.sig_filt,
                        tod_plot_dir,
                        f"{ufm}_{band_name}",
                        cfg.min_dets * 10,
                    )
                    if args.plot_only:
                        _msg = f"{band_name} Ran in no fit mode"
                        logger.log(25, "%s", _msg)
                        msg += _msg
                        continue

                    # Make the fft a fast length (ie like a prime number)
                    _ = tod_ops.filters.fft_trim(aman, prefer="center")
                    if aman.dets.count > 10 and args.profile:
                        logger.log(25, "Restricting to 10 dets for profile")
                        aman.restrict("dets", aman.dets.vals[:10])

                    # Before sending via MPI lets remove anything we don't need from the aman
                    fields = list(aman._fields.keys())
                    for field in fields:
                        if field not in ["signal", "timestamps", "boresight"]:
                            aman.move(field, None)

                    # Now submit to the workers
                    logger.log(25, "Attempting to fit %s detectors", aman.dets.count)
                    t0 = time.time()
                    det_splits = np.array_split(aman.dets.vals, P)
                    fp_futures = [
                        executor.submit(
                            fit_tod_pointing,
                            aman.restrict("dets", det_splits[d], in_place=False),
                            (cfg.hp_fc, cfg.lp_fc),
                            fwhm=np.deg2rad(cfg.nominal_fwhm[band_name] / 60.0),
                            source=source_name,
                            **cfg.fit_pars,
                        )
                        for d in range(1, P)
                        if len(det_splits[d]) > 0
                    ]
                    fp0 = fit_tod_pointing(
                        aman.restrict("dets", det_splits[0], in_place=False),
                        (cfg.hp_fc, cfg.lp_fc),
                        fwhm=np.deg2rad(cfg.nominal_fwhm[band_name] / 60.0),
                        source=source_name,
                        **cfg.fit_pars,
                    )
                    wait(fp_futures)
                    fps = [fp0] + [fp_future.result() for fp_future in fp_futures]
                    t1 = time.time()
                    logger.log(25, "Took %s seconds to fit", t1 - t0)
                    for focal_plane in fps:
                        # Do a quick cut based on FWHM tol
                        msk = (
                            np.abs(
                                1
                                - focal_plane.fwhm
                                / np.deg2rad(cfg.nominal_fwhm[band_name] / 60)
                            )
                            < cfg.fwhm_tol
                        )
                        focal_plane = focal_plane.restrict("dets", msk)

                        # Convert to results set
                        sarray = np.fromiter(
                            zip(
                                np.array(focal_plane.dets.vals),
                                np.array(focal_plane.xi, dtype=np.float32),
                                np.array(focal_plane.eta, dtype=np.float32),
                                np.array(focal_plane.gamma, dtype=np.float32),
                                np.array(focal_plane.fwhm, dtype=np.float32),
                                np.array(focal_plane.amp, dtype=np.float32),
                                np.array(focal_plane.dist, dtype=np.float32),
                                np.array(focal_plane.hits, dtype=np.int32),
                                np.array(focal_plane.az, dtype=np.float32),
                                np.array(focal_plane.el, dtype=np.float32),
                                np.array(focal_plane.roll, dtype=np.float32),
                                np.array(focal_plane.reduced_chisq, dtype=np.float32),
                                np.array(focal_plane.R2, dtype=np.float32),
                            ),
                            dtype=outdt,
                            count=cast(int, focal_plane.dets.count),
                        )
                        rsets += [metadata.ResultSet.from_friend(sarray)]
                    _msg = f"{band_name} Success!"
                    logger.log(25, "%s", _msg)
                    msg += msg

                logger.extra["extra"] = f" [{obs_id} {ufm} ({i+1}/{len(joblist) - 1})]"
                # Get ready to save
                if args.plot_only:
                    to_save = (None, None, None)
                    continue
                if len(rsets) == 0:
                    to_save = (None, None, None)
                    set_tag(job, "message", msg)
                    job.jstate = cast(sqy.Column[str], jobdb.JState.failed)
                    continue

                # Combine rsets
                rset = reduce(lambda q, p: p + q, rsets)
                if len(rset) == 0:
                    to_save = (None, None, None)
                    if msg != "":
                        msg += " "
                    _msg = "ResultSet empty somehow!"
                    logger.error("%s", _msg)
                    msg += _msg
                    set_tag(job, "message", msg)
                    job.jstate = cast(sqy.Column[str], jobdb.JState.failed)
                    continue

                # Kill bad fits
                focal_plane = rset.to_axismanager(axis_key="dets:readout_id")
                # Source should be positive in pW
                msk = np.array(focal_plane.amp) > 0
                # Kill fits that are statistically bad
                msk *= np.array(focal_plane.reduced_chisq < cfg.max_chisq)
                # How many times it saw the source (ie. hits).
                msk *= np.array(focal_plane.hits) >= cfg.min_hits
                high_hits = np.array(focal_plane.hits) >= cfg.high_hits
                if np.sum(msk * high_hits) >= cfg.min_dets:
                    med_xi = np.median(np.array(focal_plane.xi[msk * high_hits]))
                    med_eta = np.median(np.array(focal_plane.eta[msk * high_hits]))
                    msk *= np.sqrt(
                        (np.array(focal_plane.xi) - med_xi) ** 2
                        + np.abs(np.array(focal_plane.eta) - med_eta) ** 2
                    ) < 1.5 * get_ufm_rad(nominal, ufm)

                # Instead of cutting the rset we set R2 to 0
                # This is because det match does not like missing dets
                rset = rset.asarray()
                rset[~msk]["R2"] = 0.0
                rset = metadata.ResultSet.from_friend(rset)
                focal_plane.restrict(
                    "dets", msk * (rset["R2"] > cfg.min_R2)
                )  # Only used for plotting

                if len(rset) == 0 or np.sum(msk) < cfg.min_dets:
                    to_save = (None, None, None)
                    if msg != "":
                        msg += " "
                    _msg = "Too many bad fits!"
                    logger.error("%s", _msg)
                    msg += _msg
                    set_tag(job, "message", msg)
                    job.jstate = cast(sqy.Column[str], jobdb.JState.failed)
                    continue
                # Plot focal plane, encoders, and a histrogram of fhwp, amp, hits
                # TODO: Split by band?
                plot_focal_plane(focal_plane, fit_plot_dir, ufm, obs_id)

                # Ready to save
                logger.log(25, "Saving %s fits (%s good).", len(rset), np.sum(msk))
                if cfg.pad:
                    with log_lvl(logger, logging.ERROR):
                        all_dets = ctx.get_det_info(
                            obs["obs_id"], dets={"stream_id": ufm}
                        )["readout_id"]
                    pad_dets = all_dets[~np.isin(all_dets, rset["dets:readout_id"])]
                    if outdt[0][1] is None:
                        outdt[0] = (outdt[0][0], pad_dets.dtype)
                    pad_res = np.zeros(len(pad_dets), dtype=outdt)
                    pad_res["dets:readout_id"] = pad_dets
                    for field, dtype in outdt:
                        if np.issubdtype(dtype, np.floating):
                            pad_res[field][:] = np.nan
                    rset = rset + metadata.ResultSet.from_friend(pad_res)
                to_save = (rset, obs_id, ufm)

                if args.profile:
                    to_save = (None, None, None)
                    msg = "Ran profile"
                    logger.info("%s", msg)
                    set_tag(job, "message", msg)
                    job.jstate = cast(sqy.Column[str], jobdb.JState.open)
                    continue
                job.jstate = cast(sqy.Column[str], jobdb.JState.done)

    if h5_file is not None:
        h5_file.close()
    nominal.close()

    if args.profile and ismaster and profiler is not None:
        profiler.stop()
        profiler.write_html(f"profile_{myrank}.html")


if __name__ == "__main__":
    main()
