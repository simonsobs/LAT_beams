"""
More generic fitting script.
Still somewhat LAT specific but could be genralized if desired.
"""

#### TODO ####
# More functions
# Switch to logger
# Time domain fitting of single source [just jup notebook is ok]
# more plots:
# - fit for each detector [on demand]
# - input priors
# - coverage showing planet trajectory through focal plane [just use/modify the matthew functions]
# - should develop tests for this
# - allow a list of sources

import logging
import os
import sys
import time
from functools import partial, reduce

import h5py
import mpi4py.rc
import numpy as np
import yaml
from pshmem.locking import MPILock
from sotodlib import tod_ops
from sotodlib.coords import planets as cp
from sotodlib.core import AxisManager, Context, metadata
from sotodlib.io.metadata import write_dataset
from sotodlib.mapmaking import downsample_obs
from sotodlib.site_pipeline.jobdb import Job
from typing_extensions import cast

from lat_beams.fitting import fit_tod_pointing
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


def get_jobdict(jdb):
    jobdict = {
        f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}": job
        for job in jdb.get_jobs(jclass="fit_pointing")
    }
    return jobdict


def get_jobit(jdb, obs_ids, ctx, start_time, stop_time, source_list, max_dur, logger):
    with log_lvl(logger, 25):
        if obs_ids is not None:
            obslist = [ctx.obsdb.get(obs_id) for obs_id in obs_ids]
        else:
            src_str = "==1 or ".join(source_list) + "==1"
            obslist = ctx.obsdb.query(
                f"type=='obs' and subtype=='cal' and start_time > {start_time} and stop_time < {stop_time} and duration < {max_dur * 3600} and ({src_str})",
                tags=source_list,
            )

        obslist = np.array_split(obslist, nproc)[myrank]
        obsit = []
        for obs in obslist:
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


def src_flag_cut(source_name, aman, nominal, ufm, res, mask, logger):
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
    if len(source_flags.ranges[0].ranges()) == 0:
        start = -1
        stop = -1
    else:
        start = source_flags.ranges[0].ranges()[0][0]
        stop = source_flags.ranges[0].ranges()[-1][-1]
    return start, stop


def main():
    # Setup logger
    logger = init_log()
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
        from pyinstrument import Profiler

        profiler = Profiler()
        logger.info("Running in profiler mode! Only a few dets will be kept")

    if cfg.preprocess_cfg is None:
        raise ValueError("Must specify a valid preprocess config!")
    with open(cfg.preprocess_cfg, "r") as f:
        preprocess_cfg = yaml.safe_load(f)
        preprocess_str = yaml.dump(preprocess_cfg)

    if cfg.nominal_fwhm is None:
        raise ValueError("FWHM not found in config file.")

    # Setup folders
    plot_dir, data_dir = setup_paths(cfg.root_dir, "pointing", cfg.tel, "source_fits")
    if myrank == 0:
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
    with open(cfg.ctx_path, "r") as f:
        ctx_str = yaml.dump(yaml.safe_load(f))
    ctx = Context(cfg.ctx_path)
    if ctx.obsdb is None:
        raise ValueError("No obsdb in context!")

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
    if args.profile and ismaster and profiler is not None:
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
            for i, j in enumerate(joblist):
                sys.stdout.flush()
                master_comm.barrier()
                to_save = master_comm.gather(to_save, root=0)
                if myrank == 0 and to_save is not None and h5_file is not None:
                    for ts in to_save:
                        if ts is None:
                            continue
                        rset, obs_id, ufm = ts
                        if rset is None:
                            continue
                        path = f"{obs['obs_id']}/{ufm}"
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
                logger.flush()
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
                logger.normal("Fitting %s %s (%s/%s)", obs_id, ufm, i + 1, len(joblist))
                sys.stdout.flush()

                # Save metadata and config info
                set_tag(job, "config", cfg_str)
                set_tag(job, "context", ctx_str)
                set_tag(job, "preprocess", preprocess_str)

                # Get metadata
                with log_lvl(logger, logging.ERROR):
                    obs = ctx.obsdb.get(obs_id, tags=True)
                    meta = ctx.get_meta(obs_id)
                if meta.dets.count == 0:
                    msg = "Looks like we don't have real metadata for this observation!"
                    logger.error("\\t%s", msg)
                    set_tag(job, "message", msg)
                    job.jstate = "failed"
                    continue

                # Check source
                src_names = list(source_list & set(obs["tags"]))
                if len(src_names) > 1:
                    logger.warning(
                        "\tObservation tagged for multiple sources! Only fitting the first"
                    )
                elif len(src_names) == 0:
                    msg = "Observation somehow not tagged for any sources in source_list! Skipping!"
                    logger.error("\\t%s", msg)
                    set_tag(job, "message", msg)
                    job.jstate = "failed"
                    logger.debug("\\t\\tTags were: %s", obs["tags"])
                    continue
                source = src_names[0]
                set_tag(job, "source", source)

                # TODO: Make sure to make this tagging work for sat format too
                wafers = np.unique(
                    [t[3:] for t in obs["tags"] if t[:2] == obs["tube_slot"]]
                    + cfg.forced_ws
                )

                # Generally want to force because you dont know if youre actually scanning the wafer slot you think you are
                # Less relevant for SATs but true for LAT. Will need to play around to see when this is truly necessary for SATs.
                if ws not in wafers:
                    msg = "Wafer not targetting or forced to be fit!"
                    logger.error("\\t%s", msg)
                    set_tag(job, "message", msg)
                    job.jstate = "failed"
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

                # Downsample
                aman.signal = aman.signal.astype(np.float32)
                aman = downsample_obs(aman, cfg.ds)

                # Source flags
                source_name = source
                # This is just to account for gap in sotodlib.
                # TODO: Add it to sotodlib...
                if source == "rcw38":
                    source_name = "J134.78-47.509"

                if cfg.src_msk:
                    logger.debug("\tRunning source flags")
                    start, stop = src_flag_cut(
                        source_name,
                        aman,
                        nominal,
                        ufm,
                        cfg.res,
                        cfg.pointing_mask,
                        logger,
                    )
                    msg = ""
                    if start < 0 or stop < 0:
                        if not args.plot_only:
                            msg = "No samples flagged in source flags!"
                            to_skip = True
                        else:
                            logger.warning(
                                "\\t\\tNo samples flagged! But running in plot_only mode so will continue with all samples"
                            )
                            start = 0
                            stop = int(cast(int, aman.samps.count))
                    if stop - start < cfg.min_samps:
                        if not args.plot_only:
                            msg = "Too few samples flagged in source flags!"
                            to_skip = True
                        else:
                            logger.debug(
                                "\\t\\tOnly %s flagged samples! But running in plot_only mode so will continue",
                                stop - start,
                            )
                    if to_skip:
                        logger.error("\\t\\t%s", msg)
                        set_tag(job, "message", msg)
                        job.jstate = "failed"
                        continue
                    logger.debug(
                        "\\t\\t%s samps flagged in the source range", stop - start
                    )
                    aman = aman.restrict(
                        "samps",
                        slice(
                            start + cast(int, aman.samps.offset),
                            stop + cast(int, aman.samps.offset),
                        ),
                    )

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
                rsets = []
                aman_full = aman
                bp = (aman_full.det_cal.bg % 4) // 2
                # TODO: This variable name needs to be updated to something more global. Also should check if there is better way than grabbing a hard coded character in string.
                # This just extracts if it's m or u for mf or uhf
                tube_band = ufm[4]
                outdt[0] = ("dets:readout_id", np.array(aman_full.dets.vals).dtype)
                rsets = []
                msg = ""
                for band in np.unique(bp):
                    if msg != "":
                        msg += " "
                    band_name = band_names[tube_band][band]
                    logger.normal("\\tFitting %s", band_name)
                    aman = aman_full.restrict("dets", bp == band, in_place=False)

                    # Filter
                    filt = tod_ops.filters.high_pass_butter4(cfg.hp_fc * 2)
                    sig_filt = tod_ops.filters.fourier_filter(aman, filt)

                    # Trim edges in case of FFT ringing
                    aman = aman.restrict(
                        "samps",
                        slice(cfg.trim_samps + aman.samps.offset, -1 * cfg.trim_samps),
                    )
                    sig_filt = sig_filt[:, cfg.trim_samps : (-1 * cfg.trim_samps)]

                    # Kill dets with really high noise
                    std = np.std(sig_filt, axis=-1)
                    # Threshold determined from config file n_med
                    # These numbers are true for LAT; unclear for SAT.
                    # TODO: Make histograms of std and how much is getting cut as a part of diagnostic outputs [from talking to Matthew]. Otherwise this is a good way to cut outliers.
                    # Since boxy instead of gaussians, so quantiles are cool. Pseudo sigma is e.g. ~33%-50% [median based thing so that outliers are not included]. Then can do e.g. a 4sigma cut or whatever (but shouldnt have a high percentage of outliers).
                    thresh = cfg.n_med * np.median(std[std > 0])
                    aman.restrict("dets", std < thresh)
                    sig_filt = sig_filt[std < thresh]
                    if aman.dets.count < cfg.min_dets:
                        _msg = f"{band_name} Noise too high."
                        logger.error(_msg)
                        msg += _msg
                        continue

                    # Get median std of all dets after cuts
                    std_all = np.median(std[(std < thresh) * (std > 0)])

                    # Lets try to find the source blind
                    # Check for places where signal is high compared to noise (ie how many times the std)
                    # Might not work for SAT if SNR is too low. CHECK THIS.
                    if cfg.blind_search:
                        flagged = sig_filt > cfg.n_std * std_all
                        samp_idx = np.where(np.any(flagged, 0))[0]

                        # TODO: Keep the block with the highest sum?
                        # Lets kill spurs by only keeping chunks that are mostly continous
                        # Spur definition: Glitch leftovers effectively (ie. samples with high signal randomly that are not sources).
                        # GLitch + fast jumps finder is NOT run on planet data for lat because sources look like glitches!
                        if len(samp_idx) > 2 * cfg.block_size:
                            diff_idx = np.diff(samp_idx, prepend=1)
                            m = np.r_[False, diff_idx < cfg.block_size, False]
                            idx = np.flatnonzero(m[:-1] != m[1:])
                            max_idx = (idx[1::2] - idx[::2]).argmax()
                            samp_idx = samp_idx[idx[2 * max_idx] : idx[2 * max_idx + 1]]
                            logger.debug(
                                "\\t\\tFound %s continously flagged samples",
                                len(samp_idx),
                            )

                        # Not enough samples flagged => won't bother fitting
                        if len(samp_idx) < min(cfg.block_size, cfg.min_samps / 2):
                            if args.plot_only:
                                logger.warning(
                                    "\t\tLooks like you didn't see the source at all! But running in plot_only mode so will continue"
                                )
                            else:
                                _msg = f"{band_name} Failed to find source blind."
                                logger.error(_msg)
                                msg += msg
                                continue
                        start = int(
                            max(0, np.percentile(samp_idx, 10) - (cfg.block_size * 5))
                        )
                        stop = int(
                            min(
                                cast(int, aman.samps.count),
                                np.percentile(samp_idx, 90) + (cfg.block_size * 5),
                            )
                        )
                        if stop - start < cfg.min_samps:
                            if not args.plot_only:
                                _msg = f"{band_name} Too few samples found in blind flagging."
                                logger.error(_msg)
                                msg += _msg
                                continue
                            logger.warning(
                                "\\t\\tOnly %s flagged samples! But running in plot_only mode so will continue",
                                stop - start,
                            )
                        logger.debug("\\t\\t%s samps flagged blind", stop - start)
                        # Restricting to samples where we think we see a source now. The block above this is probs hardest to generalize between LAT and SAT
                        aman = aman.restrict(
                            "samps",
                            slice(
                                start + cast(int, aman.samps.offset),
                                stop + cast(int, aman.samps.offset),
                            ),
                        )
                        sig_filt = sig_filt[:, start:stop]

                    # Make a p2p cut
                    # TODO: This cut might be different for SAT and LAT.
                    # Do some final cuts to kill dets that didn't see the source
                    ptp = np.ptp(sig_filt, axis=-1)
                    std = np.std(sig_filt, axis=-1)
                    thresh = 0.1 * np.percentile(ptp, 90)
                    msk = (ptp > thresh) * (ptp > cfg.n_std * std) * (std > 0)
                    aman = aman.restrict("dets", msk)
                    sig_filt = sig_filt[msk]
                    if aman.dets.count < cfg.min_dets:
                        _msg = (
                            f"{band_name} Too few detectors after final sanity check."
                        )
                        logger.error("\\t%s", _msg)
                        msg += _msg
                        continue

                    logger.normal(
                        "\\t\\tAttempting to fit %s detectors", aman.dets.count
                    )

                    # Plot the TOD
                    plot_tod(aman, sig_filt, tod_plot_dir, f"{ufm}_{band_name}")
                    if args.plot_only:
                        _msg = f"{band_name} Ran in no fit mode"
                        logger.normal(_msg)
                        msg += _msg
                        continue

                    # Make the fft a fast length (ie like a prime number)
                    _ = tod_ops.filters.fft_trim(aman, prefer="center")
                    if aman.dets.count > 10 and args.profile:
                        logger.normal("\tRestricting to 10 dets for profile")
                        aman.restrict("dets", aman.dets.vals[:10])

                    # Before sending via MPI lets remove anything we don't need from the aman
                    fields = list(aman._fields.keys())
                    for field in fields:
                        if field not in ["signal", "timestamps", "boresight"]:
                            aman.move(field, None)

                    # Now submit to the workers
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
                    logger.normal("\\t\\tTook %s seconds to fit", t1 - t0)
                    for focal_plane in fps:
                        # Do a quick cut based on FWHM tol
                        focal_plane.restrict(
                            "dets",
                            np.abs(
                                1
                                - focal_plane.fwhm
                                / np.deg2rad(cfg.nominal_fwhm[band_name] / 60)
                            )
                            < cfg.fwhm_tol,
                        )

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
                            count=focal_plane.dets.count,
                        )
                        rsets += [metadata.ResultSet.from_friend(sarray)]
                    _msg = f"{band_name} Success!"
                    logger.normal("\\t%s", _msg)
                    msg += msg

                # Get ready to save
                if args.plot_only:
                    to_save = (None, None, None)
                    continue
                if len(rsets) == 0:
                    to_save = (None, None, None)
                    set_tag(job, "message", msg)
                    job.jstate = "failed"
                    continue

                # Combine rsets
                rset = reduce(lambda q, p: p + q, rsets)
                if len(rset) == 0:
                    to_save = (None, None, None)
                    if msg != "":
                        msg += " "
                    _msg = "ResultSet empty somehow!"
                    logger.error(_msg)
                    msg += _msg
                    set_tag(job, "message", msg)
                    job.jstate = "failed"
                    continue

                # Kill bad fits
                focal_plane = rset.to_axismanager(axis_key="dets:readout_id")
                # Source should be positive in pW
                msk = np.array(focal_plane.amp) > 0
                # Kill fits that are statistically bad
                print(np.sum(msk))
                sys.stdout.flush()
                msk *= np.array(focal_plane.reduced_chisq < cfg.max_chisq)
                print(np.sum(msk))
                sys.stdout.flush()
                med_xi = np.median(np.array(focal_plane.xi[msk]))
                med_eta = np.median(np.array(focal_plane.eta[msk]))
                msk *= (
                    np.sqrt(
                        (np.array(focal_plane.xi) - med_xi) ** 2
                        + np.abs(np.array(focal_plane.eta) - med_eta) ** 2
                    )
                    < cfg.ufm_rad
                )
                print(np.sum(msk))
                sys.stdout.flush()
                msk *= np.array(focal_plane.amp) < cfg.n_med * np.median(
                    np.array(focal_plane.amp[msk])
                )
                print(np.sum(msk))
                sys.stdout.flush()
                msk *= (
                    np.array(focal_plane.amp)
                    > np.median(np.array(focal_plane.amp[msk])) / cfg.n_med**2
                )
                print(np.sum(msk))
                sys.stdout.flush()
                # How many times it saw the source (ie. hits).
                msk *= np.array(focal_plane.hits) >= cfg.min_hits
                print(np.sum(msk))
                sys.stdout.flush()

                # Instead of cutting the rset we set R2 to 0
                # This is because det match does not like missing dets
                rset = rset.asarray()
                rset[~msk]["R2"] = 0.0
                rset = metadata.ResultSet.from_friend(rset)
                focal_plane.restrict("dets", msk)  # Only used for plotting

                if len(rset) == 0 or np.sum(msk) < cfg.min_dets:
                    to_save = (None, None, None)
                    if msg != "":
                        msg += " "
                    _msg = "Too many bad fits!"
                    logger.error(_msg)
                    msg += _msg
                    set_tag(job, "message", msg)
                    job.jstate = "failed"
                    continue
                # Plot focal plane, encoders, and a histrogram of fhwp, amp, hits
                # TODO: Split by band?
                plot_focal_plane(focal_plane, fit_plot_dir, ufm)

                # Ready to save
                logger.normal("\\tSaving %s fits (%s good).", len(rset), np.sum(msk))
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
                    rset = metadata.ResultSet.from_friend(pad_res)
                to_save = (rset, obs_id, ufm)

                if args.profile:
                    to_save = (None, None, None)
                    msg = "Ran profile"
                    logger.info(msg)
                    set_tag(job, "message", msg)
                    job.jstate = "open"
                    continue
                job.jstate = "done"

    if h5_file is not None:
        h5_file.close()
    nominal.close()

    if args.profile and ismaster and profiler is None:
        profiler.stop()
        profiler.write_html(f"profile_{myrank}.html")
    logger.flush()


if __name__ == "__main__":
    main()
