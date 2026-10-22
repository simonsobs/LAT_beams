import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import cast

import astropy.units as u
import h5py
import latcom.utils.abscal_utils as au
import matplotlib.pyplot as plt
import numpy as np
import sqlalchemy as sqy
import yaml
from healpy.sphtfunc import beam2bl
from latcom.utils.common import date_str, save_pickle
from mpi4py import MPI
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from sotodlib.core import AxisManager, Context, metadata
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.jobdb import Job

from lat_beams import beam_utils as bu
from lat_beams.utils import (
    ErrCode,
    fail,
    get_args_cfg,
    init_log,
    log_lvl,
    make_jobdb,
    set_tag,
    setup_cfg,
    setup_jobs,
    setup_paths,
    update_jobs_retry,
)

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

max_workers = int(os.environ.get("NUM_FUTURES", os.environ.get("OMP_NUM_THREADS", 8)))


def get_jobdict(jdb):
    return {
        f"{job.tags['split']}-{job.tags['split_str']}-"
        f"{job.tags['det_split']}-{job.tags['epoch_start']}-"
        f"{job.tags['epoch_end']}": job
        for job in jdb.get_jobs(jclass="make_abscal")
    }


def get_jobit(jdb, cfg, stack_jobs, det_splits):
    _ = jdb
    jobit = []
    epochs = np.array(cfg.epochs, dtype=float)
    if myrank == 0:
        for sjob in stack_jobs:
            split = sjob.tags["split"]
            if split not in cfg.split_by:
                continue
            spl = sjob.tags["split_str"]
            det_split = sjob.tags["det_split"]
            if det_split not in det_splits:
                continue
            epoch = np.array(
                [sjob.tags["epoch_start"], sjob.tags["epoch_end"]], dtype=float
            )
            if not np.any(epochs == epoch):
                continue
            jobit += [
                (
                    split,
                    spl,
                    det_split,
                    sjob.tags["epoch_start"],
                    sjob.tags["epoch_end"],
                )
            ]
    return jobit


def get_jobstr(info):
    if isinstance(info, Job):
        return f"{info.tags['split']}-{info.tags['split_str']}-{info.tags['det_split']}-{info.tags['epoch_start']}-{info.tags['epoch_end']}"
    return f"{info[0]}-{info[1]}-{info[2]}-{info[3]}-{info[4]}"


def get_tags(info):
    split, split_str, det_split, epoch_start, epoch_end = info
    return {
        "split": split,
        "split_str": split_str,
        "det_split": det_split,
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
        "errcode": 0,
        "message": "",
        "abscal": "",
        "config": "",
        "context": "",
        "obslist": "",
    }


def _amp_obj(x, prof_interp, prof, r):
    amp, off, r_off, r_scale = x
    stack_prof = prof_interp(r_scale * r + r_off)
    prof_norm = (prof - off) / amp
    return np.sum((prof_norm - stack_prof) ** 2)
    # return np.sum((prof - amp*stack_prof - off)**2)


def get_abscal(
    fjobstr,
    fjob,
    fit,
    cfg,
    prof_interp,
    bl_stack,
    ell_msk,
    ext_rad,
    solid_angle,
    pwv,
    el,
):
    if pwv is None:
        return None, None, None, None, None

    aman = fit["aman"]
    r = aman.r.to(u.rad).value
    if cfg.abscal_from_model:
        prof = aman[aman.final_model].mprof.to(u.pW).value
    else:
        prof = aman.rprof.to(u.pW).value
    r_msk = (r < ext_rad) * (r > 0)
    r_lim = r[r_msk]
    stack_prof = prof_interp(r_lim)

    # Estimate scaling and offset in real space
    lin = np.polyfit(stack_prof, prof[r_msk], 1)
    amp = lin[0]
    off = lin[1]
    bounds = [
        (0.5 * amp, 1.5 * amp),
        (off - 10 * abs(off), off + 10 * abs(off)),
        (-1 * cfg.res, cfg.res),
        (0.9, 1.1),
    ]
    res = minimize(
        _amp_obj, (amp, off, 0, 1), (prof_interp, prof[r_msk], r_lim), bounds=bounds
    )  # , method="Powell")
    amp, off, r_off, r_scale = res.x
    stack_prof = prof_interp(r_scale * r_lim + r_off)

    # Optional window func refinement
    if cfg.abscal_lmin > 0:
        bl = beam2bl(r, (prof - off) / amp, cfg.lmax)
        amp *= np.dot(bl[ell_msk], bl_stack[ell_msk]) / np.dot(bl[ell_msk], bl[ell_msk])

    # All the metadata you need should be in `fjob` and `fit` but I load the ones I think you need below
    source = fjob.tags["source"]
    array = (fjob.tags["array"],)
    band = fit["band"]
    timestamp = fit["time"]
    # Solid angle should always be unitful sr
    # get abscal and optical efficiencies.
    abscal, abscal_cmb, opt_eff, raw_abscal, raw_abscal_cmb, raw_opt_eff = (
        au.get_single_abscal(
            amp=amp,
            planet=source,
            timestamp=timestamp,
            band=band,
            ufm=array[0],
            el_obs=el,
            solid_angle=solid_angle,
            pwv_obs=pwv,
        )
    )
    if abscal is None:
        return None, None, None, None, None
    return (
        fjobstr,
        (
            fjob.tags["obs_id"],
            fjob.tags["wafer_slot"],
            fjob.tags["stream_id"],
            fjob.tags["array"],
            fjob.tags["band"],
            fjob.tags["source"],
            amp,
            solid_angle,
            pwv,
            el,
            abscal,
            abscal_cmb,
            opt_eff,
            raw_abscal,
            raw_abscal_cmb,
            raw_opt_eff,
        ),
        r_lim,
        ((prof[r_msk] - off) / amp),
        stack_prof,
    )


def abscal_job(
    job,
    fits,
    fjobs,
    split_dict,
    cfg,
    ctx,
    cfg_str,
    ctx_str,
    data_dir,
    plot_dir,
    ext_rad,
    logger,
):
    skips = Counter()
    data_dir_spl = os.path.join(
        data_dir,
        f"abscal{cfg.test_append}",
        job.tags["split"],
        job.tags["split_str"],
        job.tags["det_split"],
        f"{job.tags['epoch_start']}_{job.tags['epoch_end']}",
    )
    plot_dir_spl = os.path.join(
        plot_dir,
        f"abscal{cfg.test_append}",
        job.tags["split"],
        job.tags["split_str"],
        job.tags["det_split"],
        f"{job.tags['epoch_start']}_{job.tags['epoch_end']}",
    )

    os.makedirs(data_dir_spl, exist_ok=True)
    os.makedirs(plot_dir_spl, exist_ok=True)
    job.mark_visited()

    # Select maps for this stack
    split_vec = split_dict[job.tags["split"]]
    smsk = (
        (split_vec == job.tags["split_str"])
        * (fits["split"] == job.tags["det_split"])
        * (fits["time"] >= float(job.tags["epoch_start"]))
        * (fits["time"] < float(job.tags["epoch_end"]))
    )
    sfjobs = fjobs[smsk]
    sfits = fits[smsk]
    pwvs = bu.get_split_vec(sfits, "pwv_mean", ctx)
    pwvs = [None if p == "None" else float(p) for p in pwvs]
    els = np.asarray(bu.get_split_vec(sfits, "el_center", ctx, round_to=6), dtype=float)
    if len(els):
        logger.log(25, "el range %1.f-%1.f deg", np.min(els), np.max(els))
    logger.log(
        25,
        "%d maps available",
        np.sum(smsk),
    )

    # Load the stack result
    stack_file = os.path.join(data_dir, f"stacks{cfg.test_append}", "beam_pars.h5")
    fit_path = os.path.join(
        job.tags["split"],
        job.tags["split_str"],
        f"{job.tags['det_split']}_{job.tags['epoch_start']}_{job.tags['epoch_end']}",
    )
    try:
        stack_fit = AxisManager.load(stack_file, fit_path)
    except Exception as e:
        msg = f"Failed to load fit with error {e!s}"
        fail(job, ErrCode.FIT_MISSING, msg, logger)
        return job, None
    solid_angle = u.Quantity(stack_fit.bessel.model_solid_angle_true, u.sr).value
    prof_h5_file = os.path.join(
        data_dir,
        f"stack_profiles{cfg.test_append}",
        job.tags["split"],
        job.tags["split_str"],
        f"beam_profiles_{job.tags['split_str']}_{job.tags['det_split']}_{job.tags['epoch_start']}_{job.tags['epoch_end']}.h5",
    )
    if not os.path.isfile(prof_h5_file):
        msg = "Profile h5 file missing!"
        fail(job, ErrCode.FIT_MISSING, msg, logger)
        return job, None
    prof_full = AxisManager.load(prof_h5_file)
    prof = prof_full.prof_cov
    prof_interp = PchipInterpolator(np.deg2rad(np.asarray(prof.r) / 3600), prof.profile)
    bl_stack = np.asarray(prof.bl)
    ell_msk = (prof.ell >= cfg.abscal_lmin) * (bl_stack > 0.01 * bl_stack[0])
    if np.sum(ell_msk) <= 1:
        raise ValueError("ell mask is fewer than 10 points")

    num_fits = len(sfits)
    obslist = []
    abscals = []
    futures = []
    rs = []
    prof_ratios = []
    prof_diffs = []
    prof_errs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, (fit, fjob, pwv, el) in enumerate(zip(sfits, sfjobs, pwvs, els)):
            fjobstr = (
                f"{fjob.tags['obs_id']}-"
                f"{fjob.tags['wafer_slot']}-"
                f"{fjob.tags['stream_id']}-"
                f"{fjob.tags['array']}-"
                f"{fjob.tags['band']}"
            )
            logger.debug("Submitting future for %s (%d/%d)", fjobstr, i + 1, num_fits)
            futures.append(
                executor.submit(
                    get_abscal,
                    fjobstr,
                    fjob,
                    fit,
                    cfg,
                    prof_interp,
                    bl_stack,
                    ell_msk,
                    ext_rad,
                    solid_angle,
                    pwv,
                    el,
                )
            )

            if len(futures) >= max_workers or (i + 1 == num_fits and len(futures) > 0):
                for future in as_completed(futures):
                    fjobstr, abscal, r_lim, prof_norm, prof_stack = future.result()
                    if abscal is None:
                        skips["no pwv /planet temp / el"] += 1
                        continue
                    prof_ratio = prof_norm / prof_stack
                    prof_diff = prof_norm - prof_stack
                    prof_err = np.std(prof_diff)
                    if np.abs(np.mean(prof_ratio)) > cfg.abscal_max_avg_prat:
                        skips["avg prof ratio too far from 1"] += 1
                        continue
                    if np.max(np.abs(prof_ratio - 1)) > cfg.abscal_max_pdiff:
                        skips["max prof ratio too far from 1"] += 1
                        continue
                    if abs(prof_diff[0]) > cfg.abscal_max_pdiff:
                        skips["first prof diff too far from 0"] += 1
                        continue
                    if prof_err > cfg.abscal_max_perr:
                        skips["prof diff rms too high"] += 1
                        continue
                    obs_id, _, stream_id, array, band = abscal[:5]
                    relcal = au.get_obs_relcal(ctx, obs_id, stream_id, array, band)
                    obslist.append(fjobstr)
                    abscals.append((*abscal, relcal))
                    rs.append(r_lim)
                    prof_ratios.append(prof_ratio)
                    prof_diffs.append(prof_diff)
                    prof_errs.append(prof_err)
                logger.log(
                    25,
                    "%d/%d maps processed (%d in obslist)",
                    i + 1,
                    num_fits,
                    len(obslist),
                )
                del futures
                futures = []

    logger.log(
        25,
        "Skipped by reason: %s, (no pwv: %d)",
        dict(skips),
        sum(p is None for p in pwvs),
    )
    obslist = np.unique(obslist)
    if len(obslist) == 0:
        msg = "No maps made it into abscal!"
        fail(job, ErrCode.NO_MAPS, msg, logger)
        return job, None

    logger.log(
        25,
        "%d maps in abscal",
        len(obslist),
    )
    logger.debug(obslist)

    # Setup output rset
    all_src = np.array([job.tags["source"] for job in sfjobs])
    dtype = [
        ("obs_id", fits.dtype["obs_id"]),
        ("wafer_slot", fits.dtype["wafer_slot"]),
        ("stream_id", fits.dtype["stream_id"]),
        ("array", fits.dtype["array"]),
        ("band", fits.dtype["band"]),
        ("source", all_src.dtype),
        ("amp", float),
        ("solid_angle", float),
        ("pwv", float),
        ("el", float),
        ("abscal", float),
        ("abscal_cmb", float),
        ("opt_eff", float),
        ("raw_abscal", float),
        ("raw_abscal_cmb", float),
        ("raw_opt_eff", float),
        ("relcal", float),
    ]
    abscal_perobs = metadata.ResultSet.from_friend(
        np.fromiter(abscals, dtype, count=len(abscals))
    )

    # Save
    fname = f"abscals_{job.tags['split_str']}_{job.tags['det_split']}_{job.tags['epoch_start']}_{job.tags['epoch_end']}.h5"
    outpath = os.path.join(data_dir_spl, fname)
    with h5py.File(outpath, "w") as f:
        write_dataset(abscal_perobs, f, address="abscal_perobs", overwrite=True)

    plt.plot(
        3600 * np.rad2deg(np.asarray(rs).T),
        np.asarray(prof_ratios).T,
        color="b",
        alpha=0.25,
    )
    plt.xlabel('radius (")')
    plt.ylabel("Normalized Profile/Stack Profile")
    plt.title(f"Abscal Profile Ratio for {job.tags['split_str']}")
    plt.savefig(os.path.join(plot_dir_spl, "prof_ratio.png"))
    plt.close()

    plt.plot(
        3600 * np.rad2deg(np.asarray(rs).T),
        np.asarray(prof_diffs).T,
        color="b",
        alpha=0.25,
    )
    plt.xlabel('radius (")')
    plt.ylabel("Normalized Profile - Stack Profile")
    plt.title(f"Abscal Profile Difference for {job.tags['split_str']}")
    plt.savefig(os.path.join(plot_dir_spl, "prof_diff.png"))
    plt.close()

    plt.hist(np.asarray(prof_errs))
    plt.xlabel("Profile Error (rms)")
    plt.ylabel("Counts")
    plt.title(f"Abscal Profile Error for {job.tags['split_str']}")
    plt.savefig(os.path.join(plot_dir_spl, "prof_err.png"))
    plt.close()

    set_tag(job, "config", cfg_str)
    set_tag(job, "context", ctx_str)
    set_tag(job, "message", "Success!")
    set_tag(job, "abscal", outpath)
    set_tag(job, "obslist", ",".join(obslist))
    job.jstate = cast(sqy.Column[str], jobdb.JState.done)

    return job, abscal_perobs


# Setup logger
logger = init_log()
if logger.extra is None:
    raise ValueError("Logger doesn't have adapter set up!")
logger.extra = cast(
    dict,
    logger.extra,
)

# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg(
    args,
    cfg_dict,
    {"map_mask_size": "mask_size", "abscal_source_list": "source_list"},
)

with open(cfg.ctx_path) as f:
    ctx_str = yaml.dump(yaml.safe_load(f))
ctx = Context(cfg.ctx_path)

if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")

ext_rad = np.deg2rad(cfg.extent / 3600)
ext_rad *= cfg.abscal_r_frac

# Setup folders
plot_dir, data_dir = setup_paths(
    cfg.root_dir,
    "beams",
    cfg.tel,
    f"{cfg.pointing_type}{(cfg.append != '') * '_'}{cfg.append}{(cfg.single_det) * '_single_det'}",
)
os.makedirs(plot_dir, exist_ok=True)
fpath = os.path.join(data_dir, f"beam_pars{cfg.test_append}.h5")
if myrank == 0:
    of_path_noa = os.path.join(data_dir, "beam_pars.h5")
    if os.path.isfile(of_path_noa) and cfg.copy_fits_test and cfg.test_append != "":
        shutil.copyfile(of_path_noa, fpath)
jdb = make_jobdb(comm, data_dir, cfg.test_append)

# Det splits
det_split_names = ["full"] + cfg.det_splits

# Setup jobdb
jdb, all_jobs = setup_jobs(
    comm,
    data_dir,
    "make_abscal",
    get_jobdict,
    partial(
        get_jobit,
        cfg=cfg,
        stack_jobs=jdb.get_jobs(jclass="stack_maps", jstate="done"),
        det_splits=det_split_names,
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
    cfg.test_append,
)
all_jobs = np.array(all_jobs)

# Load fits
logger.info("Loading map metadata and fits")

all_fits = None
fjobs = None
mjobdict = None
if myrank == 0:
    fjobs = np.array(jdb.get_jobs(jclass="fit_map", jstate="done"))
    fjobs = [job for job in fjobs if (job.tags["source"] in cfg.source_list)]
    fjobs = [job for job in fjobs if (job.tags["split"] in det_split_names)]
    logger.info("Fit jobs loaded")
    all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs)
    logger.info("Fits loaded")
    snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
    solid_angle = bu.get_fit_vec(all_fits, "gauss.data_solid_angle_corr")
    fwhm_exp = (
        np.array([cfg.nominal_fwhm[band] for band in all_fits["band"]]) * u.arcmin
    )
    data_fwhm = bu.get_fit_vec(all_fits, "data_fwhm")
    msk = snr > cfg.min_stack_snr
    msk *= data_fwhm < 1.5 * fwhm_exp
    msk *= data_fwhm > 0.5 * fwhm_exp
    msk *= solid_angle > 0
    pwv = bu.get_split_vec(all_fits, "pwv_mean", ctx)
    pwv[pwv == "None"] = "1"
    pwv = np.array(pwv, float)
    el = np.deg2rad(np.array(bu.get_split_vec(all_fits, "el_center", ctx), float))
    msk *= pwv / np.sin(el) <= cfg.max_pwv
    all_fits = all_fits[msk]
    fjobs = np.array(fjobs)[msk]
    logger.info("Fits filtered")
logger.info("Broadcasting")
all_fits = comm.bcast(all_fits)
fjobs = comm.bcast(fjobs)
mjobdict = comm.bcast(mjobdict)

logger.info("%d maps to add", len(fjobs))
if len(fjobs) == 0:
    sys.exit(0)

if args.plot_only:
    logger.info("Running in plot only mode!")
    logger.error("Plot only mode broken right now!")
    sys.exit(1)

# Get splits
all_splits = np.unique([job.tags["split"] for job in all_jobs])
split_dict = {
    split: bu.get_split_vec(all_fits, split, ctx, metasplits=cfg.metasplits)
    for split in all_splits
}

# Work out which maps each job needs.
if nproc > 1:
    assignments = None
    job_maps = None
    if myrank == 0:
        logger.info("Building job/map overlap graph")
        job_maps = [bu.get_job_maps(job, all_fits, split_dict) for job in all_jobs]
        assignments = bu.distribute_jobs(all_jobs, job_maps, nproc, logger)
    job_idx = comm.scatter(assignments, root=0)
    joblist = all_jobs[job_idx].tolist() + [None]
    job_maps = comm.bcast(job_maps, root=0)
    needed = np.fromiter(set().union(*(job_maps[i] for i in job_idx)), dtype=np.int64)
    fits = all_fits[needed]
    fjobs_local = fjobs[needed]
    split_dict = {key: val[needed] for key, val in split_dict.items()}
else:
    joblist = all_jobs.tolist() + [None]
    fits = all_fits
    fjobs_local = fjobs


pending_job = None
for i, j in enumerate(joblist):
    if pending_job is not None:
        logger.debug("Writing to db")
        update_jobs_retry(jdb, [pending_job], nproc * 10, logger)
    pending_job = None
    job = None
    if j is not None:
        with jdb.session_scope() as session:
            job = session.get(Job, j.id)
            if job is not None:
                session.expunge(job)
    if job is None:
        continue

    job_str = get_jobstr(job)
    logger.extra["extra"] = f" [{job_str} ({i + 1}/{len(joblist) - 1})]"
    logger.log(25, "Making stack")

    with log_lvl(logger, 15):
        pending_job, abscal_perobs = abscal_job(
            job,
            fits,
            fjobs_local,
            split_dict,
            cfg,
            ctx,
            cfg_str,
            ctx_str,
            data_dir,
            plot_dir,
            ext_rad,
            logger,
        )
    logger.log(25, "Done with abscal")

comm.barrier()
if myrank == 0:
    jobs = [
        job
        for job in jdb.get_jobs(jclass="make_abscal", jstate="done")
        if job.tags["split"] == "mainsrc+tube_slot+band"
        and job.tags["det_split"] == "full"  # TODO: make split used here configurable
    ]
    if not jobs:
        logger.info("No abscal jobs found, skipping db creation")
        sys.exit(0)
    result_dict = au.make_results_dict(
        [read_dataset(job.tags["abscal"], "abscal_perobs") for job in jobs]
    )
    out_dir = os.path.join(data_dir, f"abscal{cfg.test_append}")
    save_pickle(result_dict, os.path.join(out_dir, f"results_{date_str()}.pk"))
    epochs = {f"{start:.0f}_{stop:.0f}": (start, stop) for start, stop in cfg.epochs}
    db = au.make_db(result_dict, os.path.join(out_dir, "abscal.h5"), epochs=epochs)
    db.to_file(os.path.join(out_dir, "abscals.sqlite"))
sys.stdout.flush()
logger.extra["extra"] = ""
logger.info("Done with all abscals")
