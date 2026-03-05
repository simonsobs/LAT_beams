import os
import sys
from functools import partial

import h5py
import numpy as np
from astropy import constants as const
from astropy import units as u
from mpi4py import MPI
from pixell import enmap
from pshmem.locking import MPILock
from sotodlib.core import AxisManager, Context
from sotodlib.site_pipeline.jobdb import Job

import lat_beams.fitting.models as bm
from lat_beams.beam_utils import (
    crop_maps,
    estimate_cent,
    get_fwhm_radial_bins,
    process_model,
    radial_profile,
)
from lat_beams.fitting.map import (
    fit_bessel_map,
    fit_gauss_map,
    fit_multipole_map,
    make_guess,
)
from lat_beams.plotting import plot_map_complete
from lat_beams.utils import (
    get_args_cfg,
    init_log,
    set_tag,
    setup_cfg,
    setup_jobs,
    setup_paths,
)

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()


def get_jobdict(jdb):
    jobdict = {
        f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['band']}-{job.tags['comps']}": job
        for job in jdb.get_jobs(jclass="fit_map")
    }
    return jobdict


def get_jobit(jdb):
    maplist = jdb.get_jobs(jclass="beam_map", jstate="done", locked=False)
    maplist = np.array_split(maplist, nproc)[myrank]
    return maplist


def get_jobstr(mjob, ctx, start_time, stop_time):
    job_str = f"{mjob.tags['obs_id']}-{mjob.tags['wafer_slot']}-{mjob.tags['stream_id']}-{mjob.tags['band']}-{mjob.tags['comps']}"
    obs = ctx.obsdb.get(mjob.tags["obs_id"])
    if args.obs_ids is None and (
        obs["timestamp"] < start_time or obs["timestamp"] >= stop_time
    ):
        return None
    if args.obs_ids is not None and obs["obs_id"] not in args.obs_ids:
        return None
    return job_str


def get_tags(mjob):
    tags = {
        "obs_id": mjob.tags["obs_id"],
        "wafer_slot": mjob.tags["wafer_slot"],
        "stream_id": mjob.tags["stream_id"],
        "band": mjob.tags["band"],
        "comps": mjob.tags["comps"],
        "source": mjob.tags["source"],
        "message": "",
        "resid": "",
        "resid_weights": "",
        "config": "",
    }
    return tags


# Setup logger
logger = init_log()

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

# Setup folders and files
plot_dir, data_dir = setup_paths(
    cfg.root_dir,
    "beams",
    cfg.tel,
    f"{cfg.pointing_type}{(cfg.append!="")*'_'}{cfg.append}",
)
outfile = None
if myrank == 0:
    outfile = h5py.File(os.path.join(data_dir, "beam_pars.h5"), "a")

# Get the jobs, make them if we need to
ctx = Context(cfg.ctx_path)
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
jdb, all_jobs = setup_jobs(
    comm,
    data_dir,
    "fit_map",
    get_jobdict,
    get_jobit,
    partial(get_jobstr, ctx=ctx, start_time=cfg.start_time, stop_time=cfg.stop_time),
    get_tags,
    cfg.source_list,
    args.overwrite,
    args.retry_failed,
    args.job_memory,
    args.job_memory_buffer,
    False,
    logger,
)

# Even things out
joblist = np.array_split(all_jobs, nproc)[myrank].tolist()
n_maps = comm.allgather(len(joblist))
max_maps = np.max(n_maps)
if n_maps[0] != max_maps:
    raise ValueError("Root doesn't have max maps!")
joblist += [None] * (1 + max_maps - len(joblist))

to_save = (None, None)
map_jobs = jdb.get_jobs(jclass="beam_map", jstate="done")
map_jobdict = {
    f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['band']}": job
    for job in map_jobs
}
job = None
mpilock = MPILock(comm)
for i, j in enumerate(joblist):
    comm.barrier()
    sys.stdout.flush()
    to_save = comm.gather(to_save, root=0)
    if myrank == 0 and to_save is not None and outfile is not None:
        for aman, path in to_save:
            if aman is None:
                continue
            aman.save(outfile, path, overwrite=True)
        outfile.flush()
    comm.barrier()

    # To avoid multiproc issues where the database is locked we lock and unlock serially
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
        to_save = (None, None)
        continue

    job.mark_visited()
    obs_id = job.tags["obs_id"]
    ufm = job.tags["stream_id"]
    ws = job.tags["wafer_slot"]
    band = job.tags["band"]
    logger.normal("Fitting %s %s %s(%s/%s)", obs_id, ufm, band, i + 1, n_maps[myrank])

    # Get map job
    job_str = f"{obs_id}-{ws}-{ufm}-{band}"
    if job_str not in map_jobdict:
        msg = "No map job"
        logger.debug("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue
    map_job = map_jobdict[job_str]
    comps = map_job.tags["comps"]

    # Save config info
    set_tag(job, "config", cfg_str)
    set_tag(job, "comps", comps)

    # Load the maps
    try:
        solved = enmap.read_map(os.path.join(data_dir, map_job.tags["solved"]))[0]
        weights = enmap.read_map(os.path.join(data_dir, map_job.tags["weights"]))[0][0]
    except FileNotFoundError:
        msg = "Missing map files"
        logger.error("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue
    pixsize = 3600 * solved.wcs.wcs.cdelt[1]  # type: ignore

    # Check if this is a bogus map
    if np.sum(~(weights == 0)) == 0:
        msg = "Weights all 0"
        logger.error("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue

    # Estimatr SNR
    snr_extent_pix = int(cfg.snr_extent // pixsize)
    cent = estimate_cent(solved, cfg.smooth_kern / pixsize, cfg.buf)
    sig = solved[cent]
    noise = solved.copy()
    xmin = max(0, cent[0] - snr_extent_pix)
    xmax = min(solved.shape[0], cent[0] + snr_extent_pix)
    ymin = max(0, cent[1] - snr_extent_pix)
    ymax = min(solved.shape[1], cent[1] + snr_extent_pix)
    noise[xmin:xmax, ymin:ymax] = np.nan
    noise = np.nanstd(np.diff(noise))
    snr = sig / noise

    if snr < cfg.min_snr:
        msg = "Data SNR too low"
        logger.error("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue

    # Slice things
    solved, weights = crop_maps([solved, weights], cent, int(cfg.extent // pixsize))
    posmap = enmap.posmap(solved.shape, solved.wcs)
    cent = estimate_cent(solved, cfg.smooth_kern / pixsize, cfg.buf_crop)
    fscale_fac = 90.0 / float(band[1:])
    band_mask_size = np.deg2rad(fscale_fac * cfg.mask_size)

    # Make weights and zero things out
    weights[~np.isfinite(weights)] = 0
    weights[~np.isfinite(solved)] = 0
    solved[~np.isfinite(solved)] = 0

    # Setup aman for output
    aman = AxisManager()
    aman.wrap("noise", noise * u.pW)

    # Fit gaussian model
    cent = estimate_cent(solved, cfg.smooth_kern / pixsize, cfg.buf_crop)
    guess = make_guess(
        amp=solved[cent],
        fwhm_xi=np.deg2rad(cfg.nomimal_fwhm[band] / 60.0),
        fwhm_eta=np.deg2rad(cfg.nomimal_fwhm[band] / 60.0),
        xi0=posmap[1][cent[0], cent[1]],
        eta0=posmap[0][cent[0], cent[1]],
        phi=0,
        off=0,
    )
    gauss_params, model = fit_gauss_beam(
        solved,
        weights,
        posmap,
        guess,
        "pW",
        cfg.sym_gauss,
        7,
    )
    if gauss_params is None or model is None:
        msg = "Fit failed"
        logger.error("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue

    # Compute the gaussian model
    model = bm.gaussian2d_from_aman(posmap, gauss_params)

    # Check clipping
    c = np.unravel_index(np.argmax(model, axis=None), model.shape)
    min_c_dist = np.min(np.hstack((c, np.array(solved.shape) - np.array(c)))) * pixsize
    if min_c_dist < 120 * cfg.nomimal_fwhm[band]:
        msg = "Source too close to edge of map"
        logger.error("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue

    # Get FWHM from data
    rprof = radial_profile(solved, c[::-1])
    r = np.linspace(0, len(rprof), len(rprof)) * pixsize
    rmsk = r < 3 * 60 * cfg.nomimal_fwhm[band] / 2.355
    data_fwhm = get_fwhm_radial_bins(r[rmsk], rprof[rmsk], interpolate=True) * u.arcsec
    aman.wrap("data_fwhm", data_fwhm)
    aman.wrap("r", r * u.arcsec)
    aman.wrap("rprof", rprof * u.pW)

    # FWHM check
    if abs(1 - data_fwhm.value / (60 * cfg.nomimal_fwhm[band])) > cfg.fwhm_tol:
        msg = "Data FWHM out of tolerance"
        logger.error("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue

    # Process and save fit model
    gauss_params = process_model(
        gauss_params,
        solved,
        model - gauss_params.off.value,
        noise,
        cfg.min_snr,
        c,
        u.pW,
        pixsize,
        data_fwhm,
        cfg.min_sigma,
        job,
        logger,
    )
    if gauss_params is None:
        to_save = (None, None)
    aman.wrap("gauss", gauss_params)
    for to_parent in ["amp", "off", "xi0", "eta0"]:
        aman.wrap(to_parent, gauss_params[to_parent])
    aman.wrap("final_model", "gauss")

    # Get gauss multipoles if we want them
    if cfg.gauss_multipole:
        base_beam = (model - gauss_params.off.value) / gauss_params.amp.value
        gauss_multipole_params, model = fit_multipole_model(
            solved - gauss_params.off.value,
            weights,
            posmap,
            gauss_params,
            "pW",
            base_beam,
            cfg.n_multipoles,
        )
        gauss_multipole_params = process_model(
            gauss_multipole_params,
            solved,
            model,
            noise,
            cfg.min_snr,
            c,
            u.pW,
            pixsize,
            data_fwhm,
            cfg.min_sigma,
            job,
            logger,
        )
        aman.wrap("gauss_multipole", gauss_multipole_params)
        aman.final_model = "gauss_multipole"

    # Get bessel beam if we want
    if cfg.bessel_beam:
        bessel_beam_params, model = fit_bessel_model(
            solved,
            weights,
            posmap,
            gauss_params,
            "pW",
            cfg.n_bessel,
            cfg.n_multipoles,
            cfg.aperature,
            const.c / (float(band[1:]) * u.GHz),
            cfg.force_bessel_cent,
            cfg.bessel_wing,
            band_mask_size,
            data_fwhm,
        )
        bessel_beam_params = process_model(
            bessel_beam_params,
            solved,
            model,
            noise,
            cfg.min_snr,
            c,
            u.pW,
            pixsize,
            data_fwhm,
            cfg.min_sigma,
            job,
            logger,
        )
        if bessel_beam_params is None:
            continue
        aman.wrap("bessel", bessel_beam_params)
        aman.final_model = "bessel"

    # Save residual
    resid = solved.copy()
    resid -= model
    fname = map_job.tags["solved"]
    enmap.write_map(
        os.path.join(data_dir, f"{'_'.join(fname.split('_')[:-1])}_resid.fits"),
        resid,
        "fits",
        allow_modify=True,
    )
    enmap.write_map(
        os.path.join(data_dir, f"{'_'.join(fname.split('_')[:-1])}_resid_weights.fits"),
        weights,
        "fits",
        allow_modify=True,
    )
    set_tag(
        job,
        "resid",
        f"{'_'.join(fname.split('_')[:-1])}_resid.fits",
    )
    set_tag(
        job,
        "resid_weights",
        f"{'_'.join(fname.split('_')[:-1])}_resid_weights.fits",
    )

    # Plot
    obs = ctx.obsdb.get(obs_id)
    ufm_plot_dir = os.path.join(
        plot_dir,
        job.tags["source"],
        str(obs["timestamp"])[:5],
        obs_id,
        job.tags["stream_id"],
    )
    os.makedirs(ufm_plot_dir, exist_ok=True)
    [[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(
        solved.corners(corner=False)
    )
    plt_cent = (aman.xi0.to(u.arcsec).value, aman.eta0.to(u.arcsec).value)
    norm = 1.0 / aman.amp.value
    posmap = np.rad2deg(posmap) * 3600
    for dat, label in [(model, "model"), (resid, "resid")]:
        plot_map_complete(
            dat,
            posmap,
            pixsize,
            cfg.extent,
            plt_cent,
            ufm_plot_dir,
            f"{obs_id} {ufm} {band}",
            comps="T",
            log_thresh=cfg.log_thresh,
            append=label,
            units='"',
            lognorm=norm,
        )

    # Save
    aman.wrap("data_solid_angle_corr", aman[aman.final_model].data_solid_angle_corr)
    aman_path = os.path.join(obs_id, ufm, band)
    to_save = (aman, aman_path)

    set_tag(job, "message", "Success")
    job.jstate = "done"

comm.barrier()
if outfile is not None:
    outfile.close()
sys.stdout.flush()
logger.info("Done with all fits")
logger.flush()
