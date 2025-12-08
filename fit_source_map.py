import argparse
import glob
import os
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sqlalchemy as sqy
import yaml
from astropy import units as u
from mpi4py import MPI
from pixell import enmap
from sotodlib.core import AxisManager, Context
from sotodlib.site_pipeline import jobdb
from sqlalchemy.pool import NullPool
from tqdm import tqdm

from lat_beams.beam_utils import (
    crop_maps,
    estimate_cent,
    estimate_solid_angle,
    get_fwhm_radial_bins,
    radial_profile,
)
from lat_beams.fitting import fit_gauss_beam
from lat_beams.plotting import plot_map
from lat_beams.utils import print_once, set_tag

plt.rcParams["image.cmap"] = "RdGy_r"


comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

fwhm = {"f090": 2, "f150": 1.3, "f220": 0.95, "f280": 0.83}  # arcmin


parser = argparse.ArgumentParser()
parser.add_argument("cfg", help="Path to the config file")
parser.add_argument("--obs_ids", nargs="+", help="Pass a list of obs ids to run on")
parser.add_argument(
    "--overwrite", "-o", action="store_true", help="Overwrite an existing fit"
)
parser.add_argument(
    "--start_from", "-s", default=0, type=int, help="Skip to the nth obs (0 indexed)"
)
parser.add_argument(
    "--lookback",
    "-l",
    type=float,
    help="Amount of time to lookback for query, overides start time from config",
)
parser.add_argument(
    "--retry_failed", "-r", action="store_true", help="Retry failed maps"
)
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

# Get some global settings
extent = cfg["extent"] = cfg.get("extent", 900)
snr_extent = cfg["snr_extent"] = cfg.get("snr_extent", 500)
min_sigma = cfg["min_sigma"] = cfg.get("min_sigma_fit", 3)
min_snr = cfg["min_snr"] = cfg.get("min_snr", 10)
gauss_multipoles = cfg["gauss_multipoles"] = cfg.get("gauss_multipoles", tuple())
gauss_multipoles = tuple(gauss_multipoles)
sym_gauss = cfg["sym_gauss"] = cfg.get("sym_gauss", True)
fwhm_tol = cfg["fwhm_tol"] = cfg.get("fwhm_tol", 3)
pointing_type = cfg["pointing_type"] = cfg.get("pointing_type", "pointing_model")
buf = cfg["buf"] = cfg.get("buffer", 30)
log_thresh = cfg["log_thresh"] = cfg.get("log_thresh", 1e-3)
smooth_kern = cfg["smooth_kern"] = cfg.get("smooth_kern", 60)
append = cfg["append"] = cfg.get("append", "")
cfg_str = yaml.dump(cfg)

# Setup folders and files
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
outfile = None

# Let rank 0 make jobdb first to avoid race conditions
if myrank == 0:
    outfile = h5py.File(os.path.join(data_dir, "beam_pars.h5"), "a")
    engine = sqy.create_engine(
        f'sqlite:///{os.path.join(data_dir, "jobdb.db")}',
        connect_args={"timeout": 10},
        poolclass=NullPool,
    )
    jdb = jobdb.JobManager(engine=engine)
    jdb.clear_locks(jobs="all")
comm.barrier()
if myrank != 0:
    engine = sqy.create_engine(
        f'sqlite:///{os.path.join(data_dir, "jobdb.db")}',
        connect_args={"timeout": 10},
        poolclass=NullPool,
    )
    jdb = jobdb.JobManager(engine=engine)

# Get the jobs, make them if we need to
print_once("Setting up jobdb")
start_time = cfg["start_time"]
if args.lookback is not None:
    start_time = time.time() - 3600 * args.lookback
stop_time = cfg["stop_time"]
ctx = Context(cfg.get("context", "/so/metadata/lat/contexts/smurf_detcal.yaml"))
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
maplist = jdb.get_jobs(jclass="beam_map", jstate="done", locked=False)
maplist = np.array_split(maplist, nproc)[myrank]
joblist = []
jobs_to_make = []
print_once("Getting beam map jobs")
jobdict = {
    f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['band']}-{job.tags['comps']}": job
    for job in jdb.get_jobs(jclass="fit_map")
}
print_once("Processing possible jobs")
it = maplist
if myrank == 0:
    it = tqdm(maplist, file=sys.stdout)
for mjob in it:
    sys.stdout.flush()
    job_str = f"{mjob.tags['obs_id']}-{mjob.tags['wafer_slot']}-{mjob.tags['stream_id']}-{mjob.tags['band']}-{mjob.tags['comps']}"
    obs = ctx.obsdb.get(mjob.tags["obs_id"])
    if args.obs_ids is None and (
        obs["timestamp"] < start_time or obs["timestamp"] >= stop_time
    ):
        continue
    elif args.obs_ids is not None and obs["obs_id"] not in args.obs_ids:
        continue
    if job_str in jobdict:
        job = jobdict[job_str]
    else:
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
        job = jdb.make_job(jclass="fit_map", tags=tags, check_existing=False)
        jobs_to_make += [job]
    if job.lock:
        continue
    if (
        args.overwrite
        or job.jstate.name == "open"
        or (job.jstate.name == "failed" and args.retry_failed)
    ):
        joblist += [job]
comm.barrier()

# Make the missing jobs
# Doing this serially so that we don't lock up the db
tot_missing = 0
tot_missing = comm.reduce(len(jobs_to_make), root=0)
print_once(f"Adding {tot_missing} new jobs")
t0 = time.time()
for i in range(nproc):
    print_once(f"\tRank {i} writing")
    if myrank == i:
        jdb.commit_jobs(jobs_to_make)
    comm.barrier()
t1 = time.time()
print_once(f"Took {t1-t0} seconds to add")

# Get the final job list
all_jobs = comm.allgather(joblist)
all_jobs = [job for jobs in all_jobs for job in jobs]
print_once(f"{len(all_jobs)} maps to fit!")
joblist = np.array_split(all_jobs, nproc)[myrank].tolist()

# Even things out
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
    # with jdb.locked(j) as job:
    for r in range(nproc):
        if r == myrank:
            if job is not None:
                jdb.unlock(job)
            job = None
            if j is not None:
                job = jdb.lock(j.id)
        comm.barrier()
    if job is None:
        to_save = (None, None)
        continue

    job.mark_visited()
    obs_id = job.tags["obs_id"]
    ufm = job.tags["stream_id"]
    ws = job.tags["wafer_slot"]
    band = job.tags["band"]
    print(f"(rank {myrank}) Fitting {obs_id} {ufm} {band}({i+1}/{n_maps[myrank]})")

    # Get map job
    job_str = f"{obs_id}-{ws}-{ufm}-{band}"
    if job_str not in map_jobdict:
        msg = "No map job"
        print(f"\t{msg}")
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
        solved = enmap.read_map(map_job.tags["solved"])[0]
        weights = enmap.read_map(map_job.tags["weights"])[0][0]
    except FileNotFoundError:
        msg = "Missing map files"
        print(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue
    pixsize = 3600 * solved.wcs.wcs.cdelt[1]  # type: ignore

    # Check if this is a bogus map
    if np.sum(~(weights == 0)) == 0:
        msg = "Weights all 0"
        print(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue

    # Estimate SNR
    snr_extent_pix = int(snr_extent // pixsize)
    cent = estimate_cent(solved, smooth_kern / pixsize, buf)
    sig = solved[cent]
    noise = solved.copy()
    xmin = max(0, cent[0] - snr_extent_pix)
    xmax = min(solved.shape[0], cent[0] + snr_extent_pix)
    ymin = max(0, cent[1] - snr_extent_pix)
    ymax = min(solved.shape[1], cent[1] + snr_extent_pix)
    noise[xmin:xmax, ymin:ymax] = np.nan
    noise = np.nanstd(np.diff(noise))
    snr = sig / noise

    if snr < min_snr:
        msg = "Data SNR too low"
        print(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue

    # Slice things
    solved, weights = crop_maps([solved, weights], cent, int(extent // pixsize))
    pixmap = enmap.pixmap(solved.shape, solved.wcs)

    # Fit
    cent = estimate_cent(solved, smooth_kern / pixsize, buf)
    fit_params, model = fit_gauss_beam(
        solved, weights, pixmap, cent, gauss_multipoles, sym_gauss, "pW"
    )
    if fit_params is None or model is None:
        msg = "Fit failed"
        print(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue

    # Check SNR again
    if np.nanmax(model) / noise < min_snr:
        msg = "Model SNR too low"
        print(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue

    # Get FWHM from data
    c = np.unravel_index(np.argmax(model, axis=None), model.shape)
    rprof = radial_profile(solved - fit_params["off"].value, c[::-1])
    mprof = radial_profile(model - fit_params["off"].value, c[::-1])
    r = np.linspace(0, len(rprof), len(rprof)) * pixsize
    data_fwhm = get_fwhm_radial_bins(r, rprof, interpolate=True)
    model_fwhm = get_fwhm_radial_bins(r, mprof, interpolate=True)

    # FWHM check
    if abs(1 - data_fwhm / (60 * fwhm[band])) > fwhm_tol:
        msg = "Data FWHM out of tolerance"
        print(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue
    if abs(1 - model_fwhm / (60 * fwhm[band])) > fwhm_tol:
        msg = "Model FWHM out of tolerance"
        print(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        to_save = (None, None)
        continue

    # Get solid angle
    (
        data_solid_angle_meas,
        model_solid_angle_meas,
        model_solid_angle_true,
        data_solid_angle_corr,
    ) = estimate_solid_angle(
        solved, model, pixsize, data_fwhm, c, fit_params["off"].value, min_sigma
    )

    # Adjust shift
    dec, ra = 3600 * np.rad2deg(solved.pix2sky((0, 0)))
    fit_params["xi0"] = ra * u.arcsec - fit_params["xi0"]
    fit_params["eta0"] += dec * u.arcsec

    # Save residual
    resid = solved.copy()
    resid -= model
    fname = map_job.tags["solved"]
    enmap.write_map(
        f"{'_'.join(fname.split('_')[:-1])}_resid.fits",
        resid,
        "fits",
        allow_modify=True,
    )
    enmap.write_map(
        f"{'_'.join(fname.split('_')[:-1])}_resid_weights.fits",
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
    plt_extent = (ra_min, ra_max, dec_min, dec_max)
    plt_cent = (ra_min - pixsize * cent[1], dec_min + pixsize * cent[0])
    for dat, label in [(model, "model"), (resid, "resid")]:
        for log in [False, True]:
            plot_map(
                dat,
                pixsize,
                extent,
                plt_extent,
                cent,
                plt_cent,
                1.0,
                ufm_plot_dir,
                obs_id,
                ufm,
                band,
                "T",
                log,
                log_thresh,
                label,
            )

    # Save
    aman = AxisManager()
    for name, par in fit_params.items():
        aman.wrap(name, par)
    aman.wrap("data_fwhm", data_fwhm * u.arcsec)
    aman.wrap("data_solid_angle_meas", data_solid_angle_meas * u.sr)
    aman.wrap("data_solid_angle_corr", data_solid_angle_corr * u.sr)
    aman.wrap("model_solid_angle_meas", model_solid_angle_meas * u.sr)
    aman.wrap("model_solid_angle_true", model_solid_angle_true * u.sr)
    aman.wrap("noise", noise * u.pW)
    aman.wrap("r", r * u.arcsec)
    aman.wrap("rprof", rprof * u.pW)
    aman.wrap("mprof", mprof * u.pW)
    aman_path = os.path.join(obs_id, ufm, band)
    to_save = (aman, aman_path)

    set_tag(job, "message", "Success")
    job.jstate = "done"

comm.barrier()
sys.stdout.flush()
print_once("Done with all fits")
