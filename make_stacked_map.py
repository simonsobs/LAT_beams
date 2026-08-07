import os
import yaml
from copy import deepcopy
from functools import partial
import sys
from typing import cast

import astropy.units as u
import numpy as np
from pixell import enmap, reproject
from sotodlib.core import Context
from mpi4py import MPI
import sqlalchemy as sqy

from lat_beams import beam_utils as bu
from lat_beams.plotting import plot_map_complete
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.jobdb import Job
from lat_beams.utils import get_args_cfg, init_log, make_jobdb, setup_cfg, setup_paths
from lat_beams.utils import (
    get_args_cfg,
    init_log,
    make_jobdb,
    set_tag,
    setup_cfg,
    setup_jobs,
    setup_paths,
    ErrCode,
    fail,
)
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

def get_jobdict(jdb):
    jobdict = {
        f"{job.tags['split']}-{job.tags['split_str']}-{job.tags['det_split']}-{job.tags['epoch_start']}-{job.tags['epoch_end']}": job
        for job in jdb.get_jobs(jclass="solve_maps")
    }
    return jobdict


def get_jobit(jdb, cfg, all_fits, all_fjobs):
    _ = jdb
    jobit = []
    if myrank == 0:
        for epoch in cfg.epochs:
            times = all_fits["time"]
            tmsk = (times >= epoch[0]) * (times < epoch[1])
            if np.sum(tmsk) == 0:
                continue
            fjobs = all_fjobs[tmsk]
            fits = bu.load_beam_fits_from_jobs(fpath, fjobs)
            for split in cfg.split_by:
                split_vec = bu.get_split_vec(fits, split, ctx, metasplits=cfg.metasplits)
                for spl in np.unique(split_vec):
                    if "NOMATCH" in spl:
                        continue
                    for det_split in np.unique(fits["split"]):
                        jobit += [(split, spl, det_split, epoch[0], epoch[1])]
    return jobit


def get_jobstr(info):
    return f"{info[0]}-{info[1]}-{info[2]}-{info[3]}-{info[4]}"


def get_tags(info):
    split, split_str, det_split, epoch_start, epoch_end = info
    tags = {
        "split": split,
        "split_str": split_str,
        "det_split": det_split,
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
        "errcode": 0,
        "message": "",
        "map_stack": "",
        "ivar_stack": "",
        "map_ivar": "",
        "resid_map_stack": "",
        "resid_ivar_stack": "",
        "resid_map_ivar": "",
        "config": "",
        "context": "",
        "obslist": "",
    }
    return tags

def view_TQU(imap):
    padded = imap
    if len(imap) == 1:
        padded = enmap.zeros((3,) + imap.shape[1:], imap.wcs)
        padded[0][:] = imap[0][:]
    return padded


# Setup logger
logger = init_log()

# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg(
    args,
    cfg_dict,
    {
        "map_mask_size": "mask_size",
    },
)
with open(cfg.ctx_path) as f:
    ctx_str = yaml.dump(yaml.safe_load(f))
ctx = Context(cfg.ctx_path)
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
pixsize = 3600 * np.rad2deg(cfg.res)
op = np.ndarray.__iadd__

# Setup folders
plot_dir, data_dir = setup_paths(
    cfg.root_dir,
    "beams",
    cfg.tel,
    f"{cfg.pointing_type}{(cfg.append!='')*'_'}{cfg.append}{(cfg.single_det)*'_single_det'}",
)
os.makedirs(plot_dir, exist_ok=True)
fpath = os.path.join(data_dir, "beam_pars.h5")
jdb = make_jobdb(None, data_dir)

# Get jobs
mjobdict = {
    f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['array']}-{job.tags['band']}": job
    for job in jdb.get_jobs(jclass="beam_map", jstate="done")
}
fjobs = np.array(jdb.get_jobs(jclass="fit_map", jstate="done"))

logger.info("%d maps to add", len(fjobs))
if len(fjobs) == 0:
    sys.exit(0)

# Load fits
all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs.tolist())
snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
solid_angle = bu.get_fit_vec(all_fits, "gauss.data_solid_angle_corr")
msk = snr > cfg.min_stack_snr 
msk *= solid_angle > 0
fwhm_exp = np.array([cfg.nominal_fwhm[band] for band in all_fits["band"]]) * u.arcmin
data_fwhm = bu.get_fit_vec(all_fits, "data_fwhm")
msk *= data_fwhm < 2 * fwhm_exp
msk *= data_fwhm > .5 * fwhm_exp
all_fits = all_fits[msk]
fjobs = fjobs[msk]

# Setup jobdb
jdb, all_jobs = setup_jobs(
    comm,
    data_dir,
    "stack_maps",
    get_jobdict,
    partial(
        get_jobit,
        cfg=cfg,
        all_fits=all_fits,
        all_fjobs=fjobs,
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

# Make template map
ext_rad = np.deg2rad(cfg.mask_size)
pix_extent = 2 * int(3600 * cfg.mask_size // pixsize)
# rowmajor = True here to match sotodlib
twcs = enmap.wcsutils.build(
    [0, 0],
    res=np.rad2deg(cfg.res),
    shape=(pix_extent, pix_extent),
    system="tan",
    rowmajor=True,
)
tmap = enmap.zeros((3, pix_extent, pix_extent), twcs)

# Put template maps in a dict
# Structure here is job_str -> type (map/resid) -> data (map/ivar)
map_types = ("", "resid")
template_dict = {
        map_type: {d : deepcopy(tmap) for d in ("map_stack", "ivar_stack", "map_ivar")} 
        for map_type in map_types
}

if args.plot_only:
    logger.info("Running in plot only mode!")
    logger.error("Plot only mode broken right now!")
    sys.exit(1)

# Split inputs up by mpi rank
fjobs = np.array_split(fjobs, nproc)[myrank]
fits = np.array_split(all_fits, nproc)[myrank]

# Get splits
all_splits = np.unique([job.tags['split'] for job in all_jobs])
split_dict = {split : bu.get_split_vec(fits, split, ctx, metasplits=cfg.metasplits) for split in all_splits}  

# Loop through jobs 
true_job = None
for job in all_jobs:
    job_str = f"{job.tags['split']}-{job.tags['split_str']}-{job.tags['det_split']}-{job.tags['epoch_start']}-{job.tags['epoch_end']}"
    jobdict = deepcopy(template_dict)
    logger.info("Making stack: %s", job_str)

    comm.barrier()

    split_vec = split_dict[job.tags['split']]
    smsk = (split_vec == job.tags['split_str']) * (fits["split"] == job.tags['det_split'])
    sfjobs = fjobs[smsk]
    sfits = fjobs[smsk]
    logger.info("%d maps to add", np.sum(smsk))

    obslist = []
    comm.barrier()
    for fit, fjob in zip(sfits, sfjobs):
        fjobstr = f"{fjob.tags['obs_id']}-{fjob.tags['wafer_slot']}-{fjob.tags['stream_id']}-{fjob.tags['array']}-{fjob.tags['band']}"
        logger.debug("Adding %s", fjobstr)
        if fjobstr not in mjobdict:
            logger.debug("Map job not found for %s", fjobstr)
            continue
        mjob = mjobdict[fjobstr]
        # Load
        for map_type in map_types:
            if map_type == "":
                map_path = os.path.join(data_dir, mjob.tags["solved"].format(split=fjob.tags["split"]))
                ivar_path = os.path.join(data_dir, mjob.tags["weights"].format(split=fjob.tags["split"]))
            elif map_type == "resid":
                map_path = os.path.join(data_dir, fjob.tags["resid"])
                ivar_path = os.path.join(data_dir, fjob.tags["resid_weights"])
            else:
                raise ValueError(f"Bad map type {map_type}")
            try:
                imap = enmap.read_map(map_path)
                if len(imap.shape) == 2:
                    imap = imap.reshape((1,) + imap.shape)
                ivar = enmap.read_map(ivar_path)
                if len(ivar.shape) == 4:
                    ivar = ivar[np.diag_indices(len(ivar))]
                ivar = ivar.reshape(imap.shape)
            except FileNotFoundError:
                logger.debug("Maps missing for job: %s", fjobstr)
                continue
            # Make everything look like TQU
            imap = view_TQU(imap)
            ivar = view_TQU(ivar)

            # Crop, recenter, and normalize
            cent = np.array(
                (
                    fit["aman"].gauss.eta0.to(u.rad).value,
                    fit["aman"].gauss.xi0.to(u.rad).value,
                )
            )
            imap = (
                reproject.thumbnails(
                    imap - fit["aman"].gauss.off.value * (map_type == ""),
                    r=ext_rad,
                    coords=cent,
                    oshape=(pix_extent, pix_extent),
                    owcs=twcs,
                    oversample=1,
                )
                / fit["aman"].gauss.amp.value
            )
            ivar = (
                reproject.thumbnails_ivar(
                    ivar,
                    r=ext_rad,
                    coords=cent,
                    oshape=(pix_extent, pix_extent),
                    owcs=twcs,
                )
                * fit["aman"].gauss.amp.value**2
            )

            # If the new center seems very far from the origin then lets skip
            if map_type == "":
                cent_est = bu.estimate_cent(imap[0], sigma=10, buf=1)
                dist = np.linalg.norm(cent_est - imap.wcs.wcs.crpix)
                if dist > cfg.miscenter_thresh:
                    logger.debug(
                        "%s (%s) seems miscentered! Skipping!",
                        fjobstr,
                        mjob.tags["source"],
                    )
                    break

            # Add, structure here is type (map/resid) -> data (map/ivar)
            np.nan_to_num(imap, copy=False, nan=0, posinf=0, neginf=0)
            np.nan_to_num(ivar, copy=False, nan=0, posinf=0, neginf=0)
            jobdict[map_type]["map_stack"].insert(imap * ivar, op=op)
            jobdict[map_type]["ivar_stack"].insert(ivar, op=op)
            jobdict[map_type]["map_ivar"].insert(imap**2 * ivar, op=op)
            obslist += [fjobstr]

    # Wait for all tasks
    comm.barrier()

    # Gather and combine in rank 0
    for map_type in map_types:
        for dat in jobdict[map_type].keys():
             tot = comm.reduce(jobdict[map_type][dat])
             if myrank == 0:
                 if tot is None:
                     raise ValueError("Reduce returned none?")
                 jobdict[map_type][dat] = tot
    obslist = comm.reduce(obslist)

    if myrank != 0:
        continue

    with jdb.session_scope() as session:
        job = session.get(Job, job.id)
        session.expunge(job)

    if job is None:
        raise ValueError("job is None!")
    job.mark_visited()

    if obslist is None or len(obslist) == 0:
        msg = "No maps made it into stack!"
        fail(job, ErrCode.NO_MAPS, msg, logger)
        continue

    # Divide weights, save, and plot
    for map_type in map_types:
        with np.errstate(divide="ignore", invalid="ignore"):
            mv = deepcopy(jobdict[job_str][map_type]["map_stack"])
            jobdict[map_type]["map_stack"] /= jobdict[map_type]["ivar_stack"] # type: ignore
            jobdict[map_type]["map_ivar"] = (jobdict[map_type]["map_ivar"]/jobdict[map_type]["ivar_stack"]) + (mv/jobdict[map_type]["ivar_stack"])**2 # type: ignore
            jobdict[map_type]["map_ivar"] = 1/jobdict[map_type]["map_ivar"] # type: ignore
            np.nan_to_num( jobdict[map_type]["map_stack"], copy=False, nan=0, posinf=0, neginf=0)
            np.nan_to_num( jobdict[map_type]["map_ivar"], copy=False, nan=0, posinf=0, neginf=0)

        for name in ["map_stack", "ivar_stack", "map_ivar"]:
            omap = jobdict[map_type][name]
            omap = cast(enmap.ndmap, omap)
            data_dir_spl = os.path.join(data_dir, "stacks", job.tags["split"], job.tags["split_str"], job.tags["det_split"], f"{job.tags['epoch_start']}_{job.tags['epoch_end']}")
            plot_dir_spl = os.path.join(plot_dir, "stacks", job.tags["split"], job.tags["split_str"], job.tags["det_split"], f"{job.tags['epoch_start']}_{job.tags['epoch_end']}")
            os.makedirs(data_dir_spl, exist_ok=True)
            os.makedirs(plot_dir_spl, exist_ok=True)
            path = os.path.join(data_dir_spl, 
                f"{job.tags['split_str']}_{job.tags['det_split']}_{job.tags['epoch_start']}_{job.tags['epoch_end']}{'_'*bool(map_type)}{map_type}_{name}.fits",
            )
            enmap.write_map(
                    path,
                    omap,
                    "fits",
                    allow_modify=True,
                )
            set_tag(job, f"{map_type}{'_'*bool(map_type)}{name}", path) 
            if "ivar" in name:
                continue

            posmap = omap.posmap()
            posmap = np.rad2deg(posmap) * 3600
            for append, smap in [
                ("", omap),
                ("_smooth3pix", enmap.smooth_gauss(omap, 3 * cfg.res)),
            ]:
                plot_map_complete(
                    smap,
                    posmap,
                    pixsize,
                    cfg.extent,
                    (0, 0),
                    plot_dir_spl,
                    f"{job.tags['split_str']} {job.tags['det_split']} {job.tags['epoch_start']} {job.tags['epoch_end']}{' '*bool(map_type)}{map_type} {name}",
                    log_thresh=cfg.log_thresh,
                    append=name + append,
                    qrur=True,
                )

    # Save metadata and close out job
    set_tag(job, "config", cfg_str)
    set_tag(job, "context", ctx_str)
    set_tag(job, "message", "Success!")
    job.jstate = cast(sqy.Column[str], jobdb.JState.done)
    with jdb.session_scope() as session:
        session.merge(job)
        session.commit()
