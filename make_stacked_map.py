import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import partial
from typing import cast

import astropy.units as u
import numpy as np
import psutil
import sqlalchemy as sqy
import yaml
from expiringdict import ExpiringDict
from mpi4py import MPI
from pixell import enmap, reproject
from sotodlib.coords.planets import get_source_pos
from sotodlib.core import Context
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.jobdb import Job

from lat_beams import beam_utils as bu
from lat_beams.plotting import plot_debug_map, plot_map_complete
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
        for job in jdb.get_jobs(jclass="stack_maps")
    }


def get_jobit(jdb, cfg, all_fits, ctx, det_splits):
    _ = jdb
    jobit = []
    if myrank == 0:
        for epoch in cfg.epochs:
            times = all_fits["time"]
            tmsk = (times >= epoch[0]) * (times < epoch[1])
            if np.sum(tmsk) == 0:
                continue
            fits = all_fits[tmsk]
            for split in cfg.split_by:
                split_vec = bu.get_split_vec(
                    fits, split, ctx, metasplits=cfg.metasplits
                )
                for spl in np.unique(split_vec):
                    if "NOMATCH" in spl:
                        continue
                    for det_split in np.unique(fits["split"]):
                        if det_split not in det_splits:
                            continue
                        jobit += [(split, spl, det_split, epoch[0], epoch[1])]
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


def get_job_maps(job, fits, split_dict):
    split_vec = split_dict[job.tags["split"]]
    smsk = (
        (split_vec == job.tags["split_str"])
        * (fits["split"] == job.tags["det_split"])
        * (fits["time"] >= float(job.tags["epoch_start"]))
        * (fits["time"] < float(job.tags["epoch_end"]))
    )
    return set(np.where(smsk)[0])


def distribute_jobs(all_jobs, job_maps, nranks, logger):
    njobs = len(all_jobs)
    map_jobs = {}
    for ji, maps in enumerate(job_maps):
        for m in maps:
            map_jobs.setdefault(m, set()).add(ji)
    total_unique = len(set().union(*job_maps))
    target = total_unique / nranks
    assignments = [[] for _ in range(nranks)]
    rank_maps = [set() for _ in range(nranks)]
    order = sorted(range(njobs), key=lambda ji: len(job_maps[ji]), reverse=True)

    for r in range(nranks):
        ji = order.pop(0)
        maps = set(job_maps[ji])
        assignments[r].append(ji)
        rank_maps[r].update(maps)
    for ji in order:
        maps = job_maps[ji]
        best_rank = 0
        best_score = np.inf
        for r in range(nranks):
            new_unique = len(rank_maps[r] | maps)
            imbalance = abs(new_unique - target)
            reuse = len(rank_maps[r] & maps)
            job_penalty = len(assignments[r]) * 0.25
            score = imbalance - 0.5 * reuse + job_penalty
            if score < best_score:
                best_score = score
                best_rank = r
        assignments[best_rank].append(ji)
        rank_maps[best_rank].update(maps)

    for r in range(nranks):
        maps = rank_maps[r]
        map_loads = sum(len(job_maps[ji]) for ji in assignments[r])
        reuse = map_loads - len(maps)

        logger.info(
            f"Rank {r}: "
            f"{len(assignments[r])} jobs, "
            f"{len(maps)} unique maps, "
            f"{map_loads} map loads, "
            f"{reuse} cache reuse"
        )

    return assignments


def get_cache_max_len(
    template_map,
    nproc,
    safety_factor=1.2 * max_workers,
    n_cached_types=2,
):
    bytes_per_map = template_map.nbytes * 2 * n_cached_types
    avail = psutil.virtual_memory().available / nproc
    max_len = int((avail / bytes_per_map) / safety_factor)
    return max(1, max_len)


def view_TQU(imap) -> enmap.ndmap:
    padded = imap
    if len(imap) == 1:
        padded = enmap.zeros((3,) + imap.shape[1:], imap.wcs)
        padded[0][:] = imap[0][:]
    return padded


def check_source_overlap(mjob, cfg, imap):
    # Fudging coordinates a bit here
    msrc = mjob.tags["source"]
    X = yaml.safe_load(mjob.tags["X"])
    t = float(X["timestamp"])
    ra, dec = float(X["ra"]), float(X["dec"])
    theta = float(X["psi"] - X["roll"])
    box = imap.box()
    corners = np.array(
        [
            [box[0, 0], box[0, 1]],
            [box[0, 0], box[1, 1]],
            [box[1, 0], box[0, 1]],
            [box[1, 0], box[1, 1]],
        ]
    )
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]])
    rot_corners = corners @ R.T
    rot_box = np.array(
        [
            rot_corners.min(axis=0),
            rot_corners.max(axis=0),
        ]
    )
    overlaps = []
    for source in cfg.map_source_list:
        if source == msrc:
            continue
        src = source
        if src == "taua":
            src = "J83.6272579p22.02159891"
        elif src == "3c279":
            src = "J194.0409868m5.79174024"
        elif src == "2026yeh":
            src = "J12.3065339p35.5610483"
        ra_src, dec_src, _ = get_source_pos(src.title(), t)
        coord = (dec_src - dec, ra - ra_src)
        inside = np.all((coord >= rot_box[0]) & (coord <= rot_box[1]))
        if inside:
            overlaps += [source]
    return overlaps


def process_one_map(
    fit,
    fjob,
    map_type,
    mjobdict,
    data_dir,
    cfg,
    twcs,
    pix_extent,
    ext_rad,
):
    fjobstr = (
        f"{fjob.tags['obs_id']}-"
        f"{fjob.tags['wafer_slot']}-"
        f"{fjob.tags['stream_id']}-"
        f"{fjob.tags['array']}-"
        f"{fjob.tags['band']}"
    )

    if fjobstr not in mjobdict:
        return fjobstr, fjob.tags["split"], map_type, "Map job not found"
    mjob = mjobdict[fjobstr]

    if map_type == "":
        map_path = os.path.join(
            data_dir, mjob.tags["solved"].format(split=fjob.tags["split"])
        )
        ivar_path = os.path.join(
            data_dir, mjob.tags["weights"].format(split=fjob.tags["split"])
        )
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
        return fjobstr, fjob.tags["split"], map_type, f"Missing map {fjobstr}"

    imap = view_TQU(imap)
    ivar = view_TQU(ivar)

    cent = np.array(
        (fit["aman"].gauss.eta0.to(u.rad).value, fit["aman"].gauss.xi0.to(u.rad).value)
    )

    pix = imap.sky2pix(cent)
    ny, nx = imap.shape[-2:]
    y, x = pix

    pixscale = np.abs(imap.pixshape())
    distance_rad = min(
        y * pixscale[0],
        (ny - 1 - y) * pixscale[0],
        x * pixscale[1],
        (nx - 1 - x) * pixscale[1],
    )

    if distance_rad < 0.1 * ext_rad:
        return (
            fjobstr,
            fjob.tags["split"],
            map_type,
            "{fjobstr} ({mjob.tags['source']}) too close to edge! Skipping!",
            None,
        )

    # overlaps = check_source_overlap(mjob, cfg, imap, ext_rad)
    # if len(overlaps) > 0:
    #     return fjobstr, fjob.tags["split"], map_type, f"{fjobstr} ({mjob.tags['source']}) has {overlaps} in map!", None

    norm = (
        fit["aman"].gauss.amp.value
        + fit["aman"].gauss.off.value
        - fit["aman"].bessel.off.value
    )
    imap = (
        reproject.thumbnails(
            imap - fit["aman"].bessel.off.value * (map_type == ""),
            r=ext_rad,
            coords=cent,
            oshape=(pix_extent, pix_extent),
            owcs=twcs,
            oversample=1,
            order=1,
        )
        / norm
    )

    ivar = (
        reproject.thumbnails_ivar(
            ivar,
            r=ext_rad,
            coords=cent,
            oshape=(pix_extent, pix_extent),
            owcs=twcs,
            order=1,
        )
        * norm**2
    )

    ivar_msk = enmap.upgrade(enmap.downgrade(ivar[0], 2), 2, oshape=imap[0].shape) == 0
    modrmap = enmap.modrmap(imap.shape, imap.wcs)
    debug = (
        fjobstr,
        imap,
        ivar,
        np.zeros_like(ivar_msk),
        None,
        None,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )
    if np.sum(ivar_msk[modrmap < 3 * fit["aman"].data_fwhm.to(u.radian).value]):
        return (
            fjobstr,
            fjob.tags["split"],
            map_type,
            f"{fjobstr} ({mjob.tags['source']}) has poor sampling near the source! Skipping!",
            debug,
        )

    if map_type == "":
        cent_est = bu.estimate_cent(
            imap[0],
            ivar[0],
            sigma=10,
            buf=1,
            ret_smooth=False,
        )
        if np.linalg.norm(cent_est - imap.wcs.wcs.crpix) > cfg.miscenter_thresh:
            return (
                fjobstr,
                fjob.tags["split"],
                map_type,
                f"{fjobstr} ({mjob.tags['source']}) seems miscentered! Skipping!",
                debug,
            )
        if imap[0, cent_est[0], cent_est[1]] < 0:
            return (
                fjobstr,
                fjob.tags["split"],
                map_type,
                f"{fjobstr} ({mjob.tags['source']}) looks like bad weather! Skipping!",
                debug,
            )

        vm = ivar[0] > 0
        ivar_smooth = enmap.smooth_gauss(
            ivar[0], 1.0 * fit["aman"].data_fwhm.to(u.radian).value / 2.355
        )
        irat = np.median(ivar_smooth[vm]) / np.std(ivar_smooth[vm])
        if irat < 2 * cfg.min_irat:
            imsk = np.abs(ivar_smooth - ivar[0]) > 3 * np.std(ivar[0, vm])
            ivar[:, imsk] = 0
            ivar_smooth = enmap.smooth_gauss(
                ivar[0], 1.0 * fit["aman"].data_fwhm.to(u.radian).value / 2.355
            )
            irat = np.median(ivar_smooth[vm]) / np.std(ivar_smooth[vm])

        prof = (fit["aman"].rprof.value - fit["aman"].bessel.off.value) / norm
        wn, cn, ps2d, lmap, hot, theta = bu.get_map_noise(
            imap[0],
            ivar[0],
            fit["aman"].r.to(u.radian).value,
            prof,
            ext_rad,
            cfg.n_lmin,
            cfg.n_lmax,
            np.deg2rad(cfg.nominal_fwhm[mjob.tags["band"]] / 60),
            True,
        )
        ivar[:, hot] = 0
        debug = (fjobstr, imap, ivar, hot, ps2d, lmap, cfg.n_lmin, cfg.n_lmax, wn, cn)
        ratio = cn / wn
        high_cn = np.log10(cn) > cfg.max_cn
        high_wn = np.log10(wn) > cfg.max_cn - 1
        rat_adj = max(0, -10 * (np.log10(wn) - cfg.max_cn - 1))
        cut = high_wn + high_cn + (ratio > cfg.corr_ratio_cut + rat_adj)
        if cut:
            return (
                fjobstr,
                fjob.tags["split"],
                map_type,
                f"{fjobstr} ({mjob.tags['source']}) has high noise ({np.round(ratio, 2)}, {np.round(np.log10(wn), 1)}, {np.round(np.log10(cn), 1)})! Skipping!",
                debug,
            )
        if np.abs(theta) > np.pi / 8:
            return (
                fjobstr,
                fjob.tags["split"],
                map_type,
                f"{fjobstr} ({mjob.tags['source']}) has high noise clocking ({np.round(np.rad2deg(theta), 2)})! Skipping!",
                debug,
            )
        rmsk = modrmap < ext_rad
        cut_frac = np.sum(ivar[0, rmsk] == 0) / np.sum(rmsk)
        if cut_frac > cfg.max_cut_pix_frac:
            return (
                fjobstr,
                fjob.tags["split"],
                map_type,
                f"{fjobstr} ({mjob.tags['source']}) has {np.round(cut_frac*100, 2)}% of pixels cut! Skipping!",
                debug,
            )

        if irat < cfg.min_irat:
            return (
                fjobstr,
                fjob.tags["split"],
                map_type,
                f"{fjobstr} ({mjob.tags['source']}) has a low ivar median-variance ratio: {np.round(irat, 2)}! Skipping!",
                debug,
            )

    np.nan_to_num(imap, copy=False, nan=0, posinf=0, neginf=0)
    np.nan_to_num(ivar, copy=False, nan=0, posinf=0, neginf=0)

    return fjobstr, fjob.tags["split"], map_type, (imap, ivar), debug


def stack_job(
    job,
    fits,
    fjobs,
    mjobdict,
    split_dict,
    cache,
    cfg,
    cfg_str,
    ctx_str,
    data_dir,
    plot_dir,
    tmap,
    twcs,
    pix_extent,
    ext_rad,
    pixsize,
    map_types,
    logger,
):
    data_dir_spl = os.path.join(
        data_dir,
        f"stacks{cfg.test_append}",
        job.tags["split"],
        job.tags["split_str"],
        job.tags["det_split"],
        f"{job.tags['epoch_start']}_{job.tags['epoch_end']}",
    )
    plot_dir_spl = os.path.join(
        plot_dir,
        f"stacks{cfg.test_append}",
        job.tags["split"],
        job.tags["split_str"],
        job.tags["det_split"],
        f"{job.tags['epoch_start']}_{job.tags['epoch_end']}",
    )
    debug_plot_dir = os.path.join(plot_dir_spl, "indiv")

    os.makedirs(data_dir_spl, exist_ok=True)
    os.makedirs(plot_dir_spl, exist_ok=True)
    os.makedirs(debug_plot_dir, exist_ok=True)
    job.mark_visited()
    # Make output maps
    jobdict = {
        map_type: {d: deepcopy(tmap) for d in ("map_stack", "ivar_stack", "map_ivar")}
        for map_type in map_types
    }

    zoom = 1
    if "band" in job.tags["split"]:
        b_idx = np.where("band" == np.array(job.tags["split"].split("+")))[0][0]
        band = job.tags["split_str"].split("+")[b_idx]
        zoom = 90 / float(band[1:])

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
    logger.log(
        25,
        "%d maps available",
        np.sum(smsk),
    )
    obslist = []

    def add_to_stack(map_type, imap, ivar):
        jobdict[map_type]["map_stack"].insert(imap * ivar, op=op)
        jobdict[map_type]["ivar_stack"].insert(ivar, op=op)
        jobdict[map_type]["map_ivar"].insert(imap**2 * ivar, op=op)

    futures = []
    num_fits = len(sfits)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for map_type in sorted(map_types):
            logger.info("Adding %s", f"{map_type}{' '*bool(map_type)}maps")
            for i, (fit, fjob) in enumerate(zip(sfits, sfjobs)):
                fjobstr = (
                    f"{fjob.tags['obs_id']}-"
                    f"{fjob.tags['wafer_slot']}-"
                    f"{fjob.tags['stream_id']}-"
                    f"{fjob.tags['array']}-"
                    f"{fjob.tags['band']}"
                )
                if fjobstr not in mjobdict:
                    logger.debug("Map job not found for %s", fjobstr)
                    continue
                key = (fjobstr, map_type, fjob.tags["split"])
                if map_type != "":
                    if fjobstr not in obslist:
                        logger.debug(
                            "%s not in obslist from stack of filter-bin maps. Skipping!",
                            fjobstr,
                        )
                        continue
                # Cache hit
                if key in cache:
                    fjobstr, split, map_type, maps, debug = cache[key]
                    if isinstance(maps, str):
                        logger.debug(
                            "%s marked as bad in cache (%d/%d)",
                            fjobstr,
                            i + 1,
                            num_fits,
                        )
                        if debug is not None:
                            plot_debug_map(debug_plot_dir, maps, *debug)
                        continue
                    logger.debug(
                        "Adding %s from cache (%d/%d)", fjobstr, i + 1, num_fits
                    )
                    if debug is not None:
                        plot_debug_map(debug_plot_dir, "Added to stack", *debug)
                    imap, ivar = maps
                    add_to_stack(map_type, imap, ivar)
                    if map_type == "":
                        obslist.append(fjobstr)
                    continue

                # Cache miss
                logger.debug(
                    "Submitting future for %s (%d/%d)", fjobstr, i + 1, num_fits
                )
                futures.append(
                    executor.submit(
                        process_one_map,
                        fit,
                        fjob,
                        map_type,
                        mjobdict,
                        data_dir,
                        cfg,
                        twcs,
                        pix_extent,
                        ext_rad,
                    )
                )

                if len(futures) >= max_workers * len(map_types) or (
                    i + 1 == num_fits and len(futures) > 0
                ):
                    for future in as_completed(futures):
                        result = future.result()
                        fjobstr, split, map_type, maps, debug = result
                        key = (fjobstr, map_type, split)
                        cache[key] = result
                        if isinstance(maps, str):
                            logger.warning("%s", maps)
                            if debug is not None:
                                plot_debug_map(debug_plot_dir, maps, *debug)
                            continue
                        logger.debug("Adding %s from future", fjobstr)
                        if debug is not None:
                            plot_debug_map(debug_plot_dir, "Added to stack", *debug)
                        imap, ivar = maps
                        add_to_stack(map_type, imap, ivar)
                        if map_type == "":
                            obslist.append(fjobstr)
                    logger.log(
                        25,
                        "%d/%d maps added (%d in obslist)",
                        i + 1,
                        num_fits,
                        len(obslist),
                    )
                    del futures
                    futures = []

    obslist = np.unique(obslist)
    if len(obslist) == 0:
        msg = "No maps made it into stack!"
        fail(job, ErrCode.NO_MAPS, msg, logger)
        return job

    logger.log(
        25,
        "%d maps in stack",
        len(obslist),
    )
    logger.info(obslist)
    # Divide weights and save
    for map_type in map_types:
        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            mv = deepcopy(jobdict[map_type]["map_stack"])
            jobdict[map_type]["map_stack"] /= jobdict[map_type]["ivar_stack"]
            jobdict[map_type]["map_ivar"] = (
                jobdict[map_type]["map_ivar"] / jobdict[map_type]["ivar_stack"]
                - (mv / jobdict[map_type]["ivar_stack"]) ** 2
            )
            jobdict[map_type]["map_ivar"] = 1 / jobdict[map_type]["map_ivar"]

            np.nan_to_num(
                jobdict[map_type]["map_stack"], copy=False, nan=0, posinf=0, neginf=0
            )
            np.nan_to_num(
                jobdict[map_type]["map_ivar"], copy=False, nan=0, posinf=0, neginf=0
            )
        for name in (
            "map_stack",
            "ivar_stack",
            "map_ivar",
        ):
            omap = cast(enmap.ndmap, jobdict[map_type][name])
            path = os.path.join(
                data_dir_spl,
                f"{job.tags['split_str']}_{job.tags['det_split']}_{job.tags['epoch_start']}_{job.tags['epoch_end']}{'_' * bool(map_type)}{map_type}_{name}.fits",
            )

            enmap.write_map(path, omap, "fits", allow_modify=True)
            set_tag(job, f"{map_type}{'_' * bool(map_type)}{name}", path)
            if "ivar" in name:
                continue
            posmap = np.rad2deg(omap.posmap()) * 3600
            for append, smap, z in [
                ("", omap, 1),
                ("_mask_zoom", omap, zoom),
            ]:
                plot_map_complete(
                    smap,
                    posmap,
                    pixsize,
                    min(cfg.extent * z, cfg.extent),
                    (0, 0),
                    plot_dir_spl,
                    f"{job.tags['split_str']} {job.tags['det_split']} {job.tags['epoch_start']} {job.tags['epoch_end']}{' ' * bool(map_type)}{map_type} {name}",
                    log_thresh=cfg.log_thresh,
                    append=name + append,
                    qrur=True,
                )

    set_tag(job, "config", cfg_str)
    set_tag(job, "context", ctx_str)
    set_tag(job, "message", "Success!")
    set_tag(job, "obslist", ",".join(obslist))
    job.jstate = cast(sqy.Column[str], jobdb.JState.done)
    return job


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
    {"map_mask_size": "mask_size"},
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
    f"{cfg.pointing_type}{(cfg.append != '') * '_'}{cfg.append}{(cfg.single_det) * '_single_det'}",
)
os.makedirs(plot_dir, exist_ok=True)
fpath = os.path.join(data_dir, f"beam_pars{cfg.test_append}.h5")
if myrank == 0:
    of_path_noa = os.path.join(data_dir, f"beam_pars.h5")
    if os.path.isfile(of_path_noa) and cfg.copy_fits_test:
        shutil.copyfile(of_path_noa, fpath)
jdb = make_jobdb(comm, data_dir, cfg.test_append)

# Load fits
logger.info("Loading map metadata and fits")

all_fits = None
fjobs = None
mjobdict = None
if myrank == 0:
    mjobdict = {
        f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['array']}-{job.tags['band']}": job
        for job in jdb.get_jobs(jclass="beam_map", jstate="done")
    }
    logger.info("Map jobs loaded")
    fjobs = np.array(jdb.get_jobs(jclass="fit_map", jstate="done"))
    logger.info("Fit jobs loaded")
    all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs.tolist())
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
    fjobs = fjobs[msk]
    logger.info("Fits filtered")
logger.info("Broadcasting")
all_fits = comm.bcast(all_fits)
fjobs = comm.bcast(fjobs)
mjobdict = comm.bcast(mjobdict)

logger.info("%d maps to add", len(fjobs))
if len(fjobs) == 0:
    sys.exit(0)

# Det splits
det_split_names = ["full"] + cfg.det_splits

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
        ctx=ctx,
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

# Make template map
ext_rad = np.deg2rad(cfg.extent / 3600)
pix_extent = int(2 * ext_rad // cfg.res)
twcs = enmap.wcsutils.build(
    [0, 0],
    res=np.rad2deg(cfg.res),
    shape=(pix_extent, pix_extent),
    system="tan",
    rowmajor=True,
)
tmap = enmap.zeros((3, pix_extent, pix_extent), twcs)
map_types = ("",)  # , "resid")

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
        job_maps = [get_job_maps(job, all_fits, split_dict) for job in all_jobs]
        assignments = distribute_jobs(all_jobs, job_maps, nproc, logger)
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

max_len = get_cache_max_len(tmap, nproc)
cache = ExpiringDict(max_len=max_len, max_age_seconds=3600)
logger.info("Making cache of size %d", max_len)
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
        pending_job = stack_job(
            job=job,
            fits=fits,
            fjobs=fjobs_local,
            mjobdict=mjobdict,
            split_dict=split_dict,
            cache=cache,
            cfg=cfg,
            cfg_str=cfg_str,
            ctx_str=ctx_str,
            data_dir=data_dir,
            plot_dir=plot_dir,
            tmap=tmap,
            twcs=twcs,
            pix_extent=pix_extent,
            ext_rad=ext_rad,
            pixsize=pixsize,
            map_types=map_types,
            logger=logger,
        )
    logger.log(25, "Done with stack")

comm.barrier()
sys.stdout.flush()
logger.info("Done with all stacks")
