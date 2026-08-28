import os
from collections import defaultdict
from functools import partial
from typing import cast

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soma.beams as sb
import sqlalchemy as sqy
from astropy import constants as const
from astropy import units as u
from healpy.sphtfunc import beam2bl
from pixell import enmap
from sotodlib.core import AxisManager
from sotodlib.site_pipeline import jobdb
from sotodlib.site_pipeline.jobdb import Job

import lat_beams.fitting.map.bessel as fb
import lat_beams.fitting.map.gauss as fg
from lat_beams.beam_utils import (
    get_fwhm_radial_bins,
    radial_profile,
    radial_profile_lin,
)
from lat_beams.fitting.map.base import make_guess
from lat_beams.plotting import auto_relplot, plot_map_complete
from lat_beams.utils import (
    ErrCode,
    fail,
    get_args_cfg,
    init_log,
    make_jobdb,
    set_tag,
    setup_cfg,
    setup_jobs,
    setup_paths,
)

palette = sns.color_palette("colorblind")
sns.set_palette(palette)


def get_jobdict(jdb):
    jobdict = {
        f"{job.tags['split']}-{job.tags['split_str']}-{job.tags['det_split']}-{job.tags['epoch_start']}-{job.tags['epoch_end']}": job
        for job in jdb.get_jobs(jclass="fit_stacks")
    }
    return jobdict


def get_jobit(jdb, cfg):
    det_splits = ["full"] + cfg.det_splits
    epoch_strings = [f"{start}-{end}" for start, end in cfg.epochs]
    splits = cfg.split_by
    maplist = jdb.get_jobs(jclass="stack_maps", jstate="done", locked=False)
    to_ret = [
        job
        for job in maplist
        if (
            job.tags["det_split"] in det_splits
            and job.tags["split"] in splits
            and f"{job.tags['epoch_start']}-{job.tags['epoch_end']}" in epoch_strings
        )
    ]
    return to_ret


def get_jobstr(job):
    return f"{job.tags['split']}-{job.tags['split_str']}-{job.tags['det_split']}-{job.tags['epoch_start']}-{job.tags['epoch_end']}"


def get_tags(info):
    tags = {
        "split": info.tags["split"],
        "split_str": info.tags["split_str"],
        "det_split": info.tags["det_split"],
        "epoch_start": info.tags["epoch_start"],
        "epoch_end": info.tags["epoch_end"],
        "errcode": 0,
        "message": "",
        "data_profile": "",
        "model_profile": "",
        "data_window": "",
        "model_window": "",
        "model_map": "",
        "resid_map": "",
        "config": "",
    }
    return tags


def view_TQU(imap):
    padded = imap
    if len(imap) == 1:
        padded = enmap.zeros((3,) + imap.shape[1:], imap.wcs)
        padded[0][:] = imap[0][:]
    return padded


def downsample(x, *ys, max_points=2000):
    x = np.asarray(x)
    if len(x) <= max_points:
        return (x, *[np.asarray(y) for y in ys])
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return (x[idx], *[np.asarray(y)[idx] for y in ys])


def plot_model_maps(
    model,
    imap,
    off,
    eta0,
    xi0,
    posmap,
    pixsize,
    extent,
    plot_dir,
    job,
    log_thresh,
    n_multipoles,
):
    plot_dir_spl = os.path.join(
        plot_dir,
        "stacks",
        job.tags["split"],
        job.tags["split_str"],
        job.tags["det_split"],
        f"{job.tags['epoch_start']}_{job.tags['epoch_end']}",
    )
    os.makedirs(plot_dir_spl, exist_ok=True)
    ypix, xpix = enmap.sky2pix(imap.shape, imap.wcs, ([[eta0], [xi0]]))
    y0, x0 = float(ypix[0]), float(xpix[0])

    maps = [(model - off, "model"), (imap - model, "model_resid"), (imap - off, "")]
    for omap, name in maps:
        if "resid" not in name:
            modes = sb.beam_modes(omap, mmax=n_multipoles + 2, center=(y0, x0))  # type: ignore
            mode_data = {
                "ell": np.tile(modes["ell"], n_multipoles),
                "rho": np.concatenate(
                    [modes["rho"][m] for m in range(1, n_multipoles + 1)]
                ),
                "mode": np.repeat(
                    [f"m={m}" for m in np.arange(1, n_multipoles + 1)],
                    len(modes["ell"]),
                ),
            }

            plt.close()
            sns.lineplot(
                data=mode_data,
                x="ell",
                y="rho",
                hue="mode",
            )
            plt.yscale("log")
            plt.xlim(0, modes["lmax"])
            plt.xlabel(r"$\ell$")
            plt.ylabel(r"$b_m(\ell)/b_0(\ell)$")
            label = "_T_stack" + "_" * (len(name) > 0) + name
            title = f"{job.tags['split_str']} {job.tags['det_split']} {job.tags['epoch_start']} {job.tags['epoch_end']}"
            plt.title(f"{title}{label.replace('_', ' ')}")
            plt.savefig(
                os.path.join(
                    plot_dir_spl, f"{title.replace(' ', '_')}_modes{label}.png"
                ),
                bbox_inches="tight",
            )
            plt.close()

        if name == "":
            continue

        plot_map_complete(
            omap,
            posmap,
            pixsize,
            extent,
            (0, 0),
            plot_dir_spl,
            f"{job.tags['split_str']} {job.tags['det_split']} {job.tags['epoch_start']} {job.tags['epoch_end']}",
            comps="T",
            log_thresh=log_thresh,
            append="stack_" + name,
            units='"',
            lognorm=1,
        )


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
pixsize = 3600 * np.rad2deg(cfg.res)

# Setup folders
plot_dir, data_dir = setup_paths(
    cfg.root_dir,
    "beams",
    cfg.tel,
    f"{cfg.pointing_type}{(cfg.append!='')*'_'}{cfg.append}{(cfg.single_det)*'_single_det'}",
)
out_file = os.path.join(data_dir, "stacks", "beam_pars.h5")

# Make highres template
ext_rad = np.deg2rad(cfg.extent_highres / 3600)
pix_extent = 2 * int(cfg.extent_highres // pixsize)
# rowmajor = True here to match sotodlib
twcs = enmap.wcsutils.build(
    [0, 0],
    res=np.rad2deg(cfg.res),
    shape=(pix_extent, pix_extent),
    system="tan",
    rowmajor=True,
)
posmap_highres = enmap.posmap((pix_extent, pix_extent), twcs)

# Get det splits
det_split_names = ["full"] + cfg.det_splits

# Setup logger
logger = init_log()

# Get jobs
jdb = make_jobdb(None, data_dir)
jdb, all_jobs = setup_jobs(
    None,
    data_dir,
    "fit_stacks",
    get_jobdict,
    partial(
        get_jobit,
        cfg=cfg,
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
stack_jobs = {
    f"{job.tags['split']}-{job.tags['split_str']}-{job.tags['det_split']}-{job.tags['epoch_start']}-{job.tags['epoch_end']}": job
    for job in jdb.get_jobs(jclass="stack_maps")
}

# Structure here is split -> spl_str -> epoch -> det_split
jobdict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for job in all_jobs:
    jobdict[job.tags["split"]][job.tags["split_str"]][
        (job.tags["epoch_start"], job.tags["epoch_end"])
    ][job.tags["det_split"]] = job


# Profiler setup
profiler = None
if args.profile:
    from pyinstrument import Profiler

    profiler = Profiler()
    print("Running in profiler mode! Only one job will be run!")
    profiler.start()

if args.plot_only:
    logger.info(
        "Running in plot only mode. Profiles and windows will be loaded and replotted. Model and residuals will not."
    )

# Loop through splits
for split in jobdict.keys():
    i = 0
    logger.info("Splitting by %s", split)
    data_dir_spl = os.path.join(data_dir, "stacks", split)
    if "band" not in split:
        logger.error("Can't fit a non band split!")
        continue
    band_idx = np.where(np.array(split.split("+")) == "band")[0][0]
    to_plot_r = {s: [] for s in split.split("+")}
    to_plot_r["r"] = []
    to_plot_r["rprof"] = []
    to_plot_r["rprof_err"] = []
    to_plot_r["epoch"] = []
    to_plot_r["dataset"] = []
    to_plot_l = {s: [] for s in split.split("+")}
    to_plot_l["ell"] = []
    to_plot_l["window"] = []
    to_plot_l["window_err"] = []
    to_plot_l["epoch"] = []
    to_plot_l["dataset"] = []
    prof_plot_dir = os.path.join(plot_dir, "stack_profiles", split)
    os.makedirs(prof_plot_dir, exist_ok=True)
    for spl in jobdict[split].keys():
        plot_dir_spl = os.path.join(plot_dir, "stacks", split, spl)
        band = spl.split("+")[band_idx]
        fscale_fac = 90.0 / float(band[1:]) if cfg.apply_fscale else 1
        band_mask_size = np.deg2rad(fscale_fac * cfg.mask_size)
        if i > 0:
            continue
        for epoch in jobdict[split][spl].keys():
            plot_dir_epc = os.path.join(plot_dir_spl, f"{epoch[0]}_{epoch[1]}")
            for det_split in jobdict[split][spl][epoch].keys():
                j = jobdict[split][spl][epoch][det_split]
                with jdb.session_scope() as session:
                    job = session.get(Job, j.id)
                    session.expunge(job)

                if job is None:
                    raise ValueError("job is None!")

                jobstr = f"{job.tags['split']}-{job.tags['split_str']}-{job.tags['det_split']}-{job.tags['epoch_start']}-{job.tags['epoch_end']}"

                if args.plot_only:
                    logger.info("Loading %s", jobstr)
                    prof_dir = os.path.join(data_dir, "stack_profiles", split, spl)
                    h5_file = os.path.join(
                        prof_dir,
                        f"beam_profiles_{spl}_{det_split}_{epoch[0]}_{epoch[1]}.h5",
                    )
                    if not os.path.isfile(h5_file):
                        logger.warning(
                            "Missing saved beam profile %s. Skipping.", h5_file
                        )
                        continue
                    prof = AxisManager.load(h5_file)
                    stack_job = stack_jobs[jobstr]
                    map_path = stack_job.tags["map_stack"]
                    fit_path = os.path.join(
                        split,
                        spl,
                        f"{det_split}_{epoch[0]}_{epoch[1]}",
                    )
                    if not os.path.isfile(map_path):
                        logger.warning(
                            "Missing stacked map %s. Skipping maps.", map_path
                        )
                        continue
                    fit = AxisManager.load(out_file, fit_path)

                    r = np.asarray(prof.data_r)
                    rprof = np.asarray(prof.data_profile)
                    rerr = np.asarray(prof.data_profile_sigma)
                    mr = np.asarray(prof.r)
                    mprof = np.asarray(prof.profile)
                    profile_sigma = np.asarray(
                        prof.profile_sigma
                    )  # * fit.bessel.sigma2
                    ells = np.asarray(prof.ell)
                    bl = np.asarray(prof.data_window)
                    mbl = np.asarray(prof.bl)
                    bl_sigma = np.asarray(prof.bl_sigma)  # * fit.bessel.sigma2

                    mmsk = mr < 1.2 * 3600 * np.rad2deg(band_mask_size)
                    dmsk = r < 3600 * np.rad2deg(band_mask_size)

                    epoch_name = f"{det_split}_{epoch[0]}_{epoch[1]}"

                    to_plot_r["r"] += mr[mmsk].tolist() + r[dmsk].tolist()
                    to_plot_r["rprof"] += mprof[mmsk].tolist() + rprof[dmsk].tolist()
                    to_plot_r["rprof_err"] += (
                        profile_sigma[mmsk].tolist() + np.zeros(np.sum(dmsk)).tolist()
                    )
                    to_plot_r["epoch"] += [epoch_name] * (np.sum(mmsk) + np.sum(dmsk))
                    to_plot_r["dataset"] += ["model"] * np.sum(mmsk) + [
                        "data"
                    ] * np.sum(dmsk)

                    plot_ell, plot_mbl, plot_bl, plot_bls = downsample(
                        ells,
                        mbl,
                        bl,
                        bl_sigma,
                    )
                    n = len(plot_ell)

                    to_plot_l["ell"] += plot_ell.tolist() + plot_ell.tolist()
                    to_plot_l["window"] += plot_mbl.tolist() + plot_bl.tolist()
                    to_plot_l["window_err"] += plot_bls.tolist() + np.zeros(n).tolist()
                    to_plot_l["epoch"] += [epoch_name] * (2 * n)
                    to_plot_l["dataset"] += ["model"] * n + ["data"] * n

                    for sc, si in zip(split.split("+"), spl.split("+")):
                        to_plot_r[sc] += [si] * (np.sum(mmsk) + np.sum(dmsk))
                        to_plot_l[sc] += [si] * (2 * n)

                    if jobstr not in stack_jobs:
                        logger.warning(
                            "Stack job missing for %s. Skipping maps.", jobstr
                        )
                        continue
                    imap = cast(enmap.ndmap, enmap.read_map(map_path)[0])
                    imap = enmap.unapply_window(imap, order=0)
                    model = cast(
                        enmap.ndmap,
                        enmap.read_map(
                            os.path.join(
                                data_dir,
                                "stacks",
                                job.tags["split"],
                                job.tags["split_str"],
                                job.tags["det_split"],
                                f"{job.tags['epoch_start']}_{job.tags['epoch_end']}",
                                f"{job.tags['split_str']}_{job.tags['det_split']}_"
                                f"{job.tags['epoch_start']}_{job.tags['epoch_end']}_model.fits",
                            )
                        )
                        + fit.bessel.off.value,
                    )
                    posmap = np.rad2deg(imap.posmap()) * 3600
                    plot_model_maps(
                        model,
                        imap,
                        fit.bessel.off.value,
                        fit.bessel.eta0.value,
                        fit.bessel.xi0.value,
                        posmap,
                        pixsize,
                        cfg.extent,
                        plot_dir,
                        job,
                        cfg.log_thresh,
                        cfg.n_multipoles,
                    )
                    continue

                job.mark_visited()

                logger.info("Fitting %s", jobstr)
                if jobstr not in stack_jobs:
                    msg = "Stack job missing"
                    fail(job, ErrCode.NO_JOB, msg, logger)
                    continue
                stack_job = stack_jobs[jobstr]

                map_path = stack_job.tags["map_stack"]
                ivar_path = stack_job.tags["map_ivar"]
                # ivar_naive_path = stack_job.tags["ivar_stack"]

                if not os.path.isfile(map_path) or not os.path.isfile(ivar_path):
                    msg = "Map missing"
                    fail(job, ErrCode.MAP_MISSING, msg, logger)
                    continue
                imap = enmap.read_map(map_path)[0]  # Just T for now
                ivar = enmap.read_map(ivar_path)[0]  # Just T for now
                # ivar_naive = enmap.read_map(ivar_naive_path)[0]  # Just T for now
                imap = enmap.unapply_window(imap, order=0)
                imap = cast(enmap.ndmap, imap)
                ivar = cast(enmap.ndmap, ivar)
                # ivar_naive = cast(enmap.ndmap, ivar_naive)
                posmap = imap.posmap()

                # Setup aman for output
                aman = AxisManager()

                # Fit a gaussian to start
                # TODO: Process models to produce solid angle and stuff
                guess = make_guess(
                    amp=1,
                    fwhm_xi=np.deg2rad(cfg.nominal_fwhm[band] / 60.0),
                    fwhm_eta=np.deg2rad(cfg.nominal_fwhm[band] / 60.0),
                    xi0=0,
                    eta0=0,
                    phi=0,
                    off=0,
                )
                gauss_params, model = fg.fit_gauss_map(
                    imap,
                    ivar,
                    posmap,
                    guess,
                    "pW",
                    cfg.sym_gauss,
                    min(band_mask_size, 1.5 * guess.fwhm_xi),
                )
                if gauss_params is None or model is None:
                    msg = "Gauss fit failed!"
                    fail(job, ErrCode.FIT_FAILED, msg, logger)
                    continue
                if abs(gauss_params.amp.value - 1) >= 0.2:  # type: ignore
                    msg = "Gauss fit looks bad!"
                    fail(job, ErrCode.FIT_FAILED, msg, logger)
                    continue
                aman.wrap("gauss", gauss_params)
                for to_parent in ["amp", "off", "xi0", "eta0"]:
                    aman.wrap(to_parent, gauss_params[to_parent])

                # Get FWHM from data
                c = np.unravel_index(
                    np.argmin(posmap[0] ** 2 + posmap[1] ** 2, axis=None),
                    posmap[0].shape,
                )
                cent = (int(c[1]), int(c[0]))
                rprof = radial_profile(imap, cent)
                rerr = radial_profile(ivar, cent, False)
                r = np.linspace(0, len(rprof), len(rprof)) * pixsize
                rmsk = r < 3 * 60 * cfg.nominal_fwhm[band] / 2.355
                data_fwhm = (
                    get_fwhm_radial_bins(r[rmsk], rprof[rmsk], interpolate=True)
                    * u.arcsec
                )
                if np.isnan(data_fwhm):
                    msg = "Data FWHM is bad! Skipping!"
                    fail(job, ErrCode.FWHM_TOL, msg, logger)
                    continue
                aman.wrap("data_fwhm", data_fwhm)
                aman.wrap("r", r * u.arcsec)
                aman.wrap("rprof", rprof * u.pW)

                # Now fit the bessel beam
                bessel_beam_params, model = fb.fit_bessel_map(
                    imap,
                    ivar,
                    posmap,
                    gauss_params,
                    "pW",
                    cfg.n_bessel,
                    cfg.n_multipoles,
                    cfg.aperature,
                    const.c / (float(band[1:]) * u.GHz),  # type: ignore
                    band_mask_size,
                    cfg.bessel_wing_n_sigma * fscale_fac,
                    cfg.skip_multipoles,
                    calc_cov=True,
                    n_opt_pixels=int(fscale_fac * 20000),
                )
                if bessel_beam_params is None or model is None:
                    msg = "Bessel fit failed!"
                    fail(job, ErrCode.FIT_FAILED, msg, logger)
                    continue
                aman.wrap("bessel", bessel_beam_params)
                aman.wrap("final_model", "bessel")
                logger.info("Bessel fit complete")

                # Make and save a higher resolution profile
                prof_dir = os.path.join(data_dir, "stack_profiles", split, spl)
                os.makedirs(prof_dir, exist_ok=True)
                prof_cov, model_highres = fb.bessel_profile_covariance(
                    aman.bessel,
                    posmap_highres,
                    cfg.lmax,
                    cfg.cov_modes,
                    pix_extent // 2,
                )
                logger.info("Bessel cov complete")
                mr = np.asarray(prof_cov.r)
                mprof = np.asarray(prof_cov.profile)
                profile_modes = np.asarray(prof_cov.profile_modes)
                mprofile = np.column_stack((mr, mprof, profile_modes))
                path = os.path.join(
                    prof_dir,
                    f"model_profile_{spl}_{det_split}_{epoch[0]}_{epoch[1]}.txt",
                )
                np.savetxt(
                    path,
                    mprofile,
                )
                set_tag(job, "model_profile", path)

                ells = np.asarray(prof_cov.ell)
                mbl = np.asarray(prof_cov.bl)
                bl_modes = np.asarray(prof_cov.bl_modes)
                mwindow = np.column_stack((ells, mbl, bl_modes))
                path = os.path.join(
                    prof_dir,
                    f"model_window_{spl}_{det_split}_{epoch[0]}_{epoch[1]}.txt",
                )
                np.savetxt(
                    path,
                    mwindow,
                    header="ell bl "
                    + " ".join(f"error_mode_{i}" for i in range(bl_modes.shape[1])),
                )
                set_tag(job, "model_window", path)

                path = os.path.join(
                    prof_dir,
                    f"model_profile_{spl}_{det_split}_{epoch[0]}_{epoch[1]}.txt",
                )
                np.savetxt(path, mprofile)
                set_tag(job, "model_profile", path)
                mwindow = np.column_stack(
                    (prof_cov["ell"], prof_cov["bl"], prof_cov["bl_modes"])
                )
                path = os.path.join(
                    prof_dir,
                    f"model_window_{spl}_{det_split}_{epoch[0]}_{epoch[1]}.txt",
                )
                np.savetxt(path, mwindow)
                set_tag(job, "model_window", path)

                # Add data profile and window
                data_r, data_profile, _ = radial_profile_lin(
                    imap,
                    posmap,
                    xi0=aman.bessel.xi0.value,
                    eta0=aman.bessel.eta0.value,
                    r=np.deg2rad(mr[mr <= r.max()] / 3600),
                )
                data_profile -= aman.bessel.off.value
                data_profile /= data_profile[0]
                data_r = 3600 * np.rad2deg(data_r)
                rprofile = np.column_stack(
                    (
                        data_r,
                        data_profile,
                        np.interp(data_r, r, rerr),
                    )
                )
                path = os.path.join(
                    prof_dir, f"profile_{spl}_{det_split}_{epoch[0]}_{epoch[1]}.txt"
                )
                np.savetxt(path, rprofile)
                set_tag(job, "data_profile", path)
                bl = beam2bl(
                    rprofile[:, 1], np.deg2rad(rprofile[:, 0] / 3600), cfg.lmax
                )
                ells = np.arange(cfg.lmax + 1)
                window = np.column_stack((ells, bl))
                path = os.path.join(
                    prof_dir, f"window_{spl}_{det_split}_{epoch[0]}_{epoch[1]}.txt"
                )
                np.savetxt(path, window)
                set_tag(job, "data_window", path)

                prof_cov.wrap("data_r", rprofile[:, 0])
                prof_cov.wrap("data_profile", rprofile[:, 1])
                prof_cov.wrap("data_profile_sigma", rerr)
                prof_cov.wrap("data_window", window[:, 1])
                prof_cov.wrap("data_ell", ells)

                # Save file
                h5_file = os.path.join(
                    prof_dir,
                    f"beam_profiles_{spl}_{det_split}_{epoch[0]}_{epoch[1]}.h5",
                )
                prof_cov.save(
                    h5_file,
                    overwrite=True,
                )

                # Save for plots
                mmsk = mr < 1.2 * 3600 * np.rad2deg(band_mask_size)
                dmsk = data_r < 3600 * np.rad2deg(band_mask_size)

                to_plot_r["r"] += mr[mmsk].tolist() + data_r[dmsk].tolist()
                to_plot_r["rprof"] += mprof[mmsk].tolist() + rprofile[dmsk, 1].tolist()
                to_plot_r["rprof_err"] += (
                    np.asarray(prof_cov.profile_sigma)[mmsk].tolist()
                    + np.zeros(np.sum(dmsk)).tolist()
                )
                to_plot_r["epoch"] += [f"{det_split}_{epoch[0]}_{epoch[1]}"] * (
                    np.sum(mmsk) + np.sum(dmsk)
                )
                to_plot_r["dataset"] += ["model"] * np.sum(mmsk) + ["data"] * np.sum(
                    dmsk
                )

                plot_ell, plot_mbl, plot_bl, plot_bls = downsample(
                    ells,
                    mbl,
                    window[:, 1],
                    prof_cov.bl_sigma,
                )
                n = len(plot_ell)

                to_plot_l["ell"] += plot_ell.tolist() + plot_ell.tolist()
                to_plot_l["window"] += plot_mbl.tolist() + plot_bl.tolist()
                to_plot_l["window_err"] += plot_bls.tolist() + np.zeros(n).tolist()
                to_plot_l["epoch"] += [f"{det_split}_{epoch[0]}_{epoch[1]}"] * (2 * n)
                to_plot_l["dataset"] += ["model"] * n + ["data"] * n

                for sc, si in zip(split.split("+"), spl.split("+")):
                    to_plot_r[sc] += [si] * (np.sum(mmsk) + np.sum(dmsk))
                    to_plot_l[sc] += [si] * (2 * n)

                # Save and plot maps
                aman.save(
                    out_file,
                    os.path.join(split, spl, f"{det_split}_{epoch[0]}_{epoch[1]}"),
                    overwrite=True,
                )
                posmap = np.rad2deg(posmap) * 3600
                resid = imap - model
                data_dir_spl = os.path.join(
                    data_dir,
                    "stacks",
                    job.tags["split"],
                    job.tags["split_str"],
                    job.tags["det_split"],
                    f"{job.tags['epoch_start']}_{job.tags['epoch_end']}",
                )
                os.makedirs(data_dir_spl, exist_ok=True)
                for omap, name in [
                    (model - aman.bessel.off.value, "model"),
                    (resid, "model_resid"),
                ]:
                    path = os.path.join(
                        data_dir_spl,
                        f"{job.tags['split_str']}_{job.tags['det_split']}_"
                        f"{job.tags['epoch_start']}_{job.tags['epoch_end']}_{name}.fits",
                    )
                    enmap.write_map(path, omap, "fits", allow_modify=True)

                plot_model_maps(
                    model,
                    imap,
                    aman.bessel.off.value,
                    aman.bessel.eta0.value,
                    aman.bessel.xi0.value,
                    posmap,
                    pixsize,
                    cfg.extent,
                    plot_dir,
                    job,
                    cfg.log_thresh,
                    cfg.n_multipoles,
                )
                logger.info("Saved")
                set_tag(job, "message", "Success!")
                job.jstate = cast(sqy.Column[str], jobdb.JState.done)
                with jdb.session_scope() as session:
                    session.merge(job)
                    session.commit()
        if args.profile:
            break

    # Plot profiles and windows
    logger.info("Plotting %s", split)
    row = "tube_slot" if "tube_slot" in split else None
    col = "band" if "band" in split else None
    combine = (
        [("tube_slot", "band")] if ("tube_slot" in split and "band" in split) else None
    )
    plt.close()
    to_plot_r = {str(key): np.array(value) for key, value in to_plot_r.items()}
    plot = auto_relplot(
        to_plot_r,
        x="r",
        y="rprof",
        errorbar="rprof_err",
        kind="line",
        estimator=None,
        style="dataset",
        hue="epoch",
        row=row,
        col=col,
        combine=combine,
        facet_kws={"sharey": True, "sharex": False},
    )
    plot.set_axis_labels('r (")', r"Beam Profile")
    plot.set(yscale="log")
    plot.figure.suptitle(f"Beam Profile by {split}", wrap=True)
    plt.subplots_adjust(top=(1 - 0.25 / len(plot.axes)))
    plt.savefig(
        os.path.join(prof_plot_dir, f"profile_{split}.png"), bbox_inches="tight"
    )
    msk = to_plot_r["dataset"] != "data"
    to_plot_r = {str(key): np.array(value)[msk] for key, value in to_plot_r.items()}
    to_plot_r["rprof"] = 100 * to_plot_r["rprof_err"] / to_plot_r["rprof"]
    del to_plot_r["rprof_err"]
    plot = auto_relplot(
        to_plot_r,
        x="r",
        y="rprof",
        kind="line",
        estimator=None,
        style="dataset",
        hue="epoch",
        row=row,
        col=col,
        combine=combine,
        facet_kws={"sharey": True, "sharex": False},
    )
    plot.set_axis_labels('r (")', r"Beam Profile Error (%)")
    # plot.set(ylim=(0, 15))
    plot.figure.suptitle(f"Beam Profile Error by {split}", wrap=True)
    plt.subplots_adjust(top=(1 - 0.25 / len(plot.axes)))
    plt.savefig(
        os.path.join(prof_plot_dir, f"profile_err_{split}.png"), bbox_inches="tight"
    )
    plt.close()

    to_plot_l = {str(key): np.array(value) for key, value in to_plot_l.items()}
    plot = auto_relplot(
        to_plot_l,
        x="ell",
        y="window",
        errorbar="window_err",
        kind="line",
        estimator=None,
        style="dataset",
        hue="epoch",
        row=row,
        col=col,
        combine=combine,
        facet_kws={"sharey": False, "sharex": True},
    )
    plot.set_axis_labels(r"$\ell$", r"Beam Window Function ($B_{\ell}^{T}$)")
    ls = ""
    for ax in plot.axes.flat:
        for line in ax.lines:
            if "model" in line.get_label():
                ls = line.get_linestyle()
                break
    for ax in plot.axes.flat:
        if ls == "":
            continue
        ys = []
        for line in ax.lines:
            if line.get_linestyle() != ls:
                continue
            d = line.get_data()[1]
            if len(d) == 0:
                continue
            ys += [np.max(d)]
        if len(ys) > 0:
            ax.set_ylim(0, 1.1 * np.max(ys))
    plot.figure.suptitle(f"Beam Window by {split}", wrap=True)
    plt.subplots_adjust(top=(1 - 0.25 / len(plot.axes)))
    plt.savefig(os.path.join(prof_plot_dir, f"window_{split}.png"), bbox_inches="tight")
    msk = to_plot_l["dataset"] != "data"
    to_plot_l = {str(key): np.array(value)[msk] for key, value in to_plot_l.items()}
    to_plot_l["window"] = 100 * to_plot_l["window_err"] / to_plot_l["window"]
    del to_plot_l["window_err"]
    plot = auto_relplot(
        to_plot_l,
        x="ell",
        y="window",
        kind="line",
        estimator=None,
        style="dataset",
        hue="epoch",
        row=row,
        col=col,
        combine=combine,
        facet_kws={"sharey": False, "sharex": True},
    )
    plot.set_axis_labels(r"$\ell$", r"Beam Window Function Error (%)")
    plot.set(ylim=(0, 2))
    plot.figure.suptitle(f"Beam Window Error by {split}", wrap=True)
    plt.subplots_adjust(top=(1 - 0.25 / len(plot.axes)))
    plt.savefig(
        os.path.join(prof_plot_dir, f"window_err_{split}.png"), bbox_inches="tight"
    )

logger.info("Done!")
if args.profile and profiler is not None:
    profiler.stop()
    profiler.write_html(f"profile.html")
