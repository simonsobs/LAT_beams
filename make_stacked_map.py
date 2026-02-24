import os
import sys

import astropy.units as u
import numpy as np
from pixell import enmap, reproject
from sotodlib.core import Context

from lat_beams import beam_utils as bu
from lat_beams.plotting import plot_map_complete
from lat_beams.utils import get_args_cfg, make_jobdb


def view_TQU(imap):
    padded = imap
    if len(imap) == 1:
        padded = enmap.zeros((3,) + imap.shape[1:], imap.wcs)
        padded[0][:] = imap[0][:]
    return padded


nominal_fwhm = {"f090": 2.0, "f150": 1.3, "f220": 0.95, "f280": 0.83}  # arcmin

args, cfg = get_args_cfg()

# Get some global setting
epochs = cfg.get("epochs", [(0, 2e10)])
pointing_type = cfg.get("pointing_type", "pointing_model")
append = cfg["append"] = cfg.get("append", "")
nominal_fwhm = cfg.get("nominal_fwhm", nominal_fwhm)
split_by = cfg.get(
    "split_by", ["band", "tube_slot+band", "source+band", "source+tube_slot+band"]
)
extent = cfg["extent"] = cfg.get("extent", 600)
mask_size = cfg.get("mask_size", 0.1)
mask_size *= u.degree
res = cfg["res"] = cfg.get("res", (10.0 / 3600.0) * np.pi / 180.0)
miscenter_thresh = cfg["miscenter_thresh"] = cfg.get("miscenter_thresh", 5)
pixsize = 3600 * np.rad2deg(res)
log_thresh = cfg["log_thresh"] = cfg.get("log_thresh", 1e-3)
ctx = Context(cfg.get("context", "/so/metadata/lat/contexts/smurf_detcal.yaml"))
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
op = np.ndarray.__iadd__

# Setup folders
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "beams/lat")
plot_dir = os.path.join(
    root_dir,
    "plots",
    project_dir,
    f"{pointing_type}{(append!="")*'_'}{append}",
    "stacks",
)
data_dir = os.path.join(
    root_dir,
    "data",
    project_dir,
    f"{pointing_type}{(append!="")*'_'}{append}",
)
os.makedirs(data_dir, exist_ok=True)
fpath = os.path.join(data_dir, "beam_pars.h5")
jdb = make_jobdb(None, data_dir)

# Get jobs
mjobdict = {
    f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['band']}": job
    for job in jdb.get_jobs(jclass="beam_map", jstate="done")
}
fjobdict = {
    f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['band']}": job
    for job in jdb.get_jobs(jclass="fit_map", jstate="done")
}
alljobstr = list(set(list(mjobdict.keys())) & set(list(fjobdict.keys())))
mjobs = np.array([mjobdict[jobstr] for jobstr in alljobstr])
fjobs = np.array([fjobdict[jobstr] for jobstr in alljobstr])

print(f"{len(alljobstr)} maps to add")
if len(alljobstr) == 0:
    sys.exit(0)

# Load fits
all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs)
snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
solid_angle = bu.get_fit_vec(all_fits, "data_solid_angle_corr")
msk = snr > 100
msk *= solid_angle > 0
all_fits = all_fits[msk]
mjobs = mjobs[msk]
fjobs = fjobs[msk]

# Make template map
ext_rad = np.deg2rad(extent / 3600)
pix_extent = 2 * int(extent // pixsize)
# rowmajor = True here to match sotodlib
twcs = enmap.wcsutils.build(
    [0, 0],
    res=np.rad2deg(res),
    shape=(pix_extent, pix_extent),
    system="tan",
    rowmajor=True,
)
tmap = enmap.zeros((3, pix_extent, pix_extent), twcs)
[[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(tmap.corners(corner=False))
plt_extent = (ra_min, ra_max, dec_min, dec_max)

if args.plot_only:
    print("Running in plot only mode!")

# Loop through splits
for split in split_by:
    print(f"Splitting by {split}")
    split_vec = bu.get_split_vec(all_fits, split, ctx)
    for spl in np.unique(split_vec):
        data_dir_spl = os.path.join(data_dir, "stacks", split, spl)
        plot_dir_spl = os.path.join(plot_dir, split, spl)
        os.makedirs(data_dir_spl, exist_ok=True)
        os.makedirs(plot_dir_spl, exist_ok=True)

        smsk = split_vec == spl
        sfits = all_fits[smsk]
        smjobs = mjobs[smsk]
        sfjobs = fjobs[smsk]
        fwhm_exp = np.array([nominal_fwhm[band] for band in sfits["band"]]) * u.arcmin
        sang_exp = (2 * np.pi * (fwhm_exp.to(u.radian) / 2.355) ** 2).to(u.sr)
        data_fwhm = bu.get_fit_vec(sfits, "data_fwhm")
        solid_angle = bu.get_fit_vec(sfits, "data_solid_angle_corr")
        msk = data_fwhm < 3 * fwhm_exp
        msk *= data_fwhm < np.percentile(data_fwhm[msk], 95)
        msk *= solid_angle < 3 * sang_exp
        sfits = sfits[msk]
        smjobs = smjobs[msk]
        sfjobs = sfjobs[msk]
        for epoch in epochs:
            plot_dir_epc = os.path.join(plot_dir_spl, f"{epoch[0]}_{epoch[1]}")
            os.makedirs(plot_dir_epc, exist_ok=True)
            print(f"\t{spl} {epoch}")
            times = sfits["time"]
            tmsk = (times >= epoch[0]) * (times < epoch[1])
            if np.sum(tmsk) == 0:
                print(f"\t\tNo maps found! Skipping...")
                continue
            mcoadd = enmap.zeros(tmap.shape, tmap.wcs)
            wcoadd = enmap.zeros(tmap.shape, tmap.wcs)
            mlcoadd = enmap.zeros(tmap.shape, tmap.wcs)
            mwcoadd = enmap.zeros(tmap.shape, tmap.wcs)
            rmcoadd = enmap.zeros(tmap.shape, tmap.wcs)
            rwcoadd = enmap.zeros(tmap.shape, tmap.wcs)
            for fit, mjob, fjob in zip(sfits[tmsk], smjobs[tmsk], sfjobs[tmsk]):
                if args.plot_only:
                    continue
                # Load
                try:
                    solved = enmap.read_map(os.path.join(data_dir, mjob.tags["solved"]))
                    weights = enmap.read_map(
                        os.path.join(data_dir, mjob.tags["weights"])
                    )[np.diag_indices(len(solved))]
                    resid = enmap.read_map(os.path.join(data_dir, fjob.tags["resid"]))
                    if len(resid.shape) == 2:
                        resid = resid.reshape((1,) + resid.shape)
                    resid_weights = enmap.read_map(
                        os.path.join(data_dir, fjob.tags["resid_weights"])
                    )
                    if len(resid_weights.shape) == 4:
                        resid_weights = resid_weights[
                            np.diag_indices(len(resid_weights))
                        ]
                    resid_weights = resid_weights.reshape(resid.shape)
                except FileNotFoundError:
                    print(f"Maps missing for job: {mjob}")
                    continue
                if "ml_map" in mjob.tags and mjob.tags["ml_map"] != "":
                    try:
                        mlmap = enmap.read_map(
                            os.path.join(data_dir, mjob.tags["ml_map"])
                        )
                        mlweights = enmap.read_map(
                            os.path.join(data_dir, mjob.tags["ml_div"])
                        )[np.diag_indices(len(mlmap))]
                    except FileNotFoundError:
                        print(f"ML Maps missing for job: {mjob}")
                        continue
                else:
                    print(f"No ML maps for jobs: {mjob}")
                    mlmap = enmap.zeros(solved.shape, solved.wcs)
                    mlweights = enmap.zeros(weights.shape, weights.wcs)

                # Make everything look like TQU
                solved = view_TQU(solved)
                weights = view_TQU(weights)
                mlmap = view_TQU(mlmap)
                mlweights - view_TQU(mlweights)
                resid = view_TQU(resid)
                resid_weights = view_TQU(resid_weights)
                if not np.all(
                    np.array(
                        [
                            len(solved),
                            len(weights),
                            len(mlmap),
                            len(mlweights),
                            len(resid),
                            len(resid_weights),
                            len(solved.shape),
                            len(weights.shape),
                            len(mlmap.shape),
                            len(mlweights.shape),
                            len(resid.shape),
                            len(resid_weights.shape),
                        ]
                    )
                    == 3
                ):
                    raise ValueError("Maps don't look like TQU!")

                # Crop, recenter, and normalize
                cent = np.array(
                    (
                        fit["aman"].gauss.eta0.to(u.rad).value,
                        fit["aman"].gauss.xi0.to(u.rad).value,
                    )
                )
                solved = reproject.thumbnails(
                    solved,
                    r=ext_rad,
                    coords=cent,
                    oshape=(pix_extent, pix_extent),
                    owcs=twcs,
                )
                solved = (
                    solved
                    - np.array([fit["aman"].gauss.off.value, 0, 0]).reshape((3, 1, 1))
                ) / fit["aman"].gauss.amp.value
                weights = (
                    reproject.thumbnails_ivar(
                        weights,
                        r=ext_rad,
                        coords=cent,
                        oshape=(pix_extent, pix_extent),
                        owcs=twcs,
                    )
                    * fit["aman"].gauss.amp.value**2
                )
                mlmap = (
                    reproject.thumbnails(
                        mlmap,
                        r=ext_rad,
                        coords=cent,
                        oshape=(pix_extent, pix_extent),
                        owcs=twcs,
                    )
                    / fit["aman"].gauss.amp.value
                )
                mlweights = (
                    reproject.thumbnails_ivar(
                        mlweights,
                        r=ext_rad,
                        coords=cent,
                        oshape=(pix_extent, pix_extent),
                        owcs=twcs,
                    )
                    * fit["aman"].gauss.amp.value**2
                )
                resid = (
                    reproject.thumbnails(
                        resid,
                        r=ext_rad,
                        coords=cent,
                        oshape=(pix_extent, pix_extent),
                        owcs=twcs,
                    )
                    / fit["aman"].gauss.amp.value
                )
                resid_weights = (
                    reproject.thumbnails_ivar(
                        resid_weights,
                        r=ext_rad,
                        coords=cent,
                        oshape=(pix_extent, pix_extent),
                        owcs=twcs,
                    )
                    * fit["aman"].gauss.amp.value**2
                )

                # If the new center seems very far from the origin then lets skip
                cent_est = bu.estimate_cent(solved[0], sigma=10, buf=1)
                dist = np.linalg.norm(cent_est - solved.wcs.wcs.crpix)
                if dist > miscenter_thresh:
                    print(
                        f"\t\t{mjob.tags['obs_id']} {mjob.tags['stream_id']} {mjob.tags['band']} ({mjob.tags['source']}) seems miscentered! Skipping!"
                    )
                    continue

                # Add
                np.nan_to_num(solved, copy=False, nan=0, posinf=0, neginf=0)
                np.nan_to_num(weights, copy=False, nan=0, posinf=0, neginf=0)
                np.nan_to_num(mlmap, copy=False, nan=0, posinf=0, neginf=0)
                np.nan_to_num(mlweights, copy=False, nan=0, posinf=0, neginf=0)
                np.nan_to_num(resid, copy=False, nan=0, posinf=0, neginf=0)
                np.nan_to_num(resid_weights, copy=False, nan=0, posinf=0, neginf=0)
                mcoadd.insert(solved * weights, op=op)
                wcoadd.insert(weights, op=op)
                mlcoadd.insert(mlmap * mlweights, op=op)
                mwcoadd.insert(mlweights, op=op)
                rmcoadd.insert(resid * resid_weights, op=op)
                rwcoadd.insert(resid_weights, op=op)

            # Divide weights
            with np.errstate(divide="ignore", invalid="ignore"):
                mcoadd /= wcoadd
                mlcoadd /= mwcoadd
                rmcoadd /= rwcoadd
            np.nan_to_num(mcoadd, copy=False, nan=0, posinf=0, neginf=0)
            np.nan_to_num(mlcoadd, copy=False, nan=0, posinf=0, neginf=0)
            np.nan_to_num(rmcoadd, copy=False, nan=0, posinf=0, neginf=0)

            # Save and plot
            for omap, name in [
                (mcoadd, "stack"),
                (mlcoadd, "ml_stack"),
                (rmcoadd, "resid_stack"),
            ]:
                path = os.path.join(
                    data_dir_spl, f"{spl}_{epoch[0]}_{epoch[1]}_{name}.fits"
                )
                if args.plot_only:
                    if not os.path.isfile(path):
                        print("\t\tMaps do not exist!")
                        continue
                    omap = enmap.read_map(path)
                else:
                    enmap.write_map(
                        path,
                        omap,
                        "fits",
                        allow_modify=True,
                    )
                posmap = omap.posmap()
                posmap = np.rad2deg(posmap) * 3600
                for append, smap in [
                    ("", omap),
                    ("_smooth3pix", enmap.smooth_gauss(omap, 3 * res)),
                ]:
                    plot_map_complete(
                        smap,
                        posmap,
                        pixsize,
                        extent,
                        (0, 0),
                        plot_dir_epc,
                        f"{spl} {epoch[0]} {epoch[1]}",
                        log_thresh=log_thresh,
                        append=name + append,
                        qrur=True,
                    )
