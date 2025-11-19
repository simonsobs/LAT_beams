import argparse
import glob
import logging
import os
import sys
import time
from copy import deepcopy

import h5py
import mpi4py.rc
import numpy as np
import yaml
from so3g.proj import RangesMatrix
from sotodlib import tod_ops
from sotodlib.coords import planets as cp
from sotodlib.core import Context, metadata
from sotodlib.preprocess.preprocess_util import preproc_or_load_group
from sotodlib.site_pipeline import jobdb

from lat_beams.beam_utils import estimate_cent, plot_map
from lat_beams.utils import print_once

mpi4py.rc.threads = False
from mpi4py import MPI

mpi4py.rc.threads = False
from mpi4py import MPI

tod_ops.filters.logger.setLevel(logging.ERROR)
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

band_names = {"m": ["f090", "f150"], "u": ["f220", "f280"]}
comps = "TQU"

cp.logger.setLevel(logging.WARNING)


def set_tag(job, key, new_val):
    # This should be provided by the Job class but it's not...
    for _t in job._tags:
        if _t.key == key:
            _t.value = new_val
            return
    else:
        raise ValueError(f'No tag called "{key}"')


parser = argparse.ArgumentParser()
parser.add_argument("cfg", help="Path to the config file")
parser.add_argument("--obs_ids", nargs="+", help="Pass a list of obs ids to run on")
parser.add_argument(
    "--overwrite", "-o", action="store_true", help="Overwrite an existing map"
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
source_list = cfg.get("source_list", ["mars", "saturn"])
min_dets = cfg.get("min_dets", 50)
min_hits = cfg.get("min_hits", 1)
min_det_secs = cfg.get("min_det_secs", 600)
min_snr = cfg.get("min_snr", 5)
n_modes = cfg.get("n_modes", 10)
del_map = cfg.get("del_map", True)
extent = cfg.get("extent", 1800)
zoom = cfg.get("zoom", 5)
buf = cfg.get("buffer", 30)
log_thresh = cfg.get("log_thresh", 1e-3)
smooth_kern = cfg.get("smooth_kern", 60)
pointing_type = cfg.get("pointing_type", "pointing_model")
preprocess_cfg = cfg.get("preprocess", None)

if preprocess_cfg is None:
    raise ValueError("Must specify a valid preprocess config!")

# Check pointing_type
if pointing_type not in ["pointing_model", "per_obs", "raw"]:
    raise ValueError(f"Invalid pointing_type {pointing_type}")
if pointing_type == "raw" and comps != "T":
    print_once(f"Running with raw pointing, changing comps from {comps} to T")
    comps = "T"

# Setup folders and jobdb
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "beams/lat")
plot_dir = os.path.join(root_dir, "plots", project_dir, "source_maps", pointing_type)
data_dir = os.path.join(root_dir, "data", project_dir, "source_maps", pointing_type)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
jdb = jobdb.JobManager(sqlite_file=os.path.join(data_dir, "jobdb.db"))

# Get the list of observations
ctx = Context(cfg.get("context", "/so/metadata/lat/contexts/smurf_detcal.yaml"))
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
if args.obs_ids is not None:
    obslist = [ctx.obsdb.get(obs_id) for obs_id in args.obs_ids]
else:
    src_str = "==1 or ".join(source_list) + "==1"
    start_time = cfg["start_time"]
    if args.lookback is not None:
        start_time = time.time() - 3600 * args.lookback
    obslist = ctx.obsdb.query(
        f"type=='obs' and subtype=='cal' and start_time > {start_time} and stop_time < {cfg['stop_time']} and ({src_str})",
        tags=source_list,
    )
print_once(f"Found {len(obslist)} observations to map")

# Keep only the ones with a focal plane
if pointing_type != "pointing_model":
    dbs = [md["db"] for md in ctx["metadata"] if "focal_plane" in md.get("name", "")]
    if len(dbs) > 1:
        print_once("Multiple pointing metadata entries found, using the first one")
    elif len(dbs) == 0:
        print_once("No pointing metadata entries found")
        sys.exit()
    print_once(f"Using ManifestDb at {dbs[0]}")
    db = metadata.ManifestDb(dbs[0])
    obs_ids = np.array([entry["obs:obs_id"] for entry in db.inspect()])
    obslist = [obs for obs in obslist if obs["obs_id"] in obs_ids]
    print_once(f"Only {len(obslist)} observations with pointing metadata")

# Add to jobdb if they don't exist
# Do this from rank 0 only
if myrank == 0:
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
            jobs = jdb.get_jobs(
                jclass="beam_map",
                tags={
                    "obs_id": obs["obs_id"],
                    "wafer_slot": ws,
                    "stream_id": ufm,
                    "band": band,
                    "comps": comps,
                },
            )
            if len(jobs) == 0:
                jdb.create_job(
                    "beam_map",
                    tags={
                        "obs_id": obs["obs_id"],
                        "wafer_slot": ws,
                        "stream_id": ufm,
                        "band": band,
                        "message": "",
                        "binned": "",
                        "detweights": "",
                        "solved": "",
                        "weights": "",
                        "comps": comps,
                        "source": "",
                    },
                )
    for j in jdb.get_jobs(jclass="beam_map"):
        if j.lock and (time.time() - j.lock) > 60:
            jdb.unlock(j.id)
        if j.jstate.name == "failed" and (j.lock or args.retry_failed):
            with jdb.locked(j) as job:
                job.jstate = "open"
            jdb.unlock(j.id)

# Get jobs list
if args.overwrite:
    joblist = jdb.get_jobs(jclass="beam_map", locked=False)
else:
    joblist = jdb.get_jobs(jclass="beam_map", jstate="open", locked=False)
joblist = joblist[args.start_from :]
print_once(f"{len(joblist)} maps to make.")

# Split for MPI
obs_idx = np.array_split(np.arange(len(joblist)), nproc)[myrank]
joblist = [job for i, job in enumerate(joblist) if i in obs_idx]

# Get settings for source mask
res = cfg.get("res", (10.0 / 3600.0) * np.pi / 180.0)
pixsize = 3600 * np.rad2deg(res)
mask_size = cfg.get("mask_size", 0.1)
search_mask = cfg.get("search_mask", {"shape": "circle", "xyr": (0, 0, 0.5)})

# Mapping loop
source_list = set(source_list)
for i, j in enumerate(joblist):
    sys.stdout.flush()
    with jdb.locked(j) as job:
        job.mark_visited()
        obs_id = job.tags["obs_id"]
        ufm = job.tags["stream_id"]
        ws = job.tags["wafer_slot"]
        band = job.tags["band"]
        print(f"(rank {myrank}) Mapping {obs_id} {ufm} {band}({i+1}/{len(joblist)})")

        obs = ctx.obsdb.get(obs_id, tags=True)
        meta = ctx.get_meta(obs_id)
        if meta.dets.count == 0:
            msg = "Looks like we don't have real metadata for this observation!"
            print(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
            continue

        src_names = list(source_list & set(obs["tags"]))
        if len(src_names) > 1:
            print("\tObservation tagged for multiple sources!")
        elif len(src_names) == 0:
            msg = "Observation somehow not tagged for any sources in source_list! Skipping!"
            print(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
            print(f"\t\tTags were: {obs['tags']}")
            continue
        src_name = "_".join(src_names)
        print(f"\tMapping {src_name}")

        if "hits" in meta.focal_plane:
            meta.restrict("dets", meta.focal_plane.hits >= min_hits)
            if meta.dets.count < min_dets:
                msg = f"Only {meta.dets.count} detectors with good pointing fits!"
                print(f"\t{msg}")
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
        try:
            err, _, _, aman = preproc_or_load_group(
                obs["obs_id"],
                preprocess_cfg,
                dets={"wafer_slot": ws, "wafer.bandpass": band},
                save_archive=False,
                overwrite=True,
            )
        except:
            msg = "Failed to load or preprocess!"
            print(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
            continue
        if aman is None:
            msg = f"Preprocess failed with error {err}"
            print(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
            continue

        aman.restrict(
            "dets",
            np.isfinite(aman.focal_plane.xi)
            * np.isfinite(aman.focal_plane.eta)
            * np.isfinite(aman.focal_plane.gamma),
        )

        if aman.dets.count < min_dets:
            msg = f"Only {aman.dets.count} dets!"
            print(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
            continue

        # Get initial source_flags
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
        sig_filt = cp.filter_for_sources(
            tod=aman,
            signal=aman.signal.copy(),
            source_flags=source_flags,
            n_modes=2 * n_modes,
        )
        smsk = source_flags.mask()
        sig_filt_src = sig_filt.copy()
        sig_filt_src[~smsk] = np.nan
        sig_filt[smsk] = np.nan
        all_src = np.all(smsk, axis=-1)
        no_src = ~np.any(smsk, axis=-1)
        sdets = ~(all_src + no_src)
        peak_snr = np.zeros(len(sig_filt))
        with np.errstate(divide="ignore"):
            peak_snr[sdets] = np.nanmax(sig_filt_src[sdets], axis=-1) / np.nanstd(
                np.diff(sig_filt[sdets], axis=-1)
            )
        to_cut = peak_snr < min_snr  # + ~np.isfinite(peak_snr)
        to_cut[~sdets] = False
        cuts = RangesMatrix.from_mask(
            np.zeros_like(aman.signal, bool) + to_cut[..., None]
        )
        print(f"\tCutting {np.sum(to_cut)} detectors from map")
        if np.sum(~to_cut) < min_dets:
            msg = f"Not enough detectors after source flag cuts!"
            print(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
            continue

        # Get time on source
        det_secs = np.sum((source_flags * ~cuts).get_stats()["samples"]) * np.mean(
            np.diff(aman.timestamps)
        )
        print(f"\t{det_secs} detector seconds on source in intial mask")
        if det_secs < min_det_secs:
            msg = f"\tNot enough time on source in initial mask."
            print(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
            continue

        # Initial map
        out = cp.make_map(
            aman,
            center_on=src_to_map,
            res=res,
            cuts=cuts,
            source_flags=source_flags,
            comps="T",
            filename=None,
            n_modes=n_modes,
        )

        # Smooth and find the center
        cent = estimate_cent(out["solved"][0], smooth_kern / pixsize, buf)

        # Estimate SNR
        peak = out["solved"][0][cent]
        snr = (
            peak
            / tod_ops.jumps.std_est(np.atleast_2d(out["solved"][0].ravel()), ds=1)[0]
        )
        print(f"\tInitial map SNR approximately {snr}")
        if snr < min_snr * np.sqrt(np.sum(~to_cut)) / 2:
            msg = f"\tInitial map SNR too low."
            print(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
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
                mask_size * 90.0 / float(band[1:]),
            ),
        }
        source_flags = cp.compute_source_flags(
            tod=aman,
            P=None,
            mask=mask,
            center_on=src_to_map,
            res=res,
            max_pix=4e8,
            wrap=None,
        )

        # Get time on source
        det_secs = np.sum((source_flags * ~cuts).get_stats()["samples"]) * np.mean(
            np.diff(aman.timestamps)
        )
        print(f"\t{det_secs} detector seconds on source")
        if det_secs < min_det_secs:
            msg = f"\tNot enough time on source."
            print(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
            continue

        # Its map time for real now
        out = cp.make_map(
            aman,
            center_on=src_to_map,
            res=res,
            cuts=cuts,
            source_flags=source_flags,
            comps=comps,
            filename=os.path.join(obs_data_dir, "{obs_id}_{ufm}_{band}_{map}.fits"),
            n_modes=n_modes,
            info={"obs_id": obs["obs_id"], "ufm": ufm, "band": band},
        )
        [[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(
            out["solved"].corners(corner=False)
        )
        plt_extent = [ra_min, ra_max, dec_min, dec_max]

        # Add paths to job
        set_tag(
            job,
            "binned",
            os.path.join(obs_data_dir, f"{obs_id}_{ufm}_{band}_binned.fits"),
        )
        set_tag(
            job,
            "detweights",
            os.path.join(obs_data_dir, f"{obs_id}_{ufm}_{band}_detweights.h5"),
        )
        set_tag(
            job,
            "solved",
            os.path.join(obs_data_dir, f"{obs_id}_{ufm}_{band}_solved.fits"),
        )
        set_tag(
            job,
            "weights",
            os.path.join(obs_data_dir, f"{obs_id}_{ufm}_{band}_weights.fits"),
        )

        # Smooth and find the center
        cent = estimate_cent(out["solved"][0], smooth_kern / pixsize, buf)

        # Estimate SNR
        peak = out["solved"][0][cent]
        snr = (
            peak
            / tod_ops.jumps.std_est(np.atleast_2d(out["solved"][0].ravel()), ds=1)[0]
        )
        print(f"\tMap SNR approximately {snr}")
        if snr < min_snr * np.sqrt(np.sum(~to_cut)) / 2:
            msg = f"Map SNR too low."
            print(f"\t{msg}")
            set_tag(job, "message", msg)
            job.jstate = "failed"
            if not del_map:
                continue
            print("\tDeleting map files")
            glob_path = os.path.join(obs_data_dir, f"{obs['obs_id']}_{ufm}_{band}*.*")
            flist = glob.glob(glob_path)
            for fname in flist:
                if os.path.isfile(fname):
                    os.remove(fname)
            set_tag(job, "binned", "")
            set_tag(job, "detweights", "")
            set_tag(job, "solved", "")
            set_tag(job, "weightweis", "")
            continue

        # Plot
        ufm_plot_dir = os.path.join(obs_plot_dir, ufm)
        os.makedirs(ufm_plot_dir, exist_ok=True)
        plt_cent = (ra_min - pixsize * cent[1], dec_min + pixsize * cent[0])
        for i, comp in enumerate(comps):
            map_norm = peak / out["solved"][i][cent]
            plot_map(
                map_norm * out["solved"][i],
                pixsize,
                extent,
                plt_extent,
                cent,
                plt_cent,
                zoom,
                ufm_plot_dir,
                obs,
                ufm,
                band,
                comp,
                False,
                log_thresh,
            )
            plot_map(
                map_norm * out["solved"][i],
                pixsize,
                extent,
                plt_extent,
                cent,
                plt_cent,
                zoom,
                ufm_plot_dir,
                obs,
                ufm,
                band,
                comp,
                True,
                log_thresh,
            )

        set_tag(job, "message", "Success")
        job.jstate = "done"

    sys.stdout.flush()

# Splits stuff to implement later
# TODO: Bin in annuli
# TODO: Per det maps?
