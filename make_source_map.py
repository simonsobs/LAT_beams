import argparse
from copy import deepcopy
import logging
import time
import glob
import os
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import yaml
from matplotlib.colors import SymLogNorm
from scipy.ndimage import gaussian_filter
from so3g.proj import RangesMatrix
from sotodlib import tod_ops
from sotodlib.coords import planets as cp
from sotodlib.core import Context, metadata
from sotodlib.core.flagman import has_any_cuts
from sotodlib.obs_ops.utils import correct_iir_params
from sotodlib.coords.pointing_model import apply_pointing_model
import mpi4py.rc
import logging

from lat_beams.utils import print_once

mpi4py.rc.threads = False
from mpi4py import MPI

mpi4py.rc.threads = False
from mpi4py import MPI

tod_ops.filters.logger.setLevel(logging.ERROR)
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

plt.rcParams["image.cmap"] = "RdGy_r"

N_FILES = 4
band_names = {"m": ["f090", "f150"], "u": ["f220", "f280"]}

cp.logger.setLevel(logging.WARNING)


def radial_profile(data, center):
    msk = np.isfinite(data.ravel())
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel()[msk], data.ravel()[msk])
    nr = np.bincount(r.ravel()[msk])
    radialprofile = tbin / nr
    return radialprofile


def plot_map(
    data,
    extent,
    plt_extent,
    cent,
    plt_cent,
    zoom,
    ufm_plot_dir,
    obs,
    ufm,
    band_name,
    comp="T",
    log=False,
    log_thresh=1e-3,
):
    _norm = None
    label = f"_{comp}"
    rprof = radial_profile(data, cent[::-1])[: int(0.5 * min(*data.shape))]
    if log:
        _norm = SymLogNorm(linthresh=log_thresh * np.max(data), clip=True)
        label = f"_{comp}_log10"
        with np.errstate(divide="ignore", invalid="ignore"):
            rprof = np.sign(rprof) * np.log10(np.abs(rprof))
    plt.close()
    plt.imshow(data, origin="lower", extent=plt_extent, norm=_norm)
    plt.colorbar()
    plt.grid()
    plt.xlabel('RA (")')
    plt.ylabel('Dec (")')
    plt.title(f"{obs['obs_id']}_{ufm}_{band_name}{label.replace('_', ' ')}")
    plt.xlim((plt_cent[0] - extent, plt_cent[0] + extent))
    plt.ylim((plt_cent[1] - extent, plt_cent[1] + extent))
    plt.savefig(
        os.path.join(ufm_plot_dir, f"{obs['obs_id']}_{ufm}_{band_name}_map{label}.png")
    )
    plt.xlim((plt_cent[0] - extent / zoom, plt_cent[0] + extent / zoom))
    plt.ylim((plt_cent[1] - extent / zoom, plt_cent[1] + extent / zoom))
    plt.savefig(
        os.path.join(
            ufm_plot_dir, f"{obs['obs_id']}_{ufm}_{band_name}_map{label}_zoom.png"
        )
    )

    plt.close()
    x = np.linspace(0, pixsize * len(rprof), len(rprof))
    plt.plot(x, rprof)
    plt.xlabel('Radius (")')
    plt.title(f"{obs['obs_id']}_{ufm}_{band_name}{label.replace('_', ' ')}")
    plt.xlim((0, extent))
    plt.savefig(
        os.path.join(
            ufm_plot_dir, f"{obs['obs_id']}_{ufm}_{band_name}_prof{label}.png"
        ),
        bbox_inches="tight",
    )
    plt.xlim((0, extent / zoom))
    lims = plt.gca().get_xlim()
    i = np.where((x >= lims[0]) & (x <= lims[1]))[0]
    plt.gca().set_ylim(rprof[i].min(), rprof[i].max())
    plt.savefig(
        os.path.join(
            ufm_plot_dir, f"{obs['obs_id']}_{ufm}_{band_name}_prof{label}_zoom.png"
        ),
        bbox_inches="tight",
    )


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
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

# Get some global settings
source_list = cfg.get("source_list", ["mars", "saturn"])
comps = cfg.get("comps", "TQU")
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
pointing_type = cfg.get("pointing_type", "pointing_model")

# Check pointing_type
if pointing_type not in ["pointing_model", "per_obs", "raw"]:
    raise ValueError(f"Invalid pointing_type {pointing_type}")
if pointing_type == "raw" and comps != "T":
    print_once(f"Running with raw pointing, changing comps from {comps} to T")
per_obs = pointing_type in ["per_obs", "raw"]

# Setup folders
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "beams/lat")
plot_dir = os.path.join(root_dir, "plots", project_dir, "source_maps", pointing_type)
data_dir = os.path.join(root_dir, "data", project_dir, "source_maps", pointing_type)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

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
if per_obs:
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

# Get settings for source mask
res = cfg.get("res", (10.0 / 3600.0) * np.pi / 180.0)
mask = cfg.get("mask", {"shape": "circle", "xyr": (0, 0, 0.3)})

# Split for MPI
obslist = np.array_split(obslist, nproc)[myrank]

# Mapping loop
source_list = set(source_list)
for i, obs in enumerate(obslist):
    sys.stdout.flush()
    if i < args.start_from:
        continue
    print(f"(rank {myrank}) Mapping {obs['obs_id']} ({i+1}/{len(obslist)})")

    obs = ctx.obsdb.get(obs["obs_id"], tags=True)
    meta = ctx.get_meta(obs["obs_id"])
    if meta.dets.count == 0:
        print(
            f"\tLooks like we don't have real metadata for this observation. Skipping..."
        )
        continue

    src_names = list(source_list & set(obs["tags"]))
    if len(src_names) > 1:
        print("\tObservation tagged for multiple sources!")
    elif len(src_names) == 0:
        print(
            "\tObservation somehow not tagged for any sources in source_list! Skipping!"
        )
        print(f"\t\tTags were: {obs['tags']}")
        continue
    src_name = "_".join(src_names)
    print(f"\tMapping {src_name}")

    meta.restrict("dets", np.isfinite(meta.det_cal.tau_eff))
    db_flag = tod_ops.flags.get_det_bias_flags(meta)
    meta.restrict("dets", ~db_flag.det_bias_flags)
    meta.restrict("dets", np.isfinite(meta.det_cal.phase_to_pW))
    if meta.dets.count < min_dets:
        print(f"Only {meta.dets.count} detectors with good bias. Skipping...")
        continue
    if "hits" in meta.focal_plane:
        meta.restrict("dets", meta.focal_plane.hits >= min_hits)
        if meta.dets.count < min_dets:
            print(f"Only {meta.dets.count} detectors with good fits. Skipping...")
            continue
    meta.restrict(
        "dets",
        np.isfinite(meta.focal_plane.xi)
        * np.isfinite(meta.focal_plane.eta)
        * np.isfinite(meta.focal_plane.gamma),
    )

    obs_plot_dir = os.path.join(
        plot_dir, src_name, str(obs["timestamp"])[:5], obs["obs_id"]
    )
    obs_data_dir = os.path.join(
        data_dir, src_name, str(obs["timestamp"])[:5], obs["obs_id"]
    )
    os.makedirs(obs_data_dir, exist_ok=True)
    ufms = np.unique(meta.det_info.stream_id)

    src_to_map = src_name.split("_")[0]
    if src_to_map == "taua":
        src_to_map = ("tauA", 83.6272579, 22.02159891)
    for ufm in ufms:
        meta_ufm = meta.copy().restrict("dets", meta.det_info.stream_id == ufm)
        bp = (meta_ufm.det_cal.bg % 4) // 2
        tube_band = ufm[4]
        rsets = []
        for band in np.unique(bp):
            meta_band = meta_ufm.copy().restrict("dets", bp == band)
            band_name = band_names[tube_band][band]
            print(f"\tMapping {ufm} {band_name}")

            # Check if we already mapped
            # TODO: add a mode to replot but not refit
            glob_path = os.path.join(
                obs_data_dir, f"{obs['obs_id']}_{ufm}_{band_name}*"
            )
            flist = glob.glob(glob_path)
            if len(flist) >= N_FILES and (not args.overwrite):
                print(
                    "\t\tMaps appear to already exist and overwrite is not set. Skipping..."
                )
                continue

            if meta_band.dets.count < min_dets:
                print(f"\t\t(bandpass) Only {meta_band.dets.count} dets! Skipping...")
                continue

            # Load and process the TOD
            aman = ctx.get_obs(meta_band)
            if "boresight" not in aman:
                print("\t\tNo boresight in TOD! Skipping...")
                continue
            try:
                filt = tod_ops.filters.iir_filter(invert=True)
                aman.signal = tod_ops.filters.fourier_filter(
                    aman, filt, signal_name="signal"
                )
            except ValueError:
                print("\t\tNo iir params! Adding defaults...")
                correct_iir_params(aman, True)
                filt = tod_ops.filters.iir_filter(invert=True)
                aman.signal = tod_ops.filters.fourier_filter(
                    aman, filt, signal_name="signal"
                )
            filt = tod_ops.filters.timeconst_filter(
                timeconst=aman.det_cal.tau_eff, invert=True
            )
            aman.signal = tod_ops.filters.fourier_filter(
                aman, filt, signal_name="signal"
            )

            tod_ops.detrend_tod(aman, "median", in_place=True)
            tf = tod_ops.flags.get_trending_flags(aman, max_trend=3, t_piece=30)
            tdets = has_any_cuts(tf)
            aman.restrict("dets", ~tdets)

            if aman.dets.count < min_dets:
                print(f"\t\t(unlocks) Only {aman.dets.count} dets! Skipping...")
                continue

            jflags, _, jfix = tod_ops.jumps.twopi_jumps(
                aman,
                signal=aman.signal,
                win_size=30,
                nsigma=3,
                fix=True,
                inplace=True,
                merge=False,
            )
            aman.signal = jfix
            gfilled = tod_ops.gapfill.fill_glitches(
                aman, nbuf=30, use_pca=False, modes=1, glitch_flags=jflags
            )
            aman.signal = gfilled
            nj = np.array(
                [len(jflags.ranges[i].ranges()) for i in range(len(aman.signal))]
            )
            aman.restrict("dets", nj < 10)

            if aman.dets.count < min_dets:
                print(f"\t\t(jumps) Only {aman.dets.count} dets! Skipping...")
                continue

            aman.signal -= np.mean(np.array(aman.signal), axis=-1)[..., None]
            aman.signal *= aman.det_cal.phase_to_pW[..., None]
            orig = aman.copy()

            # Pointing model
            if not per_obs:
                aman.boresight = apply_pointing_model(aman)

            # Get source_flags
            _mask = deepcopy(mask)
            _mask["xyr"] = _mask["xyr"][:-1] + [
                _mask["xyr"][-1] * 90.0 / float(band_name[1:]),
            ]
            source_flags = cp.compute_source_flags(
                tod=aman,
                P=None,
                mask=_mask,
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
            print(f"\t\tCutting {np.sum(to_cut)} detectors from map")
            if np.sum(~to_cut) < min_dets:
                print(f"\t\tNot enough detectors! Skipping...")
                continue

            # Get time on source
            det_secs = np.sum((source_flags * ~cuts).get_stats()["samples"]) * np.mean(
                np.diff(aman.timestamps)
            )
            print(f"\t\t{det_secs} detector seconds on source")
            if det_secs < min_det_secs:
                print(f"\t\tNot enough time on source. Skipping...")
                continue

            # Its map time!
            out = cp.make_map(
                aman,
                center_on=src_to_map,
                res=res,
                cuts=cuts,
                source_flags=source_flags,
                comps=comps,
                filename=os.path.join(
                    obs_data_dir, "{obs_id}_{ufm}_{band_name}_{map}.fits"
                ),
                n_modes=n_modes,
                info={"obs_id": obs["obs_id"], "ufm": ufm, "band_name": band_name},
            )
            [[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(
                out["solved"].corners(corner=False)
            )
            plt_extent = [ra_min, ra_max, dec_min, dec_max]
            pixsize = 3600 * out["solved"].wcs.wcs.cdelt[1]

            # Smooth and find the center
            smoothed = gaussian_filter(out["solved"][0], sigma=60 / pixsize)
            smoothed[:buf] = np.nan
            smoothed[-1 * buf :] = np.nan
            smoothed[:, :buf] = np.nan
            smoothed[:, -1 * buf :] = np.nan
            cent = np.unravel_index(np.nanargmax(smoothed, axis=None), smoothed.shape)

            # Estimate SNR, but only if we have a T map
            peak = 1.0
            if "T" in comps:
                peak = out["solved"][0][cent]
                snr = (
                    peak
                    / tod_ops.jumps.std_est(
                        np.atleast_2d(out["solved"][0].ravel()), ds=1
                    )[0]
                )
                print(f"\t\tMap SNR approximately {snr}")
                if snr < min_snr * np.sqrt(np.sum(~to_cut)) / 2:
                    print(f"\t\tMap SNR too low! Skipping...")
                    if not del_map:
                        continue
                    print("\t\tDeleting fits files")
                    glob_path = os.path.join(
                        obs_data_dir, f"{obs['obs_id']}_{ufm}_{band_name}*.fits"
                    )
                    flist = glob.glob(glob_path)
                    for fname in flist:
                        if os.path.isfile(fname):
                            os.remove(fname)
                    continue

            # Plot
            ufm_plot_dir = os.path.join(obs_plot_dir, ufm)
            os.makedirs(ufm_plot_dir, exist_ok=True)
            plt_cent = (ra_min - pixsize * cent[1], dec_min + pixsize * cent[0])
            for i, comp in enumerate(comps):
                map_norm = peak / out["solved"][i][cent]
                plot_map(
                    map_norm * out["solved"][i],
                    extent,
                    plt_extent,
                    cent,
                    plt_cent,
                    zoom,
                    ufm_plot_dir,
                    obs,
                    ufm,
                    band_name,
                    comp,
                    False,
                    log_thresh,
                )
                plot_map(
                    map_norm * out["solved"][i],
                    extent,
                    plt_extent,
                    cent,
                    plt_cent,
                    zoom,
                    ufm_plot_dir,
                    obs,
                    ufm,
                    band_name,
                    comp,
                    True,
                    log_thresh,
                )

    sys.stdout.flush()

# Splits stuff to implement later
# TODO: Bin in annuli
# TODO: Per det maps?
