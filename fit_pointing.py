"""
More generic fitting script.
Still somewhat LAT specific but could be genralized if desired.
"""

#### TODO ####
# More functions
# Time domain fitting of single source [just jup notebook is ok]
# more plots:
# - fit for each detector [on demand]
# - input priors
# - coverage showing planet trajectory through focal plane [just use/modify the matthew functions]
# - should develop tests for this
# - allow a list of sources

import argparse
import logging
import os
import sqlite3
import sys
import time
from functools import reduce

import h5py
import matplotlib.pyplot as plt
import mpi4py.rc
import numpy as np
import yaml
from sotodlib import tod_ops
from sotodlib.coords import planets as cp
from sotodlib.core import AxisManager, Context, metadata
from sotodlib.core.flagman import has_any_cuts
from sotodlib.io.metadata import write_dataset
from sotodlib.mapmaking import downsample_obs
from sotodlib.obs_ops.utils import correct_iir_params
from sotodlib.preprocess.preprocess_util import preproc_or_load_group
from sotodlib.site_pipeline import jobdb
from typing_extensions import cast

from lat_beams.fitting import fit_tod_pointing
from lat_beams.plotting import plot_focal_plane, plot_tod
from lat_beams.utils import print_once, set_tag

mpi4py.rc.threads = False
from mpi4py import MPI

tod_ops.filters.logger.setLevel(logging.ERROR)

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

band_names = {"l": ["f030", "f040"], "m": ["f090", "f150"], "u": ["f220", "f280"]}


def src_flag_cut(source, aman, nominal, ufm, res, mask):
    source_name = source
    # This is just to account for gap in sotodlib.
    # TODO: Add it to sotodlib...
    if source == "rcw38":
        source_name = "J134.78-47.509"

    # See how much of the source we saw...
    # Mask is made massive. This ONLY helps if you have prior knowledge of where source is.
    aman_dummy = aman.restrict("dets", [aman.dets.vals[0]], in_place=False)
    fp = AxisManager(aman_dummy.dets)
    fp.wrap(
        "xi",
        np.zeros(1) + xi_off + np.nanmean(np.array(nominal[ufm]["xi"][:])),
        [(0, "dets")],
    )
    fp.wrap(
        "eta",
        np.zeros(1) + eta_off + np.nanmean(np.array(nominal[ufm]["eta"][:])),
        [(0, "dets")],
    )
    fp.wrap(
        "gamma",
        np.zeros(1) + np.nanmean(np.array(nominal[ufm]["gamma"][:])),
        [(0, "dets")],
    )
    aman_dummy.wrap("focal_plane", fp)
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
    # Only the config is necessary; the rest are just for ease of use.
    # TODO: can refine these or add to them.
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="Path to the config file")
    parser.add_argument("--obs_ids", nargs="+", help="Pass a list of obs ids to run on")
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Overwrite an existing fit"
    )
    parser.add_argument(
        "--no_fit", "-n", action="store_true", help="Just plot TODs and don't fit"
    )
    parser.add_argument(
        "--forced_ws", "-ws", nargs="+", help="Force these wafer slots into the fit"
    )
    parser.add_argument(
        "--start_from",
        "-s",
        default=0,
        type=int,
        help="Skip to the nth obs (0 indexed)",
    )
    parser.add_argument(
        "--lookback",
        "-l",
        type=float,
        help="Amount of time to lookback for query, overides start time from config",
    )
    parser.add_argument("--profile", "-p", action="store_true", help="Run a profile")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    if args.no_fit:
        print_once(
            "Running in 'no_fit' mode. TOD plots will be made but pointing will not be fit"
        )

    if args.profile:
        from pyinstrument import Profiler

        profiler = Profiler()
        print_once("Running in profiler mode! Only a few dets will be kept")

    # Get some global settings
    forced_ws = args.forced_ws
    if cfg.get("try_all", False):
        forced_ws = ["ws0", "ws1", "ws2"]
    ds = cfg.get("ds", 5)
    hp_fc = cfg.get("hp_fc", 4)
    lp_fc = cfg.get("lp_fc", 30)
    n_med = cfg.get("n_med", 5)
    n_std = cfg.get("n_std", 10)
    source = cfg.get("source", "mars")
    xi_off = cfg.get("xi_off", 0.0)
    eta_off = cfg.get("eta_off", 0.0)
    min_samps = cfg.get("min_samps", 1000) / ds
    block_size = cfg.get("block_size", 5000) // ds
    min_dets = cfg.get("min_dets", 30)
    trim_samps = cfg.get("time_samps", 200) // ds
    min_hits = cfg.get("min_hits", 1)
    fwhm_tol = cfg.get("fwhm_tol", 0.2)
    fit_pars = cfg.get("fit_pars", {})
    src_msk = cfg.get("src_msk", True)
    fwhm = cfg.get("fwhm", None)
    pad = cfg.get("pad", True)
    max_chisq = cfg.get("max_chisq", 2.5)
    ufm_rad = cfg.get("ufm_rad", 0.01)
    preprocess_cfg = cfg.get("preprocess", None)
    tel = cfg.get("telescope", "lat")
    cfg_str = yaml.dump(cfg)

    if preprocess_cfg is None:
        raise ValueError("Must specify a valid preprocess config!")
    with open(preprocess_cfg, "r") as f:
        preprocess_str = yaml.dump(yaml.safe_load(preprocess_cfg))

    if fwhm is None:
        raise ValueError("FWHM not found in config file.")

    # Setup folders
    root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
    project_dir = cfg.get("project_dir", os.path.join("pointing", tel))
    plot_dir = os.path.join(root_dir, "plots", project_dir, "source_fits", source)
    data_dir = os.path.join(root_dir, "data", project_dir, "source_fits")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Get the list of observations
    ctx_path = cfg.get(
        "context",
        f"/global/cfs/cdirs/sobs/metadata/{tel}/contexts/smurf_detcal_local.yaml",
    )
    with open(ctx_path, "r") as f:
        ctx_str = yaml.dump(yaml.safe_load(f))
    ctx = Context(ctx_path)
    if ctx.obsdb is None:
        raise ValueError("No obsdb in context!")
    if args.obs_ids is not None:
        obslist = [ctx.obsdb.get(obs_id) for obs_id in args.obs_ids]
    else:
        start_time = cfg["start_time"]
        if args.lookback is not None:
            start_time = time.time() - 3600 * args.lookback
        obslist = ctx.obsdb.query(
            f"type=='obs' and subtype=='cal' and {source} and start_time > {start_time} and stop_time < {cfg['stop_time']}",
            tags=[f"{source}=1"],
        )

    # JobDB stuff
    jdb = jobdb.JobManager(sqlite_file=os.path.join(data_dir, "jobdb.db"))

    # Add to jobdb if they don't exist
    # Do this from rank 0 only
    if myrank == 0:
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
                jobs = jdb.get_jobs(
                    jclass="fit_pointing",
                    tags={
                        "obs_id": obs["obs_id"],
                        "wafer_slot": ws,
                        "stream_id": ufm,
                    },
                )
                if len(jobs) == 0:
                    jdb.create_job(
                        "beam_map",
                        tags={
                            "obs_id": obs["obs_id"],
                            "wafer_slot": ws,
                            "stream_id": ufm,
                            "message": "",
                            "source": "",
                            "config": "",
                            "context": "",
                            "preprocess": "",
                        },
                    )
        for j in jdb.get_jobs(jclass="fit_pointing"):
            if j.lock and (time.time() - j.lock) > 60:
                jdb.unlock(j.id)
            if j.jstate.name == "failed" and (j.lock or args.retry_failed):
                with jdb.locked(j) as job:
                    job.jstate = "open"
                jdb.unlock(j.id)

    # Get jobs list
    if args.overwrite:
        joblist = jdb.get_jobs(jclass="fit_pointing", locked=False)
    else:
        joblist = jdb.get_jobs(jclass="fit_pointing", jstate="open", locked=False)
    joblist = joblist[args.start_from :]
    source_list = [
        source,
    ]
    joblist = [job for job in joblist if job.tags["source"] in source_list]
    print_once(f"{len(joblist)} wafer-obs to fit.")

    # split for mpi
    obs_idx = np.array_split(np.arange(len(joblist)), nproc)[myrank]
    joblist = [job for i, job in enumerate(joblist) if i in obs_idx]

    n_fits = comm.allgather(len(joblist))
    max_fits = np.max(n_fits)
    if n_fits[0] != max_fits:
        raise ValueError("Root doesn't have max fits!")
    if len(joblist) < max_maps:
        joblist += [None] * (max_fits - len(joblist))

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
    nominal_path = os.path.expanduser(
        cfg.get("nominal", f"~/data/pointing/{tel}/nominal/focal_plane.h5")
    )
    nominal = h5py.File(nominal_path)

    # Get settings for source mask
    res = cfg.get("res", (2 / 300.0) * np.pi / 180.0)
    mask = cfg.get("mask", {"shape": "circle", "xyr": (0, 0, 0.75)})
    if args.profile:
        profiler.start()
        print_once("Restricting joblist to just 1 entry per process for profiling!")
        joblist = [joblist[0]]
    to_save = (None, None, None)
    for i, j in enumerate(joblist):
        sys.stdout.flush()
        comm.barrier()
        to_save = comm.gather(to_save, root=0)
        if myrank == 0 and to_save is not None and h5_file is not None:
            for rset, obs_id, ufm in to_save:
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
            print_once("Reloading h5 file to be safe!")
            h5_file.close()
            h5_file = h5py.File(h5_path, "a")

        if j is None:
            to_save = (None, None, None)
            continue
        with jdb.locked(j) as job:
            job.mark_visited()
            obs_id = job.tags["obs_id"]
            ufm = job.tags["stream_id"]
            ws = job.tags["wafer_slot"]
            print(f"(rank {myrank} Fitting {obs_id} {ufm}({i+1}/{len(joblist)})")

            # Save metadata and config info
            set_tag(job, "config", cfg_str)
            set_tag(job, "context", ctx_str)
            set_tag(job, "preprocess", preprocess_str)

            # Get metadata
            obs = ctx.obsdb.get(obs_id, tags=True)
            meta = ctx.get_meta(obs_id)
            if meta.dets.count == 0:
                msg = "Looks like we don't have real metadata for this observation!"
                print(f"\t{msg}")
                set_tag(job, "message", msg)
                job.jstate = "failed"
                continue

            # TODO: Make sure to make this tagging work for sat format too
            wafers = np.unique(
                [t[3:] for t in obs["tags"] if t[:2] == obs["tube_slot"]] + forced_ws
            )

            # Generally want to force because you dont know if youre actually scanning the wafer slot you think you are
            # Less relevant for SATs but true for LAT. Will need to play around to see when this is truly necessary for SATs.
            if ws not in wafers:
                msg = "Wafer not targetting or forced to be fit!"
                print(f"\t{msg}")
                set_tag(job, "message", msg)
                job.jstate = "failed"
                continue

            if h5_file is not None and myrank == 0 and obs["obs_id"] not in h5_file:
                h5_file.create_group(obs["obs_id"])

            # Load and process the TOD
            try:
                err, _, _, aman = preproc_or_load_group(
                    obs["obs_id"],
                    preprocess_cfg,
                    dets={"wafer_slot": ws},
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
            if aman.samps.count < min_samps * ds:
                msg = f"Not enough samples! Skipping..."
                print(f"\t{msg}")
                set_tag(job, "message", msg)
                job.jstate = "failed"
                continue

            # Downsample
            aman.signal = aman.signal.astype(np.float32)
            aman = downsample_obs(aman, ds)

            # Source flags
            if src_msk:
                print_once("\tRunning source flags")
                start, stop = src_flag_cut(source, aman, nominal, ufm, res, mask)
                msg = ""
                if start < 0 or stop < 0:
                    if not args.no_fit:
                        msg = "No samples flagged in source flags!"
                        to_skip = True
                    else:
                        print(
                            f"\t\tNo samples flagged! But running in no_fit mode so will continue with all samples"
                        )
                        start = 0
                        stop = int(cast(int, aman.samps.count))
                if stop - start < min_samps:
                    if not args.no_fit:
                        msg = "Too few samples flagged in source flags!"
                        to_skip = True
                    else:
                        print(
                            f"\t\tOnly {stop-start} flagged samples! But running in no_fit mode so will continue"
                        )
                if to_skip:
                    print(f"\t\t{msg}")
                    set_tag(job, "message", msg)
                    job.jstate = "failed"
                    continue
                print_once(f"\t\t{stop - start} samps flagged in the source range")
                aman = aman.restrict(
                    "samps",
                    slice(
                        start + cast(int, aman.samps.offset),
                        stop + cast(int, aman.samps.offset),
                    ),
                )

            # Setup plot dirs
            tod_plot_dir = os.path.join(
                plot_dir, "tods", str(obs["timestamp"])[:5], obs["obs_id"]
            )
            fit_plot_dir = os.path.join(
                plot_dir, "fits", str(obs["timestamp"])[:5], obs["obs_id"]
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
            sucess = False
            msg = ""
            for band in np.unique(bp):
                if msg != "":
                    msg += " "
                band_name = band_names[tube_band][band]
                print_once(f"\tFitting {band_name}")
                aman = aman_full.restrict("dets", bp == band, in_place=False)

                # Filter
                filt = tod_ops.filters.high_pass_butter4(hp_fc * 2)
                sig_filt = tod_ops.filters.fourier_filter(aman, filt)

                # Trim edges in case of FFT ringing
                aman = aman.restrict(
                    "samps", slice(trim_samps + aman.samps.offset, -1 * trim_samps)
                )
                sig_filt = sig_filt[:, trim_samps : (-1 * trim_samps)]

                # Kill dets with really high noise
                std = np.std(sig_filt, axis=-1)
                # Threshold determined from config file n_med
                # These numbers are true for LAT; unclear for SAT.
                # TODO: Make histograms of std and how much is getting cut as a part of diagnostic outputs [from talking to Matthew]. Otherwise this is a good way to cut outliers.
                # Since boxy instead of gaussians, so quantiles are cool. Pseudo sigma is e.g. ~33%-50% [median based thing so that outliers are not included]. Then can do e.g. a 4sigma cut or whatever (but shouldnt have a high percentage of outliers).
                thresh = n_med * np.median(std[std > 0])
                aman.restrict("dets", std < thresh)
                sig_filt = sig_filt[std < thresh]
                if aman.dets.count < min_dets:
                    msg += f"{band_name} Noise too high."
                    print(f"\t\t{msg}")
                    continue

                # Get median std of all dets after cuts
                std_all = np.median(std[(std < thresh) * (std > 0)])

                # TODO: Unclear if this works well for SAT.
                # Check for places where signal is high compared to noise (ie how many times the std)
                # Might not work for SAT if SNR is too low. CHECK THIS.

                # Lets try to find the source blind
                flagged = sig_filt > n_std * std_all
                samp_idx = np.where(np.any(flagged, 0))[0]

                # Commented stuff here is trying to be clever but giving up and doing the simpler method
                # ptp = np.array(
                #     np.atleast_2d(np.ptp(sig_filt, axis=0)), dtype=np.float32, order="C"
                # )
                # buf = np.zeros_like(ptp, order="C")
                # block_moment(ptp, buf, block_size, 1, 0, 0)
                # buf = buf[0]
                # samp_idx = np.where(buf > n_std * std_all)[0]

                # TODO: This needs some better logic
                # TODO: Keep the block with the highest sum?
                # Lets kill spurs by only keeping chunks that are mostly continous
                # Spur definition: Glitch leftovers effectively (ie. samples with high signal randomly that are not sources).
                # GLitch + fast jumps finder is NOT run on planet data for lat because sources look like glitches!
                # TODO: ASK matthew about this confusion between glitch and jump vs planet for e.g. lat (and sat for fast jump)
                if len(samp_idx) > 2 * block_size:
                    diff_idx = np.diff(samp_idx, prepend=1)
                    m = np.r_[False, diff_idx < block_size, False]
                    idx = np.flatnonzero(m[:-1] != m[1:])
                    max_idx = (idx[1::2] - idx[::2]).argmax()
                    samp_idx = samp_idx[idx[2 * max_idx] : idx[2 * max_idx + 1]]
                    print(f"\t\tFound {len(samp_idx)} continously flagged samples")

                # Not enough samples flagged => won't bother fitting
                if len(samp_idx) < min(block_size, min_samps / 2):
                    if args.no_fit:
                        print(
                            "\t\tLooks like you didn't see the source at all! But running in no_fit mode so will continue"
                        )
                    else:
                        msg += f"{band_name} Failed to find source blind."
                        print(f"\t{msg}")
                        continue
                start = int(max(0, np.percentile(samp_idx, 10) - (block_size * 5)))
                stop = int(
                    min(
                        cast(int, aman.samps.count),
                        np.percentile(samp_idx, 90) + (block_size * 5),
                    )
                )
                if stop - start < min_samps:
                    if not args.no_fit:
                        msg = f"{band_name} Too few samples found in blind flagging."
                        print(f"\t{msg}")
                        continue
                    print(
                        f"\t\tOnly {stop-start} flagged samples! But running in no_fit mode so will continue"
                    )
                print(f"\t\t{stop - start} samps flagged blind")
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
                msk = fake_fit + ((ptp > thresh) * (ptp > n_std * std) * (std > 0))
                aman = aman.restrict("dets", msk)
                sig_filt = sig_filt[msk]
                if aman.dets.count < min_dets:
                    msg = f"{band_name} Too few detectors after final sanity check."
                    print(f"\t{msg}")
                    continue

                print(f"\t\tAttempting to fit {tot_dets} detectors")

                # Plot the TOD
                plot_tod(aman, sig_filt, tod_plot_dir, f"{ufm}_{band_name}")
                if args.no_fit:
                    msg += "{band_name} Ran in no fit mode"
                    continue

                # Make the fft a fast length (ie like a prime number)
                _ = tod_ops.filters.fft_trim(aman, prefer="center")
                if aman.dets.count > 10 and args.profile:
                    print("\tRestricting to 10 dets for profile")
                    aman.restrict("dets", aman.dets.vals[:10])
                focal_plane = fit_tod_pointing(
                    aman,
                    (hp_fc, lp_fc),
                    fwhm=np.deg2rad(fwhm[band_name] / 60.0),
                    source=source_name,
                    show_tqdm=False,
                    **fit_pars,
                )

                # Do a quick cut based on FWHM tol
                focal_plane.restrict(
                    "dets",
                    np.abs(1 - focal_plane.fwhm / np.deg2rad(fwhm[band_name] / 60))
                    < fwhm_tol,
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
                msg += "{band_name} Success!"

            # Get ready to save
            if args.no_fit:
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
                if msg == "":
                    msg += " "
                msg += "ResultSet empty somehow!"
                set_tag(job, "message", msg)
                job.jstate = "failed"
                continue

            # Kill bad fits
            focal_plane = rset.to_axismanager(axis_key="dets:readout_id")
            # Source should be positive in pW
            msk = np.array(focal_plane.amp) > 0
            # Kill fits that are statistically bad
            msk *= np.array(focal_plane.reduced_chisq < max_chisq)
            med_xi = np.median(np.array(focal_plane.xi[msk]))
            med_eta = np.median(np.array(focal_plane.eta[msk]))
            msk *= (
                np.sqrt(
                    (np.array(focal_plane.xi) - med_xi) ** 2
                    + np.abs(np.array(focal_plane.eta) - med_eta) ** 2
                )
                < ufm_rad
            )
            msk *= np.array(focal_plane.amp) < n_med * np.median(
                np.array(focal_plane.amp[msk])
            )
            msk *= (
                np.array(focal_plane.amp)
                > np.median(np.array(focal_plane.amp[msk])) / n_med**2
            )
            # How many times it saw the source (ie. hits).
            msk *= np.array(focal_plane.hits) >= min_hits

            # Instead of cutting the rset we set R2 to 0
            # This is because det match does not like missing dets
            rset = rset.asarray()
            rset[~msk]["R2"] = 0.0
            rset = metadata.ResultSet.from_friend(rset)
            focal_plane.restrict("dets", msk)  # Only used for plotting

            if len(rset) == 0 or np.sum(msk) < min_dets:
                to_save = (None, None, None)
                if msg == "":
                    msg += " "
                msg += "Too many bad fits!"
                set_tag(job, "message", msg)
                job.jstate = "failed"
                continue
            # Plot focal plane, encoders, and a histrogram of fhwp, amp, hits
            # TODO: Split by band?
            plot_focal_plane(focal_plane, fit_plot_dir, ufm)

            # Ready to save
            print(f"\tSaving {len(rset)} fits ({np.sum(msk)} good).")
            if pad:
                all_dets = ctx.get_det_info(obs["obs_id"], dets={"stream_id": ufm})[
                    "readout_id"
                ]
                pad_dets = all_dets[~np.isin(all_dets, rset["dets:readout_id"])]
                if outdt[0][1] is None:
                    outdt[0][1] = pad_dets.dtype
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
                set_tag(job, "message", msg)
                job.jstate = "open"
                continue
            job.jstate = "done"

    if h5_file is not None:
        h5_file.close()
    nominal.close()

    if args.profile:
        profiler.stop()
        profiler.write_html(f"profile_{myrank}.html")


if __name__ == "__main__":
    print_once("Starting TOD fitting ...")
    main()
    print_once("Finished.")
