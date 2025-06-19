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
# - cuts, statistics on cuts basically
# - coverage showing planet trajectory through focal plane [just use/modify the matthew functions]
# - separate out the "special processing" so that can be it's own thing, but other features be more generic/modular/inlibrary
# - single det capabilities
# - should develop tests for this

import argparse
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
from sotodlib.obs_ops.utils import correct_iir_params
from typing_extensions import cast

from lat_beams import beam as lb
from lat_beams.fitting import fit_tod_pointing

mpi4py.rc.threads = False
from mpi4py import MPI

# TODO: Add optional argument to profile
# from pyinstrument import Profiler
# profiler = Profiler()

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

band_names = {"m": ["f090", "f150"], "u": ["f220", "f280"]}

# TODO: fwhm should come from a config file; needs to be sat vs lat here 
fwhm = {"f090": 2, "f150": 1.3, "f220": 0.95, "f280": 0.83}  # arcmin

# TODO: Add this info to docstring. Regular prints during MPI processes will print all of them. This just does the rank0 (zeroth order) one.
def print_once(*args):
    """
    Helper function to print only once when running with MPI.
    Only the rank 0 process will print.

    Parameters
    ----------
    *args : Unpack[tuple[Any, ...]]
        Arguments to pass to print.
    """
    if comm.Get_rank() == 0:
        print(*args)
        sys.stdout.flush()

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

if args.no_fit:
    print_once(
        "Running in 'no_fit' mode. TOD plots will be made but pointing will not be fit"
    )

# Get some global settings
forced_ws = args.forced_ws
if cfg.get("try_all", False):
    forced_ws = ["ws0", "ws1", "ws2"]
ds = cfg.get("ds", 5)
hp_fc = cfg.get("hp_fc", 4)
n_med = cfg.get("n_med", 5)
n_std = cfg.get("n_std", 10)
source = cfg.get("source", "mars")
xi_off = cfg.get("xi_off", 0.0)
eta_off = cfg.get("eta_off", 0.0)
min_samps = cfg.get("min_samps", 1000) / ds
block_size = cfg.get("block_size", 5000) // ds
min_dets = cfg.get("min_dets", 30)
trim_samps = cfg.get("time_samps", 200) // ds
min_hits = cfg.get("min_hits", 0)
fit_pars = cfg.get("fit_pars", {})
src_msk = cfg.get("src_msk", True)

# Setup folders
# TODO: This is currently default for LAT and needs to be globalized. Can be overwritten from config.
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "pointing/lat")
plot_dir = os.path.join(root_dir, "plots", project_dir, "source_fits", source)
data_dir = os.path.join(root_dir, "data", project_dir, "source_fits")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Get the list of observations
# TODO: Again, default is LAT here
ctx = Context(cfg.get("context", "/so/metadata/lat/contexts/smurf_detcal.yaml"))
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

# Output metadata setup
h5_path = os.path.join(data_dir, "tod_fits.h5")
h5_file = None
# Only rank0 does any writing to file.
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
]

# Load nominal pointing [i.e. template pointing from the zemax model
# TODO: Get all of these templates from Saianeesh and make sure accessible for all platforms for future uses.
nominal_path = os.path.expanduser(
    cfg.get("nominal", "~/data/pointing/lat/nominal/focal_plane.h5")
)
nominal = h5py.File(nominal_path)

# Get settings for source mask
res = cfg.get("res", (2 / 300.0) * np.pi / 180.0)
mask = cfg.get("mask", {"shape": "circle", "xyr": (0, 0, 0.75)})
# profiler.start()
print_once(f"{len(obslist)} observations to fit")
for i, obs in enumerate(obslist):
    if i < args.start_from:
        continue
    comm.barrier()
    print_once(f"Fitting {obs['obs_id']} ({i+1}/{len(obslist)})")

    obs = ctx.obsdb.get(obs["obs_id"], tags=True)
    # TODO: Make sure to make this tagging work for sat format too
    wafers = [t[3:] for t in obs["tags"] if t[:2] == obs["tube_slot"]]
    # Generally want to force because you dont know if youre actually scanning the wafer slot you think you are
    # Less relevant for SATs but true for LAT. Will need to play around to see when this is truly necessary for SATs.
    if len(wafers) == 0 and len(forced_ws) == 0:
        print_once(
            "\tObservation not tagged as seeing the source and not wafers forced, skipping..."
        )
        continue
    else:
        print_once(
            f"\tObservation tagged for wafer slots: {wafers}, fitting only those plus the forced wafers: {forced_ws}"
        )
    wafers = np.unique(wafers + forced_ws)

    if h5_file is not None and myrank == 0 and obs["obs_id"] not in h5_file:
        h5_file.create_group(obs["obs_id"])
    try:
        meta = ctx.get_meta(obs["obs_id"])
    except sqlite3.OperationalError:
        time.sleep(30)
        ctx = Context(cfg.get("context", "/so/metadata/lat/contexts/smurf_detcal.yaml"))
        meta = ctx.get_meta(obs["obs_id"])
    # General cuts to see validity of the data
    meta.restrict("dets", np.isin(meta.det_info.wafer_slot, wafers))
    meta.restrict("dets", meta.det_cal.bg > -1)
    # TODO: If incorporating time constant stuff, might not want to do this here.
    # 
    meta.restrict("dets", np.isfinite(meta.det_cal.tau_eff))

    tod_plot_dir = os.path.join(
        plot_dir, "tods", str(obs["timestamp"])[:5], obs["obs_id"]
    )
    fit_plot_dir = os.path.join(
        plot_dir, "fits", str(obs["timestamp"])[:5], obs["obs_id"]
    )
    os.makedirs(tod_plot_dir, exist_ok=True)
    os.makedirs(fit_plot_dir, exist_ok=True)
    ufms = np.unique(meta.det_info.stream_id)
    print_once(f"Fitting UFMs: {ufms}")
    for ufm in ufms:
        comm.barrier()
        # Check if we already fit
        # TODO: add a mode to replot but not refit
        to_fit = True
        if myrank == 0 and h5_file is not None:
            to_fit = f"{obs['obs_id']}/{ufm}" in h5_file and not args.overwrite
        to_fit = comm.bcast(to_fit, root=0)
        if to_fit:
            print_once(f"\t{ufm} already in the output file, skipping...")
            continue
        meta_ufm = meta.copy().restrict("dets", meta.det_info.stream_id == ufm)
        # this just uses simple logic to map bias line -> band
        bp = (meta_ufm.det_cal.bg % 4) // 2
        # TODO: This variable name needs to be updated to something more global. Also should check if there is better way than grabbing a hard coded character in string.
        # This just extracts if it's m or u for mf or uhf
        tube_band = ufm[4]
        outdt[0] = ("dets:readout_id", np.array(meta.dets.vals).dtype)
        rsets = []
        # loop through bands separately because they have different responses/power coming in and might bias results at end.
        for band in np.unique(bp):
            fake_fit = False
            comm.barrier()
            meta_band = meta_ufm.copy().restrict("dets", bp == band)
            band_name = band_names[tube_band][band]
            print_once(f"\tFitting {ufm} {band_name}")

            # Restrict for MPI
            # splitting different detectors across the processes
            # this optimizes for doing one obs fast 
            # TODO: once the code is more generalized across platforms, should give observations in parallel and detectors in parallel per obs. Check MPI test within this same folder. 
            meta_band.restrict(
                "dets", np.array_split(meta_band.dets.vals, nproc)[myrank]
            )
            if meta_band.dets.count == 0:
                continue

            # Load and process the TOD
            try:
                aman = ctx.get_obs(meta_band)
            except sqlite3.OperationalError:
                time.sleep(30)
                ctx = Context(
                    cfg.get("context", "/so/metadata/lat/contexts/smurf_detcal.yaml")
                )
                aman = ctx.get_obs(meta_band)
            # Anticipate all dets being cut by making fake aman. This allows all processes to end at roughly same time.
            # TODO: Can check if there is cleaner way to do this, but Saianeesh said that there is high chance of that failing/being stuck in infinite loop.
            fake_aman = aman.restrict("dets", [aman.dets.vals[0]], in_place=False)
            fake_aman.signal[:] = 0

            if aman.samps.count < min_samps*ds:
                print_once(f"\t\tNot enough samples! Skipping...")
                continue

            # TODO: Give option to take in preprocessing if it exists.
            # TODO: If preprocessing doesnt exist, then this should have different default SAT and LAT params (config option + default option)
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
            # would not deconvolve here, but get from model fitting
            # forward modelings -> generate signal from beam model into time domain, and then apply time constant filter
            filt = tod_ops.filters.timeconst_filter(
                timeconst=aman.det_cal.tau_eff, invert=True
            )
            aman.signal = tod_ops.filters.fourier_filter(
                aman, filt, signal_name="signal"
            )

            tod_ops.detrend_tod(aman, "median", in_place=True)
            tf = tod_ops.flags.get_trending_flags(aman, max_trend=5, t_piece=min(30, aman.obs_info.duration/2))
            tdets = has_any_cuts(tf)
            aman.restrict("dets", ~tdets)

            if aman.dets.count == 0:
                fake_fit = True
                aman = fake_aman.copy()

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

            if aman.dets.count == 0:
                fake_fit = True
                aman = fake_aman.copy()
            tod_ops.detrend_tod(aman, "linear", in_place=True)
            aman = lb.downsample_obs(aman, ds)
            fake_aman = lb.downsample_obs(fake_aman, ds)
            ptp = np.ptp(np.array(aman.signal), axis=-1)
            ptp_all = np.hstack(comm.allgather(ptp))
            aman = aman.restrict("dets", fake_fit + (ptp < n_med * np.median(ptp_all)))
            # No detectors left after preprocessing so use dummy aman to allow computations later.
            if aman.dets.count == 0 and not fake_fit:
                fake_fit = True
                aman = fake_aman.copy()

            filt = tod_ops.filters.high_pass_butter4(hp_fc * 2)
            sig_filt = tod_ops.filters.fourier_filter(aman, filt)

            # Trim edges in case of FFT ringing
            aman = aman.restrict(
                "samps", slice(trim_samps + aman.samps.offset, -1 * trim_samps)
            )
            fake_aman = fake_aman.restrict(
                "samps", slice(trim_samps + fake_aman.samps.offset, -1 * trim_samps)
            )
            sig_filt = sig_filt[:, trim_samps : (-1 * trim_samps)]

            ####### Preprocessing ends here #######

            source_name = source
            # This is just to account for gap in sotodlib. 
            # TODO: Add it to sotodlib...
            if source == "rcw38":
                source_name = "J134.78-47.509"

            # See how much of the source we saw...
            # TODO: This can be made optional.
            # Mask is made massive. This ONLY helps if you have prior knowledge of where source is.
            # See how much of the source we saw...
            if src_msk:
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
                    if not args.no_fit:
                        print_once("\t\tNo samples flagged! Skipping...")
                        continue
                    print_once(
                        "\t\tNo samples flagged! But running in no_fit mode so will continue with all samples"
                    )
                    start = 0
                    stop = int(cast(int, aman.samps.count))
                else:
                    start = source_flags.ranges[0].ranges()[0][0]
                    stop = source_flags.ranges[0].ranges()[-1][-1]
                if stop - start < min_samps:
                    if not args.no_fit:
                        print_once(f"\t\tOnly {stop-start} flagged samples... skipping")
                        continue
                    print_once(
                        f"\t\tOnly {stop-start} flagged samples! But running in no_fit mode so will continue"
                    )
                else:
                    print_once(f"\t\t{stop - start} samps flagged in the source range")
                aman = aman.restrict(
                    "samps",
                    slice(
                        start + cast(int, aman.samps.offset),
                        stop + cast(int, aman.samps.offset),
                    ),
                )
                fake_aman = fake_aman.restrict(
                    "samps",
                    slice(
                        start + cast(int, fake_aman.samps.offset),
                        stop + cast(int, fake_aman.samps.offset),
                    ),
                )
                sig_filt = sig_filt[:, start:stop]

            # Kill dets with really high noise
            std = np.std(sig_filt, axis=-1)
            std_all = np.hstack(comm.allgather(std))
            # Threshold determined from config file n_med
            # TODO: Should do better rather than pic a number by eye like Saianeesh since not all of us are gods. He normally sets it to like 5.
            # He says that 2-3sigma like I want sometimes wont be consistent because the noise is not gaussian and is more like a block. Need to investigate this further.
            # These numbers are true for LAT; unclear for SAT. TODO ASK MATTHEW ABOUT THIS TODO Saianeesh said he got this thing from Sigurd method but us mere mortals like equations.
            # TODO: Make histograms of std and how much is getting cut as a part of diagnostic outputs [from talking to Matthew]. Otherwise this is a good way to cut outliers.
            # Since boxy instead of gaussians, so quantiles are cool. Pseudo sigma is e.g. ~33%-50% [median based thing so that outliers are not included]. Then can do e.g. a 4sigma cut or whatever (but shouldnt have a high percentage of outliers).
            thresh = n_med * np.median(std_all[std_all > 0])
            aman.restrict("dets", fake_fit + (std < thresh))
            sig_filt = sig_filt[fake_fit + (std < thresh)]
            if aman.dets.count == 0 and not fake_fit:
                fake_fit = True
                aman = fake_aman.copy()
                sig_filt = np.zeros_like(aman.signal)

            # Check median and std of all obs after cuts
            # Lets get the rough std of the observations
            std_all = np.median(std_all[(std_all < thresh) * (std_all > 0)])

            # TODO: Unclear if this works well for SAT.
            # Check for places where signal is high compared to noise (ie how many times the std)
            # Might not work for SAT if SNR is too low. CHECK THIS.

            # Lets try to find the source blind
            flagged = sig_filt > n_std * std_all
            samp_idx = np.where(np.any(flagged, 0))[0]
            
            # Commented stuff here is Saianeesh wanting to be clever but giving up and doing the simpler method
            # ptp = np.array(
            #     np.atleast_2d(np.ptp(sig_filt, axis=0)), dtype=np.float32, order="C"
            # )
            # buf = np.zeros_like(ptp, order="C")
            # block_moment(ptp, buf, block_size, 1, 0, 0)
            # buf = buf[0]
            # samp_idx = np.where(buf > n_std * std_all)[0]

            # Lets sync up all the MPI procs
            samp_idx = np.unique(np.hstack(comm.allgather(samp_idx)).ravel())
            
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
                print_once(f"\t\tFound {len(samp_idx)} continously flagged samples")

            # Not enough samples flagged => won't bother fitting 
            if len(samp_idx) < min(block_size, min_samps / 2):
                if args.no_fit:
                    print_once(
                        "\t\tLooks like you didn't see the source at all! But running in no_fit mode so will continue"
                    )
                else:
                    print_once(
                        "\t\tLooks like you didn't see the source at all! Skipping"
                    )
                    continue
            else:
                start = int(max(0, np.percentile(samp_idx, 10) - (block_size * 5)))
                stop = int(
                    min(
                        cast(int, aman.samps.count),
                        np.percentile(samp_idx, 90) + (block_size * 5),
                    )
                )
                if stop - start < min_samps:
                    if not args.no_fit:
                        print_once(f"\t\tOnly {stop-start} flagged samples... skipping")
                        continue
                    print_once(
                        f"\t\tOnly {stop-start} flagged samples! But running in no_fit mode so will continue"
                    )
                else:
                    print_once(f"\t\t{stop - start} samps flagged blind")
                # Restricting to samples where we think we see a source now. The block above this is probs hardest to generalize between LAT and SAT
                aman = aman.restrict(
                    "samps",
                    slice(
                        start + cast(int, aman.samps.offset),
                        stop + cast(int, aman.samps.offset),
                    ),
                )
                fake_aman = fake_aman.restrict(
                    "samps",
                    slice(
                        start + cast(int, fake_aman.samps.offset),
                        stop + cast(int, fake_aman.samps.offset),
                    ),
                )
                sig_filt = sig_filt[:, start:stop]
            
            # Make a p2p cut 
            # TODO: This cut might be different for SAT and LAT.
            # Do some final cuts to kill dets that didn't see the source
            ptp = np.ptp(sig_filt, axis=-1)
            std = np.std(sig_filt, axis=-1)
            ptp_all = np.hstack(comm.allgather(ptp))
            thresh = 0.1 * np.percentile(ptp_all, 90)
            msk = fake_fit + ((ptp > thresh) * (ptp > n_std * std) * (std > 0))
            aman = aman.restrict("dets", msk)
            sig_filt = sig_filt[msk]
            if aman.dets.count == 0 and not fake_fit:
                fake_fit = True
                aman = fake_aman.copy()
                sig_filt = np.zeros_like(aman.signal)

            # Check how many detectors we have
            not_fit = comm.allreduce(int(fake_fit))
            num_dets = np.array(comm.allgather(aman.dets.count))
            tot_dets = np.sum(num_dets) - not_fit
            if tot_dets < min_dets:
                print_once(f"\t\tOnly {tot_dets} detectors! Skipping...")
                continue
            print_once(f"\t\tAttempting to fit {tot_dets} detectors")

            # The proc with the most detectors should plot and tqdm
            max_det_rank = np.argmax(num_dets)

            # Plot the TOD
            if myrank == max_det_rank:
                #plt.close()
                
                plt.plot(np.array(aman.signal).T, alpha=0.3)
                plt.xlabel('samples')
                plt.ylabel('signal')
                plt.savefig(os.path.join(tod_plot_dir, f"{ufm}_{band_name}_tod.png"))
                plt.close()
                
                plt.plot(sig_filt.T, alpha=0.3)
                plt.xlabel('sample')
                plt.ylabel('signal [filtered]')
                plt.savefig(
                    os.path.join(tod_plot_dir, f"{ufm}_{band_name}_tod_filt.png")
                )
                plt.close()
                # plt.plot(buf, alpha=0.3)
                # plt.savefig(
                #     os.path.join(tod_plot_dir, f"{ufm}_{band_name}_tod_ptp.png")
                # )

            if args.no_fit:
                continue

            # If my MPI proc has no good dets
            # TODO: Even out the dets at this point? Note: this might not be worth it.
            if aman.dets.count == 0 or fake_fit:
                continue

            # Fit
            # The type ignore is just for saianeesh ...... smh.
            aman.signal *= aman.det_cal.phase_to_pW[..., None]  # type: ignore
            # this is just to make the fft a fast length (ie like a prime number)
            _ = tod_ops.filters.fft_trim(aman, prefer="center")
            # if aman.dets.count > 10:
            #     aman.restrict("dets", aman.dets.vals[:10])
            focal_plane = fit_tod_pointing(
                aman,
                (4, 30),
                fwhm=np.deg2rad(fwhm[band_name] / 60.0),
                source=source_name,
                show_tqdm=(myrank == max_det_rank),
                **fit_pars,
            )

            # Convert to rset [results set]
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
                ),
                dtype=outdt,
                count=focal_plane.dets.count,
            )
            rsets += [metadata.ResultSet.from_friend(sarray)]

        # Get rsets from everyone and send to 0th rank process to write out.
        comm.barrier()
        if args.no_fit:
            continue
        rsets = comm.gather(rsets, root=0)
        if myrank != 0:
            continue
        if rsets is None or db is None or h5_file is None:
            continue
        # If we have no results lets make an empty one so we know to skip in the future
        fake_res = False
        if len(rsets) == 0:
            fake_res = True
            rsets = [[metadata.ResultSet.from_friend(np.zeros(1, dtype=outdt))]]
        rsets = reduce(lambda q, p: p + q, rsets)
        if len(rsets) == 0:
            fake_res = True
            rsets = [metadata.ResultSet.from_friend(np.zeros(1, dtype=outdt))]
        rset = reduce(lambda q, p: p + q, rsets)
        if len(rset) == 0:
            fake_res = True
            rset = metadata.ResultSet.from_friend(np.zeros(1, dtype=outdt))

        # Kill bad fits
        if not fake_res:
            focal_plane = rset.to_axismanager(axis_key="dets:readout_id")
            # Source should be positive in pW 
            msk = np.array(focal_plane.amp) > 0
            # TODO: needs to be changed for SAT or put in config file. Checks if source is far from median of dets
            # Right now it is set for LAT UFM diamater.
            med_xi = np.median(np.array(focal_plane.xi[msk]))
            med_eta = np.median(np.array(focal_plane.eta[msk]))
            # TODO: All equations below should be re-tuned for SAT. Comparative FWHM and amps can be put in config file.
            msk *= (
                np.sqrt(
                    (np.array(focal_plane.xi) - med_xi) ** 2
                    + np.abs(np.array(focal_plane.eta) - med_eta) ** 2
                )
                < 0.01
            )
            msk *= np.array(focal_plane.amp) < n_med * np.median(
                np.array(focal_plane.amp[msk])
            )
            msk *= (
                np.array(focal_plane.amp)
                > np.median(np.array(focal_plane.amp[msk])) / n_med**2
            )
            msk *= np.array(focal_plane.fwhm) < n_med * np.median(
                np.array(focal_plane.fwhm[msk])
            )
            msk *= (
                np.array(focal_plane.fwhm)
                > np.median(np.array(focal_plane.fwhm[msk])) / n_med**2
            )
            # How many times it saw the source (ie. hits). 
            # TODO: This needs to be fixed. Right now, min_hits is set to zero until debugged.
            msk *= np.array(focal_plane.hits) >= min_hits
            focal_plane.restrict("dets", msk)
            rset = rset.subset(rows=msk)
            # Plot focal plane, encoders, and a histrogram of fhwp, amp, hits 
            # TODO: Split by band?
            plt.close()
            
            plt.scatter(np.array(focal_plane.xi), np.array(focal_plane.eta), alpha=0.25)
            plt.xlabel('xi')
            plt.ylabel('eta')
            plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp.png"))
            plt.close()
            
            plt.scatter(np.array(focal_plane.az), np.array(focal_plane.el), alpha=0.25)
            plt.xlabel('az')
            plt.ylabel('el')
            plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_enc.png"))
            plt.close()
            
            plt.hist(np.array(focal_plane.amp), bins=30, alpha=0.25)
            plt.xlabel('amp')
            plt.ylabel('dets')
            plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp_amp.png"))
            plt.close()
            
            plt.hist(np.array(focal_plane.fwhm), bins=30, alpha=0.25)
            plt.xlabel('fwhm')
            plt.ylabel('dets')
            plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp_fwhm.png"))
            plt.close()
            
            plt.hist(np.array(focal_plane.hits), bins=30, alpha=0.25)
            plt.xlabel('hits')
            plt.ylabel('dets')
            plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp_hits.png"))
            plt.close()
            
            plt.hist(np.array(focal_plane.reduced_chisq), bins=30, alpha=0.25)
            plt.xlabel('reduced chi sq')
            plt.ylabel('dets')
            plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp_red_chisq.png"))
            plt.close()
            
            if len(rset) == 0:
                fake_res = True
                rset = metadata.ResultSet.from_friend(np.zeros(1, dtype=outdt))
        # Save to database
        if fake_res:
            print_once("\tNo valid fits! Writing a fake entry!")
        else:
            print_once(f"\tSaving {len(rset)} fits.")
        write_dataset(rset, h5_file, f"{obs['obs_id']}/{ufm}", True)
        db.add_entry(
            params={
                "obs:obs_id": obs["obs_id"],
                "dets:stream_id": ufm,
                "dataset": f"{obs['obs_id']}/{ufm}",
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


if h5_file is not None:
    h5_file.close()
nominal.close()

# profiler.stop()
# profiler.write_html(f"profile_{myrank}.html")
