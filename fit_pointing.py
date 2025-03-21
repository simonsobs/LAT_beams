"""
More generic fitting script.
Still somewhat LAT specific but could be genralized if desired.
"""
from sotodlib.core import Context, AxisManager
from lat_beams import beam as lb
import matplotlib.pyplot as plt
import numpy as np
import os
from sotodlib import tod_ops
from sotodlib.core import metadata
from sotodlib.io.metadata import write_dataset
from functools import reduce
from sotodlib.coords import planets as cp
from lat_beams.fitting import pointing_quickfit
from sotodlib import tod_ops as to
import argparse
from so3g import block_moment
import yaml
import h5py
import sys
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

band_names = {"m" : ["f090", "f150"], "u" : ["f220", "f280"]}
fwhm = {"f090": 2, "f150":1.3, "f220": .95,  "f280": .83} #arcmin

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

parser = argparse.ArgumentParser()
parser.add_argument("cfg", help="Path to the config file")
parser.add_argument("--obs_ids", nargs='+', help="Pass a list of obs ids to run on")
parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite an existing fit")
parser.add_argument("--no_fit", "-n", action="store_true", help="Just plot TODs and don't fit")
parser.add_argument('--forced_ws',"-ws",  nargs='+', help='Force these wafer slots into the fit')
parser.add_argument('--start_from',"-s", default=0, type=int, help='Skip to the nth obs (0 indexed)')
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

if args.no_fit:
    print_once("Running in 'no_fit' mode. TOD plots will be made but pointing will not be fit")

# Get some global settings
forced_ws = args.forced_ws
if cfg.get("try_all", False):
    forced_ws = ["ws0", "ws1", "ws2"]
ds = cfg.get("ds", 5)
hp_fc = cfg.get("hp_fc", 4)
n_med = cfg.get("n_med", 3)
source = cfg.get("source", "mars")
xi_off = cfg.get("xi_off", 0.)
eta_off = cfg.get("eta_off", 0.)
min_samps = cfg.get("min_samps", 1000)/ds
block_size = cfg.get("block_size", 5000)//ds
min_dets = cfg.get("min_dets", 30)

# Setup folders
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "first_light_unfocused")
plot_dir = os.path.join(root_dir, "plots", project_dir, "source_fits", source)
data_dir = os.path.join(root_dir, "data", project_dir, "source_fits")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Get the list of observations
ctx = Context(cfg.get("context", '/so/metadata/lat/contexts/smurf_detcal.yaml'))
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
if args.obs_ids is not None:
    obslist = [ctx.obsdb.get(obs_id) for obs_id in args.obs_ids]
else:
    obslist = ctx.obsdb.query(f"type=='obs' and subtype=='cal' and {source} and start_time > {cfg['start_time']} and stop_time < {cfg['stop_time']}", tags=[f'{source}=1'])

# Output metadata setup
h5_path = os.path.join(data_dir, "tod_fits.h5")
h5_file = None
if myrank == 0:
    h5_file = h5py.File(h5_path, 'a')
db_path = os.path.join(data_dir, "db.sqlite")
if not os.path.isfile(db_path) and myrank == 0:
    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("obs:obs_id")
    scheme.add_exact_match("dets:stream_id")
    scheme.add_data_field('dataset')
    metadata.ManifestDb(scheme=scheme).to_file(db_path)
db = None
if myrank == 0:
    db = metadata.ManifestDb(db_path)
outdt = [("dets:readout_id", None), ("xi", np.float32), ("eta", np.float32), ("gamma", np.float32), ("fwhm", np.float32), ("amp", np.float32)]

# Load nominal pointing
nominal_path = os.path.expanduser(cfg.get("nominal", "~/data/pointing/lat/nominal/focal_plane.h5"))
nominal = h5py.File(nominal_path)

# Get settings for source mask
res = cfg.get("res", (2/300.)*np.pi/180.)
mask = cfg.get("mask", {'shape': 'circle', 'xyr': (0,0, .75)})

print_once(f"{len(obslist)} observations to fit")
for i, obs in enumerate(obslist):
    if i < args.start_from:
        continue
    comm.barrier()
    print_once(f"Fitting {obs['obs_id']} ({i+1}/{len(obslist)})")

    obs = ctx.obsdb.get(obs['obs_id'], tags=True)
    wafers = [t[3:] for t in obs['tags'] if t[:2] == obs['tube_slot']]
    if len(wafers) == 0 and len(forced_ws) == 0:
        print_once("\tObservation not tagged as seeing the source and not wafers forced, skipping...")
        continue
    else:
        print_once(f"\tObservation tagged for wafer slots: {wafers}, fitting only those plus the forced wafers: {forced_ws}")
    wafers = np.unique(wafers + forced_ws)

    if h5_file is not None and myrank == 0 and obs['obs_id'] not in h5_file:
        h5_file.create_group(obs['obs_id'])
    meta = ctx.get_obs(obs['obs_id'])
    meta.restrict("dets", np.isin(meta.det_info.wafer_slot, wafers))
    meta.restrict("dets", meta.det_cal.bg > -1)
    meta.restrict("dets", np.isfinite(meta.det_cal.tau_eff))

    obs_plot_dir = os.path.join(plot_dir, obs['obs_id'])
    os.makedirs(obs_plot_dir, exist_ok=True)
    ufms = np.unique(meta.det_info.stream_id)
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
        bp = (meta_ufm.det_cal.bg % 4) // 2
        tube_band = ufm[4]
        outdt[0] = ("dets:readout_id", np.array(meta.dets.vals).dtype)
        rsets = []
        for band in np.unique(bp):
            comm.barrier()
            meta_band = meta_ufm.copy().restrict("dets", bp==band)
            band_name = band_names[tube_band][band]
            print_once(f"\tFitting {ufm} {band_name}")

            # Restrict for MPI
            meta_band.restrict("dets", np.array_split(meta_band.dets.vals, nproc)[myrank])
            if meta_band.dets.count == 0:
                continue

            # Load and process the TOD
            aman = ctx.get_obs(meta_band)
            filt = tod_ops.filters.iir_filter(invert=True)
            aman.signal = tod_ops.filters.fourier_filter(aman, filt, signal_name='signal')
            filt = tod_ops.filters.timeconst_filter(timeconst = aman.det_cal.tau_eff, invert=True)
            aman.signal = tod_ops.filters.fourier_filter(aman, filt, signal_name='signal')
            aman.signal -= np.mean(np.array(aman.signal), axis=-1)[..., None]

            # jflags, _, jfix = to.jumps.twopi_jumps(aman, signal=aman.signal, fix=True, inplace=False)
            # nj = np.array([len(jflags.ranges[i].ranges()) for i in range(len(aman.signal))])
            # aman.signal = jfix
            # aman.restrict("dets", nj < 50)
            # gfilled = to.gapfill.fill_glitches(aman, nbuf=10, use_pca=False, modes=1, glitch_flags=aman.flags.jumps_2pi)
            # aman.signal = gfilled

            # gflag = to.flags.get_glitch_flags(aman, t_glitch=.00001, hp_fc=10, n_sig=50, overwrite=True)
            # ng = np.array([len(gflag.ranges[i].ranges()) for i in range(len(aman.signal))])
            # aman.restrict("dets", ng < 30)
            # gfilled = to.gapfill.fill_glitches(aman, nbuf=10, use_pca=False, modes=1, glitch_flags=aman.flags.glitches)
            # aman.signal = gfilled

            aman = lb.downsample_obs(aman, ds)
            ptp = np.ptp(aman.signal, axis=-1)
            aman = aman.restrict("dets", ptp < 2.5*np.median(ptp))
            filt = tod_ops.filters.high_pass_butter4(hp_fc*2)
            sig_filt = tod_ops.filters.fourier_filter(aman, filt)
            pip = np.ptp(sig_filt)

            # Lets get the rough std of the observations
            std_all = comm.allreduce(np.median(np.std(sig_filt, axis=-1)))/nproc

            # See how much of the source we saw...
            aman_dummy = aman.restrict("dets", [aman.dets.vals[0]], in_place=False)
            fp = AxisManager(aman_dummy.dets)
            fp.wrap("xi", np.zeros(1) + xi_off + np.nanmean(np.array(nominal[ufm]["xi"][:])), [(0, "dets")])
            fp.wrap("eta", np.zeros(1) + eta_off + np.nanmean(np.array(nominal[ufm]["eta"][:])), [(0, "dets")])
            fp.wrap("gamma", np.zeros(1) + np.nanmean(np.array(nominal[ufm]["gamma"][:])), [(0, "dets")])
            aman_dummy.wrap("focal_plane", fp)
            source_flags = cp.compute_source_flags(tod=aman_dummy, P=None, mask=mask, center_on='mars', res=res*10, max_pix=4e8, wrap=None)
            if len(source_flags.ranges[0].ranges()) == 0:
                if not args.no_fit:
                    print_once("\t\tNo samples flagged! Skipping...")
                print_once("\t\tNo samples flagged! But running in no_fit mode so will continue with all samples")
                start = 0
                stop = int(aman.samps.count)
            else:
                start = source_flags.ranges[0].ranges()[0][0]
                stop = source_flags.ranges[0].ranges()[-1][-1]
            if stop - start < min_samps:
                if not args.no_fit:
                    print_once(f"\t\tOnly {stop-start} flagged samples... skipping")
                    continue
                print_once(f"\t\tOnly {stop-start} flagged samples! But running in no_fit mode so will continue")
            else:
                print_once(f"\t\t{stop - start} samps flagged in the source range")
            aman = aman.restrict("samps", (start, stop))
            sig_filt = sig_filt[:, start:stop]

            # Kill dets with really high noise
            std = np.std(sig_filt, axis=-1)
            thresh = n_med*np.median(std)
            aman.restrict("dets", std < thresh) 
            sig_filt = sig_filt[std < thresh]

            # Lets try to find the source blind
            ptp = np.array(np.atleast_2d(np.ptp(sig_filt, axis=0)), dtype=np.float32, order='C')
            buf = np.zeros_like(ptp, order='C')
            block_moment(ptp, buf, block_size, 1, 0, 0)
            buf = buf[0]
            samp_idx = np.where(buf > 5*std_all)[0]
            # Lets sync up all the MPI procs
            samp_idx = np.unique(np.hstack(comm.allgather(samp_idx)).ravel())
            if len(samp_idx) < min(block_size, min_samps/2):
                if args.no_fit:
                    print_once("\t\tLooks like you didn't see the source at all! But running in no_fit mode so will continue")
                else:
                    print_once("\t\tLooks like you didn't see the source at all! Skipping")
                    continue
            else:
                start = int(max(0, np.percentile(samp_idx, 10)-(block_size*3)))
                stop = int(min(aman.samps.count, np.percentile(samp_idx, 90)+(block_size*3)))
                if stop - start < min_samps:
                    if not args.no_fit:
                        print_once(f"\t\tOnly {stop-start} flagged samples... skipping")
                        continue
                    print_once(f"\t\tOnly {stop-start} flagged samples! But running in no_fit mode so will continue")
                else:
                    print_once(f"\t\t{stop - start} samps flagged blind")
                aman = aman.restrict("samps", (start, stop))
                sig_filt = sig_filt[:, start:stop]

            # Do some final cuts to kill dets that didn't see the source
            ptp = np.ptp(sig_filt, axis=-1)
            thresh = .1*np.percentile(ptp, 90)
            thresh = comm.allreduce(thresh, op=MPI.MAX)
            msk = (ptp > thresh) * (ptp > 10*std_all)
            aman = aman.restrict("dets", msk)
            sig_filt = sig_filt[msk]

            # Check how many detectors we have
            num_dets = np.array(comm.allgather(aman.dets.count))
            tot_dets = np.sum(num_dets)
            if tot_dets < min_dets:
                print_once(f"\t\tOnly {tot_dets} detectors! Skipping...")
                continue

            # The proc with the most detectors should plot and tqdm
            max_det_rank = np.argmax(num_dets)

            # Plot the TOD
            if myrank == max_det_rank:
                plt.close()
                plt.plot(np.array(aman.signal).T, alpha=.3)
                plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_{band_name}_tod.png"))
                plt.close()
                plt.plot(sig_filt.T, alpha=.3)
                plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_{band_name}_tod_filt.png"))
                plt.close()
                plt.plot(buf, alpha=.3)
                plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_{band_name}_tod_ptp.png"))

            if args.no_fit:
                continue

            # If my MPI proc has no good dets
            # TODO: Even out the dets at this point?
            if aman.dets.count == 0:
                continue

            # Fit
            # TODO: include some sort of goodness of fit metric? Normalized chisq?
            focal_plane = pointing_quickfit(aman, (4, None), fwhm=np.deg2rad(fwhm[band_name]/60.), source=source, bin_priors=True, show_tqdm=(myrank==max_det_rank))

            # Convert to rset
            sarray = np.fromiter(
                    zip(
                        np.array(focal_plane.dets.vals),
                        np.array(focal_plane.xi, dtype=np.float32),
                        np.array(focal_plane.eta, dtype=np.float32),
                        np.array(focal_plane.gamma, dtype=np.float32),
                        np.array(focal_plane.fwhm, dtype=np.float32),
                        np.array(focal_plane.amp, dtype=np.float32),
                        ),
                    dtype=outdt,
                    count = np.sum(msk),
                    )
            rsets += [metadata.ResultSet.from_friend(sarray)]

        # Get rsets from everyone
        comm.barrier()
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
        rsets = reduce(lambda q,p: p+q, rsets)
        if len(rsets) == 0:
            fake_res = True
            rsets = [metadata.ResultSet.from_friend(np.zeros(1, dtype=outdt))]
        rset = reduce(lambda q,p: p+q, rsets)
        if len(rset) == 0:
            fake_res = True
            rset = metadata.ResultSet.from_friend(np.zeros(1, dtype=outdt))
        
        # Kill bad fits
        if not fake_res:
            focal_plane = rset.to_axismanager(axis_key="dets:readout_id")
            msk =  np.array(focal_plane.amp) > 0
            med_xi = np.median(np.array(focal_plane.xi[msk]))
            med_eta = np.median(np.array(focal_plane.eta[msk]))
            msk *= np.sqrt((np.array(focal_plane.xi) - med_xi)**2 + np.abs(np.array(focal_plane.eta) - med_eta)**2) < np.deg2rad(.22) 
            msk *= np.array(focal_plane.amp) < n_med*np.median(np.array(focal_plane.amp[msk]))
            msk *= np.array(focal_plane.amp) > np.median(np.array(focal_plane.amp[msk])/n_med)
            msk *= np.array(focal_plane.fwhm) < n_med*np.median(np.array(focal_plane.fwhm[msk]))
            msk *= np.array(focal_plane.fwhm) > np.median(np.array(focal_plane.fwhm[msk])/n_med)
            focal_plane.restrict("dets", msk)
            rset = rset.subset(rows = msk)
            # Plot
            # TODO: Split by band?
            plt.close()
            plt.scatter(np.array(focal_plane.xi), np.array(focal_plane.eta), alpha=.25)
            plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_fp.png"))
            plt.close()
            plt.hist(np.array(focal_plane.amp), bins=30, alpha=.25)
            plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_fp_amp.png"))
            plt.close()
            plt.hist(np.array(focal_plane.fwhm), bins=30, alpha=.25)
            plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_fp_fwhm.png"))
            if len(rset) == 0:
                fake_res = True
                rset = metadata.ResultSet.from_friend(np.zeros(1, dtype=outdt))
        # Save to database
        if fake_res:
            print_once("\tNo valid fits! Writing a fake entry!")
        else:
            print_once(f"\tSaving {len(rset)} fits.")
        write_dataset(rset, h5_file, f"{obs['obs_id']}/{ufm}", True)
        db.add_entry(params={"obs:obs_id" : obs['obs_id'], "dets:stream_id": ufm, "dataset": f"{obs['obs_id']}/{ufm}"}, filename="tod_fits.h5", replace=True)
        h5_file.flush()
    # Just to be safe
    if i%5 == 0 and i > 0 and h5_file is not None:
        print_once("Reloading h5 file to be safe!")
        h5_file.close()
        h5_file = h5py.File(h5_path, 'a')


if h5_file is not None:
    h5_file.close()
nominal.close()
