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
import argparse
import yaml
import h5py

band_names = {"m" : ["f090", "f150"], "u" : ["f220", "f280"]}
fwhm = {"f090": 2, "f150":1.3, "f220": .95,  "f280": .83} #arcmin

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

parser = argparse.ArgumentParser()
parser.add_argument("cfg", help="Path to the config file")
parser.add_argument("--obs_id", help="Pass a single ob id to run on")
parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite an existing fit")
parser.add_argument("--no_fit", "-n", action="store_true", help="Just plot TODs and don't fit")
parser.add_argument('--forced_ws',"-ws",  nargs='+', help='Force these wafer slots into the fit')
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

if args.no_fit:
    print("Running in 'no_fit' mode. TOD plots will be made but pointing will not be fit")

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
if args.obs_id is not None:
    obslist = [ctx.obsdb.get(args.obs_id)]
else:
    obslist = ctx.obsdb.query(f'type=="obs" and subtype=="cal" and {source} and start_time > {cfg["start_time"]} and stop_time < {cfg["stop_time"]}', tags=[f'{source}=1'])

# Output metadata setup
h5_path = os.path.join(data_dir, "tod_fits.h5")
h5_file = h5py.File(h5_path, 'a')
db_path = os.path.join(data_dir, "db.sqlite")
if not os.path.isfile(db_path):
    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("obs:obs_id")
    scheme.add_exact_match("dets:stream_id")
    scheme.add_data_field('dataset')
    metadata.ManifestDb(scheme=scheme).to_file(db_path)
db = metadata.ManifestDb(db_path)
outdt = [("dets:readout_id", None), ("xi", np.float32), ("eta", np.float32), ("gamma", np.float32), ("fwhm", np.float32), ("amp", np.float32)]

# Load nominal pointing
nominal_path = cfg.get("nominal", "/so/home/saianeesh/data/pointing/lat/nominal/focal_plane.h5")
nominal = h5py.File(nominal_path)

# Get settings for source mask
res = cfg.get("res", (2/300.)*np.pi/180.)
mask = cfg.get("mask", {'shape': 'circle', 'xyr': (0,0, .75)})

print(f"{len(obslist)} observations to fit")
for i, obs in enumerate(obslist):
    print(f"Fitting {obs['obs_id']} ({i+1}/{len(obslist)})")

    obs = ctx.obsdb.get(obs['obs_id'], tags=True)
    wafers = [t[3:] for t in obs['tags'] if t[:2] == obs['tube_slot']]
    if len(wafers) == 0 and len(forced_ws) == 0:
        print("\tObservation not tagged as seeing the source and not wafers forced, skipping...")
        continue
    else:
        print(f"\tObservation tagged for wafer slots: {wafers}, fitting only those plus the forced wafers: {forced_ws}")
    wafers = np.unique(wafers + forced_ws)

    if obs['obs_id'] not in h5_file:
        h5_file.create_group(obs['obs_id'])
    meta = ctx.get_obs(obs['obs_id'])
    meta.restrict("dets", np.isin(meta.det_info.wafer_slot, wafers))
    meta.restrict("dets", meta.det_cal.bg > -1)
    meta.restrict("dets", np.isfinite(meta.det_cal.tau_eff))

    obs_plot_dir = os.path.join(plot_dir, obs['obs_id'])
    os.makedirs(obs_plot_dir, exist_ok=True)
    ufms = np.unique(meta.det_info.stream_id)
    for ufm in ufms:
        # Check if we already fit
        # TODO: add a mode to replot but not refit
        if f"{obs['obs_id']}/{ufm}" in h5_file and not args.overwrite:
            print(f"\t{ufm} already in the output file, skipping...")
            continue
        meta_ufm = meta.copy().restrict("dets", meta.det_info.stream_id == ufm)
        bp = (meta_ufm.det_cal.bg % 4) // 2
        tube_band = ufm[4]
        outdt[0] = ("dets:readout_id", np.array(meta.dets.vals).dtype)
        rsets = []
        for band in np.unique(bp):
            meta_band = meta_ufm.copy().restrict("dets", bp==band)
            band_name = band_names[tube_band][band]
            print(f"\tFitting {ufm} {band_name}")

            # Load and process the TOD
            aman = ctx.get_obs(meta_band)
            filt = tod_ops.filters.iir_filter(invert=True)
            aman.signal = tod_ops.filters.fourier_filter(aman, filt, signal_name='signal')
            filt = tod_ops.filters.timeconst_filter(timeconst = aman.det_cal.tau_eff, invert=True)
            aman.signal = tod_ops.filters.fourier_filter(aman, filt, signal_name='signal')
            aman = lb.downsample_obs(aman, ds)
            aman.signal -= np.mean(np.array(aman.signal), axis=-1)[..., None]
            ptp = np.ptp(aman.signal, axis=-1)
            aman = aman.restrict("dets", ptp < 2*np.median(ptp))
            filt = tod_ops.filters.high_pass_butter4(hp_fc*2)
            sig_filt = tod_ops.filters.fourier_filter(aman, filt)

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
                    print("\t\tNo samples flagged! Skipping...")
                print("\t\tNo samples flagged! But running in no_fit mode so will continue with all samples")
                start = 0
                stop = int(aman.samps.count)
            else:
                start = source_flags.ranges[0].ranges()[0][0]
                stop = source_flags.ranges[0].ranges()[-1][-1]
            if stop - start < min_samps:
                if not aegs.no_fit:
                    print(f"\t\tOnly {stop-start} flagged samples... skipping")
                    continue
                print(f"\t\tOnly {stop-start} flagged samples! But running in no_fit mode so will continue")
            else:
                print(f"\t\t{stop - start} samps flagged in the source range")
            aman = aman.restrict("samps", (start, stop))
            sig_filt = sig_filt[:, start:stop]

            # Kill dets with really high noise
            std = np.std(sig_filt, axis=-1)
            thresh = n_med*np.median(std)
            aman.restrict("dets", std < thresh) 
            sig_filt = sig_filt[std < thresh]

            # Do some final cuts to kill dets that didn't see the source
            ptp = np.ptp(sig_filt, axis=-1)
            thresh = .1*np.percentile(ptp, 90)
            aman = aman.restrict("dets", ptp > thresh)
            sig_filt = sig_filt[ptp > thresh]

            # Plot the TOD
            plt.close()
            plt.plot(np.array(aman.signal[::3]).T, alpha=.3)
            plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_{band_name}_tod.png"))
            plt.close()
            plt.plot(sig_filt.T, alpha=.3)
            plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_{band_name}_tod_filt.png"))

            if args.no_fit:
                continue

            # Fit and plot
            # TODO: include some sort of goodness of fit metric? Normalized chisq?
            focal_plane = pointing_quickfit(aman, (4, None), fwhm=np.deg2rad(fwhm[band_name]/60.), source='mars', bin_priors=True)
            del aman
            med_xi = np.median(np.array(focal_plane.xi))
            med_eta = np.median(np.array(focal_plane.eta))
            msk = (np.sqrt((np.array(focal_plane.xi) - med_xi)**2 + np.abs(np.array(focal_plane.eta) - med_eta)**2) < np.deg2rad(.22)) * (np.array(focal_plane.amp) > 0)
            msk *= np.array(focal_plane.amp) < n_med*np.median(np.array(focal_plane.amp[msk]))
            msk *= np.array(focal_plane.amp) > np.median(np.array(focal_plane.amp[msk])/n_med)
            msk *= np.array(focal_plane.fwhm) < n_med*np.median(np.array(focal_plane.fwhm[msk]))
            msk *= np.array(focal_plane.fwhm) > np.median(np.array(focal_plane.fwhm[msk])/n_med)
            focal_plane.restrict("dets", msk)
            plt.close()
            plt.scatter(np.array(focal_plane.xi), np.array(focal_plane.eta))
            plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_{band_name}_fp.png"))
            plt.close()
            plt.hist(np.array(focal_plane.amp), bins=30)
            plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_{band_name}_fp_amp.png"))
            plt.close()
            plt.hist(np.array(focal_plane.fwhm), bins=30)
            plt.savefig(os.path.join(obs_plot_dir, f"{ufm}_{band_name}_fp_fwhm.png"))

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
        # Save to database
        if len(rsets) == 0:
            continue
        rset = reduce(lambda q,p: p+q, rsets)
        write_dataset(rset, h5_file, f"{obs['obs_id']}/{ufm}", True)
        db.add_entry(params={"obs:obs_id" : obs['obs_id'], "dets:stream_id": ufm, "dataset": f"{obs['obs_id']}/{ufm}"}, filename="tod_fits.h5", replace=True)
        h5_file.flush()

h5_file.close()
nominal.close()
