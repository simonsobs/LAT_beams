from sotodlib.core import Context
import h5py
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from sotodlib import tod_ops
from so3g.proj import RangesMatrix
from sotodlib.coords import planets as cp
from scipy.ndimage import gaussian_filter
from matplotlib.colors import SymLogNorm
from so3g.proj import RangesMatrix
from sotodlib.core import metadata
import sys
import argparse
import yaml
plt.rcParams['image.cmap'] = 'RdGy_r'

N_FILES = 4
band_names = {"m" : ["f090", "f150"], "u" : ["f220", "f280"]}

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
parser.add_argument("--obs_ids", nargs='+', help="Pass a list of obs ids to run on")
parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite an existing map")
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

# Get some global settings
source = cfg.get("source", "mars")
xi_off = cfg.get("xi_off", np.nan)
eta_off = cfg.get("eta_off", np.nan)
min_dets = cfg.get("min_dets", 30)
extent = cfg.get("extent", 50)
log_thresh = cfg.get("log_thresh", 1e-5)
norm = SymLogNorm(log_thresh)

# Setup folders
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "first_light_unfocused")
plot_dir = os.path.join(root_dir, "plots", project_dir, "source_maps", source)
data_dir = os.path.join(root_dir, "data", project_dir, "source_maps", source)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Get the list of observations
ctx = Context(cfg.get("context", '/so/metadata/lat/contexts/smurf_detcal.yaml'))
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
if args.obs_ids is not None:
    obslist = [ctx.obsdb.get(obs_id) for obs_id in args.obs_ids]
else:
    obslist = ctx.obsdb.query(f'type=="obs" and subtype=="cal" and {source} and start_time > {cfg["start_time"]} and stop_time < {cfg["stop_time"]}', tags=[f"{source}=1"])
print(f"Found {len(obslist} observations to map")

# Keep only the ones with a focal plane 
if cfg.get("per_obs_point", True):
    dbs = [md['db'] for md in ctx['metadata'] if "focal_plane" in md.get("name", "")]
    if len(dbs) > 1:
        print("Multiple pointing metadata entries found, using the first one")
    elif len(dbs) == 0:
        print("No pointing metadata entries found")
        sys.exit()
    print(f"Using ManifestDb at {dbs[0]}")
    db = metadata.ManifestDb(dbs[0])
    obs_ids = np.array([entry['obs:obs_id'] for entry in db.inspect()])
    obslist = [obs for obs in obslist if obs['obs_id'] in obs_ids]
    print(f"Only {len(obslist)} observations with pointing metadata")

# Load nominal pointing
nominal_path = cfg.get("nominal", "/so/home/saianeesh/data/pointing/lat/nominal/focal_plane.h5")
nominal = h5py.File(nominal_path)

# Get settings for source mask
res = cfg.get("res", (2/300.)*np.pi/180.)
mask = cfg.get("mask", {'shape': 'circle', 'xyr': (0, 0, .1)})

# Mapping loop
for i, obs in enumerate(obslist):
    print(f"Mapping {obs['obs_id']} ({i+1}/{len(obslist)})")

    obs = ctx.obsdb.get(obs['obs_id'], tags=True)
    meta = ctx.get_obs(obs['obs_id'])
    meta.restrict("dets", np.isfinite(meta.det_cal.tau_eff))
    db_flag = tod_ops.flags.get_det_bias_flags(meta)
    meta.restrict("dets", db_flag.det_bias_flags)
    meta.restrict("dets", np.isfinite(meta.det_cal.phase_to_pW))

    obs_plot_dir = os.path.join(plot_dir, obs['obs_id'])
    obs_data_dir = os.path.join(plot_dir, obs['obs_id'])
    os.makedirs(obs_plot_dir, exist_ok=True)
    os.makedirs(obs_data_dir, exist_ok=True)
    ufms = np.unique(meta.det_info.stream_id)
    for ufm in ufms:
        meta_ufm = meta.copy().restrict("dets", meta.det_info.stream_id == ufm)
        bp = (meta_ufm.det_cal.bg % 4) // 2
        tube_band = ufm[4]
        rsets = []
        for band in np.unique(bp):
            meta_band = meta_ufm.copy().restrict("dets", bp==band)
            band_name = band_names[tube_band][band]
            print(f"\tMapping {ufm} {band_name}")

            # Check if we already mapped 
            # TODO: add a mode to replot but not refit
            glob_path = os.path.join(obs_data_dir, f"{obs['obs_id']}_{ufm}_{band_name}*")
            flist = glob.glob(glob_path)
            if len(flist) >= N_FILES and not args.overwrite:
                print("\t\tMaps appear to already exist and overwrite is not set. Skipping...")

            if meta_band.dets.count == 0:
                print("\t\tNo dets! Skipping...")
                continue

            # Load and process the TOD
            aman = ctx.get_obs(meta_band)
            filt = tod_ops.filters.iir_filter(invert=True)
            aman.signal = tod_ops.filters.fourier_filter(aman, filt, signal_name='signal')
            filt = tod_ops.filters.timeconst_filter(timeconst = aman.det_cal.tau_eff, invert=True)
            aman.signal = tod_ops.filters.fourier_filter(aman, filt, signal_name='signal')
            aman.signal -= np.mean(np.array(aman.signal), axis=-1)[..., None]
            aman.signal *= aman.det_cal.phase_to_pW[..., None]
            orig = aman.copy()

            if aman.dets.count < min_dets:
                print("\t\tOnly {aman.dets.count} dets! Skipping...")
                continue

            # Its map time!
            cuts = RangesMatrix.zeros(aman.signal.shape)
            source_flags = cp.compute_source_flags(tod=aman, P=None, mask=mask, center_on='mars', res=res, max_pix=4e8, wrap=None)
            print(f"\t\t{np.sum(source_flags.get_stats()['samples'])*np.mean(np.diff(aman.timestamps))} detector seconds on source")
            out = cp.make_map(aman, center_on='mars', res=res, cuts=cuts, source_flags=source_flags, comps="T", filename=os.path.join(obs_data_dir, '{obs_id}_{ufm}_{band_name}_{map}.fits'), info={'obs_id':obs['obs_id'], 'ufm':ufm, 'band_name':band_name})

            # Smooth and find the center
            smoothed = gaussian_filter(out["solved"][0], sigma=1)
            cent = np.unravel_index(np.argmax(smoothed, axis=None), smoothed.shape)

            # Plot
            plt.close()
            plt.imshow(out["solved"][0], origin='lower')
            plt.xlim((cent[1]-extent, cent[1]+extent))
            plt.ylim((cent[0]-extent, cent[0]+extent))
            plt.colorbar()
            plt.grid()
            plt.savefig(os.path.join(plot_dir, f"{obs['obs_id']}_{ufm}_{band_name}_map.png"))
            
            plt.close()
            lognormed, _ = norm.process_values(out["solved"][0])
            lognormed[lognormed < log_thresh] = np.nan
            plt.imshow(lognormed, origin='lower')
            plt.xlim((cent[1]-extent, cent[1]+extent))
            plt.ylim((cent[0]-extent, cent[0]+extent))
            plt.colorbar()
            plt.grid()
            plt.savefig(os.path.join(plot_dir, f"{obs['obs_id']}_{ufm}_{band_name}_map_log10.png"))
            
            rprof = radial_profile(smoothed, cent[::-1])
            rlen = len(rprof)
            rprof = np.hstack([np.flip(rprof), rprof])
            plt.close()
            fig, axd = plt.subplot_mosaic([['A', 'A', 'A'], ['A', 'A', 'A'], ['A', 'A', 'A'], ['B', 'B', 'B']], layout='constrained', figsize=(7,10))
            axd['A'].imshow(smoothed[cent[0]-extent:cent[0]+extent, cent[1]-extent:cent[1]+extent], origin="lower")
            axd['B'].plot(rprof[rlen-extent:rlen+extent])
            plt.savefig(os.path.join(plot_dir, f"{obs['obs_id']}_{ufm}_{band_name}_map_smooth.png"), bbox_inches='tight')

nominal.close()
            

# Splits stuff to implement later
# xi_off = np.mean(aman.focal_plane.xi) - ws_cent[0]
# eta_off = np.mean(aman.focal_plane.xi) - ws_cent[1]
# ot_cent -= np.array([xi_off, eta_off])
# dist = np.sqrt((ot_cent[0] - aman.focal_plane.xi)**2 + (ot_cent[1] - aman.focal_plane.eta)**2)
# msk = dist > np.deg2rad(.5)
# msk = np.repeat(msk, int(aman.samps.count)).reshape(len(msk), -1)
# ranges = RangesMatrix.from_mask(msk)
# splits = {"Within .5 deg" : ranges, "Outside .5 deg" : ~ranges} 
# out = cp.make_map(aman, center_on='mars', res=res, cuts=cuts, data_splits=splits, source_flags=source_flags, comps="T")
#
# for split in out["splits"].keys():
#     dat = out["splits"][split]
#     title = split.lower().replace(" ", "_")
#     plt.close()
#     plt.title(split)
#     plt.imshow(dat["solved"][0])
#     plt.xlim((cent[1]-50, cent[1]+50))
#     plt.ylim((cent[0]-50, cent[0]+50))
#     plt.colorbar()
#     plt.grid()
#     plt.savefig(os.path.join(plot_dir, f"map_{title}.png"))
#     
#     plt.close()
#     plt.title(split)
#     plt.imshow(dat["solved"][0], norm=SymLogNorm(1e-6))
#     plt.xlim((cent[1]-50, cent[1]+50))
#     plt.ylim((cent[0]-50, cent[0]+50))
#     plt.colorbar()
#     plt.grid()
#     plt.savefig(os.path.join(plot_dir, f"map_{title}_log10.png"))
