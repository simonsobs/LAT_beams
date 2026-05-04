import sys
from astropy import constants as const
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from sotodlib.core import Context
from pixell import enmap
import astropy.units as u
from sotodlib.io import hkdb

from lat_beams import beam_utils as bu
from lat_beams.utils import get_args_cfg, make_jobdb, setup_cfg, setup_paths

# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg( args, cfg_dict, { "map_mask_size": "mask_size" } )
ctx = Context(cfg.ctx_path)
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")

# Setup folders
plot_dir, data_dir = setup_paths(
    cfg.root_dir,
    "beams",
    cfg.tel,
    f"{cfg.pointing_type}{(cfg.append!="")*'_'}{cfg.append}",
)
plot_dir = os.path.join(plot_dir, "cross_summary")
os.makedirs(plot_dir, exist_ok=True)
fpath = os.path.join(data_dir, "beam_pars.h5")
cfpath = os.path.join(data_dir, "beam_pars_no_cross.h5")
jdb = make_jobdb(None, data_dir)
cjdb = make_jobdb(None, data_dir, "_c")

# Get jobs
fjobs = np.array(jdb.get_jobs(jclass="fit_map", jstate="done"))
cfjobs = np.array(cjdb.get_jobs(jclass="fit_map", jstate="done"))

print(f"{len(fjobs)} fits to check")
if len(fjobs) == 0:
    sys.exit(0)

# Load fits
all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs.tolist())
snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
solid_angle = bu.get_fit_vec(all_fits, "bessel.model_solid_angle_true")
msk = snr > 100
msk *= solid_angle > 0
all_fits = all_fits[msk]
fjobs = fjobs[msk]

all_cfits = bu.load_beam_fits_from_jobs(cfpath, cfjobs.tolist())
snr = bu.get_fit_vec(all_cfits, "amp") / bu.get_fit_vec(all_cfits, "noise")
solid_angle = bu.get_fit_vec(all_cfits, "bessel.model_solid_angle_true")
msk = snr > 100
msk *= solid_angle > 0
all_cfits = all_cfits[msk]
cfjobs = cfjobs[msk]

# Put them in the same order
vec = bu.get_split_vec(all_fits, "obs_id+stream_id+band", ctx)
cvec = bu.get_split_vec(all_cfits, "obs_id+stream_id+band", ctx)
overlap, idx, cidx = np.intersect1d(vec, cvec, return_indices=True)
all_fits = all_fits[idx]
fjobs = fjobs[idx]
all_cfits = all_cfits[idx]
cfjobs = cfjobs[cidx]

# Get octopole solid angle
sastr = "bessel.model_solid_angle_true"
fsolid_angle = bu.get_fit_vec(all_fits, sastr)
csolid_angle = bu.get_fit_vec(all_cfits, sastr)
cross_sa = np.abs(fsolid_angle - csolid_angle)/fsolid_angle

# Kill obvious outliers
msk = (cross_sa >= 0) * (cross_sa <= 1)
msk *= (cross_sa < np.percentile(cross_sa[msk], 90)) * (cross_sa> np.percentile(cross_sa[msk], 1)) 
cross_sa = cross_sa[msk]
all_fits = all_fits[msk]
fjobs = fjobs[msk]
all_cfits = all_cfits[msk]
cfjobs = cfjobs[msk]

print(f"{len(all_fits)} good fits to plot")
if len(fjobs) == 0:
    sys.exit(0)

# Grab the air temp for each obs
times = bu.get_split_vec(all_fits, "start_time+stop_time", ctx)
starts, stops = np.array(np.char.split(times, "+").tolist()).astype(float).T
temps = np.zeros(len(all_fits))
hcfg = hkdb.HkConfig.from_yaml("/global/cfs/cdirs/sobs/users/mhasse/work/250404/hkdb-site.cfg")
hdb = hkdb.HkDb(hcfg)
print("Loading air temps")
for i, (t0, t1) in enumerate(tqdm(zip(starts, stops), total=len(temps))):
    lspec = hkdb.LoadSpec(cfg=hcfg, start=t0, end=t1, fields=['env-vantage.weather_data.temp_outside'], downsample_factor=10, hkdb=hdb)
    result = hkdb.load_hk(lspec)
    if 'env-vantage.weather_data.temp_outside' in result.data:
        temps[i] = np.nanmean(result.data['env-vantage.weather_data.temp_outside'][1])
    else:
        temps[i] = np.nan

# Loop through splits
epoch_lines = np.unique(cfg.epochs)
epoch_lines = epoch_lines[
    (epoch_lines >= np.min(all_fits["time"])) * (epoch_lines < np.max(all_fits["time"]))
]
for split in cfg.split_by:
    print(f"Splitting by {split}")
    split_vec = bu.get_split_vec(all_fits, split, ctx)
    spls = np.unique(split_vec)
    plot_dir_spl = os.path.join(plot_dir, split)
    os.makedirs(plot_dir_spl, exist_ok=True)

    # Now lets make some plots
    # First ones with all epochs together
    # Colored scatter plot of everything
    plt.scatter(all_fits["time"], split_vec, c=cross_sa, alpha=.4)
    plt.colorbar(label="Amplitude", ax=plt.gca())
    for e in epoch_lines:
        plt.axvline(e)
    plt.title(f"Fractional Octopole Solid Angle by Time")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(plot_dir_spl, "time_scatter.png"), bbox_inches = "tight")
    plt.close()

    # Single split scatter plots
    for spl in spls:
        smsk = split_vec == spl
        plt.scatter(all_fits["time"][smsk], cross_sa[smsk], alpha=.4)
        for e in epoch_lines:
            plt.axvline(e)
        plt.title(f"{spl} Fractional Octopole Solid Angle by Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Fractional Octopole Solid Angle")
        plt.savefig(os.path.join(plot_dir_spl, f"time_scatter_{spl}.png"), bbox_inches = "tight")
        plt.close()

    # Now plot things within each epoch
    for (start, end) in cfg.epochs:
        tmsk = (all_fits["time"] >= start) * (all_fits["time"] < end)
        if np.sum(tmsk) == 0:
            continue
        tfits, tamps, tvec = all_fits[tmsk], cross_sa[tmsk], split_vec[tmsk]

        # Colored scatter vs time of day
        plt.scatter(tfits["hour"], tvec, c=tamps, alpha=.4)
        plt.colorbar(label="Amplitude", ax=plt.gca())
        plt.title(f"Fractional Octopole Solid Angle by Hour ({start}, {end})")
        plt.xlabel("Hour of Day (hr)")
        plt.savefig(os.path.join(plot_dir_spl, f"hour_scatter_{start}_{end}.png"), bbox_inches = "tight")
        plt.close()

        enc = bu.get_split_vec(tfits, "az_center+el_center+roll_center", ctx)
        az, el, roll = np.array(np.char.split(enc, "+").tolist()).astype(float).T
        amp = bu.get_fit_vec(tfits, "gauss.amp")
        fwhm = bu.get_fit_vec(tfits, "data_fwhm")
        to_scatter = [("hour", "Hour of Day (hr)", tfits["hour"]),
                      ("az", "Azimuth (deg)", az), 
                      ("el", "Elevation (deg)", el), 
                      ("roll", "Roll (deg)", roll),
                      ("corot", "Corotation (deg)", el - 60 - roll),
                      ("amp", "Amp (pW)", amp),
                      ("FWHM", 'FWHM (")', fwhm),
                      ("temperature", "Temperature (C)", temps[tmsk]),
                      ("std(PWV)", "std(PWV) (mm)", np.array(bu.get_split_vec(tfits, "pwv_std", ctx), float)), 
                      ("PWV", "PWV (mm)", np.array(bu.get_split_vec(tfits, "pwv_mean", ctx), float))] 


        # Individual splits
        for spl in np.unique(tvec):
            smsk = tvec == spl

            # Scatter vs interesting things
            for name, xax, dat in to_scatter:
                plt.scatter(dat[smsk], tamps[smsk], alpha=.4)
                plt.title(f"{spl} Fractional Octopole Solid Angle by {name.title()} ({start}, {end})")
                plt.xlabel(xax)
                plt.ylabel("Fractional Octopole Solid Angle")
                plt.savefig(os.path.join(plot_dir_spl, f"{name.lower()}_scatter_{spl}_{start}_{end}.png"), bbox_inches="tight")
                plt.close()

            # Simple histogram
            plt.hist(tamps[smsk], bins="auto")
            plt.title(f"{spl} Fractional Octopole Solid Angle ({start}, {end})")
            plt.ylabel("Fractional Octopole Solid Angle")
            plt.savefig(os.path.join(plot_dir_spl, f"hist_{spl}_{start}_{end}.png"), bbox_inches = "tight")
            plt.close()
