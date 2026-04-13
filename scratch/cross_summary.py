import argparse
import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sotodlib.core import Context

from lat_beams import beam_utils as bu
from lat_beams.utils import get_args_cfg, make_jobdb, setup_cfg, setup_paths

# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg( args, cfg_dict, { },)
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
jdb = make_jobdb(None, data_dir)

# Get jobs
fjobs = np.array(jdb.get_jobs(jclass="fit_map", jstate="done"))

print(f"{len(fjobs)} fits to check")
if len(fjobs) == 0:
    sys.exit(0)

# Load fits
all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs)
snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
solid_angle = bu.get_fit_vec(all_fits, "data_solid_angle_corr")
msk = snr > 100
msk *= solid_angle > 0
all_fits = all_fits[msk]

# Get octopole amplitude
amps = np.abs(bu.get_fit_vec(all_fits, "bessel.amps"))
cross_amp = amps[:, :, :, 4, :].reshape(len(amps), -1).sum(axis=-1).value 
tot_amp = amps.reshape(len(amps), -1).sum(axis=-1).value
cross_amp /= tot_amp
# Kill obvious outliers
msk = (cross_amp < np.percentile(cross_amp, 99)) * (cross_amp > np.percentile(cross_amp, 1))
cross_amp = cross_amp[msk]
all_fits = all_fits[msk]
print(f"{len(all_fits)} good fits to plot")
if len(fjobs) == 0:
    sys.exit(0)

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
    plt.scatter(all_fits["time"], split_vec, c=cross_amp, alpha=.4)
    plt.colorbar(label="Amplitude", ax=plt.gca())
    for e in epoch_lines:
        plt.axvline(e)
    plt.title(f"Octopole Amplitude by Time")
    plt.xlabel("Time (s)")
    plt.savefig(os.path.join(plot_dir_spl, "time_scatter.png"))
    plt.close()

    # Single split scatter plots
    for spl in spls:
        smsk = split_vec == spl
        plt.scatter(all_fits["time"][smsk], cross_amp[smsk], alpha=.4)
        for e in epoch_lines:
            plt.axvline(e)
        plt.title(f"Octopole Amplitude by Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Octopole Amplitude")
        plt.savefig(os.path.join(plot_dir_spl, f"time_scatter_{spl}.png"))
        plt.close()

    # Now plot things within each epoch
    for (start, end) in cfg.epochs:
        tmsk = (all_fits["time"] >= start) * (all_fits["time"] < end)
        tfits, tamps, tvec = all_fits[tmsk], cross_amp[tmsk], split_vec[tmsk]

        # Colored scatter vs time of day
        plt.scatter(tfits["hour"], tvec, c=tamps, alpha=.4)
        plt.colorbar(label="Amplitude", ax=plt.gca())
        plt.title(f"Octopole Amplitude by Hour ({start}, {end})")
        plt.xlabel("Hour of Day (hr)")
        plt.savefig(os.path.join(plot_dir_spl, f"hour_scatter_{start}_{end}.png"))
        plt.close()

        enc = bu.get_split_vec(tfits, "az_center+el_center+roll_center", ctx)
        az, el, roll = np.array(np.char.split(enc, "+").tolist()).astype(float).T
        to_scatter = [("hour", "Hour of Day (hr)", tfits["hour"]),
                      ("az", "Azimuth (deg)", az), 
                      ("el", "Elevation (deg)", el), 
                      ("roll", "Roll (deg)", roll),
                      ("corot", "Corotation (deg)", el - 60 - roll),
                      ("pwv", "PWV (mm)", bu.get_split_vec(tfits, "pwv_mean", ctx))] 


        # Individual splits
        for spl in np.unique(tvec):
            smsk = tvec == spl

            # Scatter vs interesting things
            for name, xax, dat in to_scatter:
                plt.scatter(dat[smsk], tamps[smsk], alpha=.4)
                plt.title(f"{spl} Octopole Amplitude by {name.title()} ({start}, {end})")
                plt.xlabel(xax)
                plt.ylabel("Octopole Amplitude")
                plt.savefig(os.path.join(plot_dir_spl, f"{name}_scatter_{spl}_{start}_{end}.png"))
                plt.close()

            # Simple histogram
            plt.hist(tamps[smsk], bins="auto")
            plt.title(f"{spl} Octopole Amplitude ({start}, {end})")
            plt.ylabel("Octopole Amplitude")
            plt.savefig(os.path.join(plot_dir_spl, f"hist_{spl}_{start}_{end}.png"))
            plt.close()
