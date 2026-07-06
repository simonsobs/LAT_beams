import os
import sys

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from pixell import enmap, reproject
from scipy.ndimage import rotate
from sotodlib.core import Context
from sotodlib.io import hkdb
from tqdm import tqdm
from scipy.optimize import minimize

from lat_beams import beam_utils as bu
from lat_beams.utils import get_args_cfg, make_jobdb, setup_cfg, setup_paths
from lat_beams.fitting.models import bessel_beam_from_aman

# Get settings
args, cfg_dict = get_args_cfg()
cfg, cfg_str = setup_cfg(args, cfg_dict, {"map_mask_size": "mask_size"})
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
plot_dir = os.path.join(plot_dir, "cross_template")
os.makedirs(plot_dir, exist_ok=True)
fpath = os.path.join(data_dir, "beam_pars_no_cross.h5")
jdb = make_jobdb(None, data_dir, "_c")

# Get jobs
fjobs = np.array(jdb.get_jobs(jclass="fit_map", jstate="done"))
mjobdict = {
    f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['band']}": job
    for job in jdb.get_jobs(jclass="beam_map", jstate="done")
}
fjobs = np.array([job for job in fjobs if job.tags.get("split", "") == ""])

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

# Load stack residuals by band
templates = {}
stack_dir = os.path.join(data_dir, "stacks_c", "band")
for epoch in cfg.epochs:
    for band in np.unique(all_fits["band"]):
        map_path = os.path.join(stack_dir, band, f"{band}_{epoch[0]}_{epoch[1]}_stack_resid.fits")
        templates[band] = enmap.read_map(map_path)
    break

def objective(x, imap, ivar, template):
    amp, theta = x
    model = amp * rotate(template, theta, reshape=False)
    return np.sum(ivar * (imap - model)**2)

# Load, recenter, and normalize each map before fitting
ext_rad = np.deg2rad(cfg.mask_size)
cross_amp = np.zeros(len(all_fits)) + np.nan
cross_ang = np.zeros(len(all_fits)) + np.nan
idx = np.arange(len(all_fits))
for epoch in cfg.epochs:
    times = all_fits["time"]
    tmsk = (times >= epoch[0]) * (times < epoch[1])
    for i, fit, fjob in tqdm(zip(idx[tmsk], all_fits[tmsk], fjobs[tmsk]), total=np.sum(tmsk)):
        jobstr = f"{fjob.tags['obs_id']}-{fjob.tags['wafer_slot']}-{fjob.tags['stream_id']}-{fjob.tags['band']}"
        mjob = mjobdict[jobstr]
        map_path = os.path.join(data_dir, mjob.tags["solved"])
        ivar_path = os.path.join(data_dir, mjob.tags["weights"])
        imap = enmap.read_map(map_path)[0]
        ivar = enmap.read_map(ivar_path)[0][0]
        tmap = templates[fit["band"]]
        cent = np.array(
            (
                fit["aman"].gauss.eta0.to(u.rad).value,
                fit["aman"].gauss.xi0.to(u.rad).value,
            )
        )
        imap = (
            reproject.thumbnails(
                imap - fit["aman"].gauss.off.value,
                r=ext_rad,
                coords=cent,
                oshape=tmap.shape,
                owcs=tmap.wcs,
            )
            / fit["aman"].gauss.amp.value
        )
        ivar = (
            reproject.thumbnails_ivar(
                ivar,
                r=ext_rad,
                coords=cent,
                oshape=tmap.shape,
                owcs=tmap.wcs,
            )
            * fit["aman"].gauss.amp.value**2
        )
        if tmap.shape != imap.shape:
            print(map_path)
            continue
        fscale_fac = 90.0 / float(fit["band"][1:])
        band_mask_size = np.deg2rad(fscale_fac * cfg.mask_size)
        posmap = ivar.posmap()
        r = np.sqrt(posmap[0]**2 + posmap[1]**2)
        ivar[r > band_mask_size] = 0
        bessel = (bessel_beam_from_aman(posmap, fit["aman"]) - fit["aman"].gauss.off.value)/fit["aman"].gauss.amp.value
        res = minimize(objective, (0, 0), args=(imap - bessel, ivar, tmap), bounds=[(0, 100), (0, 360)], method="Powell")
        cross_amp[i], cross_ang[i] = res.x

# Kill obvious outliers
msk = np.isfinite(cross_amp)
msk *= (cross_amp < np.percentile(cross_amp[msk], 95)) * (
    cross_amp > np.percentile(cross_amp[msk], 5)
)
cross_amp = cross_amp[msk]
cross_ang = cross_ang[msk]
all_fits = all_fits[msk]
fjobs = fjobs[msk]
cross_dat = {"Amplitude": cross_amp, "Angle": cross_ang}

print(f"{len(all_fits)} good fits to plot")
if len(fjobs) == 0:
    sys.exit(0)

# Grab the air temp for each obs
air_temp = False
temps = []
if air_temp:
    times = bu.get_split_vec(all_fits, "start_time+stop_time", ctx)
    starts, stops = np.array(np.char.split(times, "+").tolist()).astype(float).T
    temps = np.zeros(len(all_fits))
    hcfg = hkdb.HkConfig.from_yaml(
        "/global/cfs/cdirs/sobs/users/mhasse/work/250404/hkdb-site.cfg"
    )
    hdb = hkdb.HkDb(hcfg)
    print("Loading air temps")
    for i, (t0, t1) in enumerate(tqdm(zip(starts, stops), total=len(temps))):
        lspec = hkdb.LoadSpec(
            cfg=hcfg,
            start=t0,
            end=t1,
            fields=["env-vantage.weather_data.temp_outside"],
            downsample_factor=10,
            hkdb=hdb,
        )
        result = hkdb.load_hk(lspec)
        if "env-vantage.weather_data.temp_outside" in result.data:
            temps[i] = np.nanmean(result.data["env-vantage.weather_data.temp_outside"][1])
        else:
            temps[i] = np.nan

# Loop through splits
epoch_lines = np.unique(cfg.epochs)
epoch_lines = epoch_lines[
    (epoch_lines >= np.min(all_fits["time"])) * (epoch_lines < np.max(all_fits["time"]))
]
for split in cfg.split_by:
    print(f"Splitting by {split}")
    for name, cross_d in cross_dat.items():
        split_vec = bu.get_split_vec(all_fits, split, ctx)
        spls = np.unique(split_vec)
        plot_dir_spl = os.path.join(plot_dir, split, name)
        os.makedirs(plot_dir_spl, exist_ok=True)

        # Now lets make some plots
        # First ones with all epochs together
        # Colored scatter plot of everything
        plt.scatter(all_fits["time"], split_vec, c=cross_d, alpha=0.4)
        plt.colorbar(label=name, ax=plt.gca())
        for e in epoch_lines:
            plt.axvline(e)
        plt.title(f"Octopole {name} by Time")
        plt.xlabel("Time (s)")
        plt.savefig(os.path.join(plot_dir_spl, "time_scatter.png"), bbox_inches="tight")
        plt.close()

        # Single split scatter plots
        for spl in spls:
            smsk = split_vec == spl
            plt.scatter(all_fits["time"][smsk], cross_d[smsk], alpha=0.4)
            for e in epoch_lines:
                plt.axvline(e)
            plt.title(f"{spl} Octopole {name} by Time")
            plt.xlabel("Time (s)")
            plt.ylabel(f"Octopole {name}")
            plt.savefig(
                os.path.join(plot_dir_spl, f"time_scatter_{spl}.png"), bbox_inches="tight"
            )
            plt.close()

        # Now plot things within each epoch
        for start, end in cfg.epochs:
            tmsk = (all_fits["time"] >= start) * (all_fits["time"] < end)
            if np.sum(tmsk) == 0:
                continue
            tfits, tamps, tvec = all_fits[tmsk], cross_d[tmsk], split_vec[tmsk]

            # Colored scatter vs time of day
            plt.scatter(tfits["hour"], tvec, c=tamps, alpha=0.4)
            plt.colorbar(label=f"{name}", ax=plt.gca())
            plt.title(f"Octopole {name} by Hour ({start}, {end})")
            plt.xlabel("Hour of Day (hr)")
            plt.savefig(
                os.path.join(plot_dir_spl, f"hour_scatter_{start}_{end}.png"),
                bbox_inches="tight",
            )
            plt.close()

            enc = bu.get_split_vec(tfits, "az_center+el_center+roll_center", ctx)
            az, el, roll = np.array(np.char.split(enc, "+").tolist()).astype(float).T
            amp = bu.get_fit_vec(tfits, "gauss.amp")
            fwhm = bu.get_fit_vec(tfits, "data_fwhm")
            to_scatter = [
                ("hour", "Hour of Day (hr)", tfits["hour"]),
                ("az", "Azimuth (deg)", az),
                ("el", "Elevation (deg)", el),
                ("roll", "Roll (deg)", roll),
                ("corot", "Corotation (deg)", el - 60 - roll),
                ("amp", "Amp (pW)", amp),
                ("FWHM", 'FWHM (")', fwhm),
                (
                    "std(PWV)",
                    "std(PWV) (mm)",
                    np.array(bu.get_split_vec(tfits, "pwv_std", ctx), float),
                ),
                (
                    "PWV",
                    "PWV (mm)",
                    np.array(bu.get_split_vec(tfits, "pwv_mean", ctx), float),
                ),
            ]
            if air_temp:
                to_scatter += [("temperature", "Temperature (C)", temps[tmsk])]

            # Individual splits
            for spl in np.unique(tvec):
                smsk = tvec == spl

                # Scatter vs interesting things
                for sname, xax, dat in to_scatter:
                    plt.scatter(dat[smsk], tamps[smsk], alpha=0.4)
                    plt.title(
                        f"{spl} Octopole {name} {sname.title()} ({start}, {end})"
                    )
                    plt.xlabel(xax)
                    plt.ylabel(f"Octopole {name}")
                    plt.savefig(
                        os.path.join(
                            plot_dir_spl, f"{sname.lower()}_scatter_{spl}_{start}_{end}.png"
                        ),
                        bbox_inches="tight",
                    )
                    plt.close()

                # Simple histogram
                plt.hist(tamps[smsk], bins="auto")
                plt.title(f"{spl} Octopole {name} ({start}, {end})")
                plt.ylabel(f"Octopole {name}")
                plt.savefig(
                    os.path.join(plot_dir_spl, f"hist_{spl}_{start}_{end}.png"),
                bbox_inches="tight",
                )
                plt.close()
