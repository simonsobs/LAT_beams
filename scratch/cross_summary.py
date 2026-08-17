import os
import sys
import time as time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soma.harmonic as sh
from pixell import enmap
from sotodlib.core import Context
from sotodlib.io import hkdb
from tqdm import tqdm

import lat_beams.fitting.map.bessel as fb
from lat_beams import beam_utils as bu
from lat_beams.plotting import auto_relplot
from lat_beams.utils import get_args_cfg, make_jobdb, setup_cfg, setup_paths

palette = sns.color_palette("colorblind", 13)
sns.set_palette(palette)

mode = "model"
print(f"Running in {mode} mode")

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
plot_dir = os.path.join(plot_dir, "cross_summary")
os.makedirs(plot_dir, exist_ok=True)
fpath = os.path.join(data_dir, "beam_pars.h5")
cfpath = os.path.join(data_dir, "beam_pars_no_cross.h5")
jdb = make_jobdb(None, data_dir)
cjdb = make_jobdb(None, data_dir, "_c")

# Get jobs
fjobs = np.array(jdb.get_jobs(jclass="fit_map", jstate="done"))
mjobdict = {
    f"{job.tags['obs_id']}-{job.tags['wafer_slot']}-{job.tags['stream_id']}-{job.tags['array']}-{job.tags['band']}": job
    for job in jdb.get_jobs(jclass="beam_map", jstate="done")
}

print(f"{len(fjobs)} fits to check")
if len(fjobs) == 0:
    sys.exit(0)

# Load fits
all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs.tolist())
snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
solid_angle = bu.get_fit_vec(all_fits, "bessel.model_solid_angle_true")
msk = snr > 100
msk *= solid_angle > 0
msk *= all_fits["split"] == "full"
all_fits = all_fits[msk]
fjobs = fjobs[msk]

print(f"{len(all_fits)} good fits to plot (ignoring splits)")
if len(fjobs) == 0:
    sys.exit(0)

# Get multipole stats
t0 = time.time()
pix_extent = int(2 * (np.deg2rad(cfg.mask_size) // cfg.res))
twcs = enmap.wcsutils.build(
    [0, 0],
    res=np.rad2deg(cfg.res),
    shape=(pix_extent, pix_extent),
    system="tan",
    rowmajor=True,
)
tmap = enmap.zeros((3, pix_extent, pix_extent), twcs)
posmap = tmap.posmap()

mamps_list = []
mangs_list = []
for i, fit in enumerate(tqdm(all_fits)):
    aman = fit["aman"]
    if mode == "model":
        imap = fb.bessel_beam_from_aman(posmap, aman)
        imap -= aman.bessel.off.value
    elif mode == "data":
        fjob = fjobs[i]
        fjobstr = f"{fjob.tags['obs_id']}-{fjob.tags['wafer_slot']}-{fjob.tags['stream_id']}-{fjob.tags['array']}-{fjob.tags['band']}"
        if fjobstr not in mjobdict:
            raise ValueError("Map job not found for %s", fjobstr)
        mjob = mjobdict[fjobstr]
        map_path = os.path.join(
            data_dir, mjob.tags["solved"].format(split=fjob.tags["split"])
        )
        imap = cast(enmap.ndmap, enmap.read_map(map_path)[0])
    else:
        raise ValueError("Invalid mode %s", mode)

    ypix, xpix = enmap.sky2pix(
        imap.shape, imap.wcs, ([[aman.eta0.value], [aman.xi0.value]])
    )
    y0, x0 = float(ypix[0]), float(xpix[0])
    modes = sh.azimuthal_modes(imap, mmax=4, center=(y0, x0))
    a_m = modes["a_m"]
    ell = modes["ell"]
    rho, frac = sh.mode_metrics(a_m)
    frac_integrated = sh.integrated_multipole_fractions(frac, ell, a_m)
    mamps_list.append(frac_integrated)
    fit_angles = sh.integrated_multipole_angles(a_m, ell, True)
    mangs_list.append(fit_angles)
mamps = 100 * np.array(mamps_list)
mangs = np.array(mangs_list)

# Kill obvious outliers
msk = (mamps >= 0) * (mamps <= 1)
mmsk = np.any(msk, axis=-1)
pl = np.percentile(mamps[mmsk], 10, axis=0)
ph = np.percentile(mamps[mmsk], 90, axis=0)
msk *= (mamps > pl) * (mamps < ph)
msk = np.any(msk, axis=-1)
all_fits = all_fits[msk]
fjobs = fjobs[msk]
mamps = mamps[msk]
mangs = mangs[msk]

print(f"{len(all_fits)} good fits to plot after computing decomp")
if len(fjobs) == 0:
    sys.exit(0)

# Grab the air temp for each obs
if False:
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
            temps[i] = np.nanmean(
                result.data["env-vantage.weather_data.temp_outside"][1]
            )
        else:
            temps[i] = np.nan

# Build the dataset
dset = {f"m={m} Amplitude (%)": mamps[:, m] for m in range(1, 5)}
dset.update({f"m={m} Angle (deg)": mangs[:, m] for m in range(1, 5)})
dset["Hour of Day (hr)"] = all_fits["hour"]
dset["ctime (s)"] = all_fits["time"]
enc = bu.get_split_vec(all_fits, "az_center+el_center+roll_center", ctx)
az, el, roll = np.array(np.char.split(enc, "+").tolist()).astype(float).T
dset["Azimuth (deg)"] = az
dset["Elevation (deg)"] = el
dset["Roll (deg)"] = roll
dset["Corot (deg)"] = el - 60 - roll
dset["Amp (pW)"] = bu.get_fit_vec(all_fits, "gauss.amp").value
dset['FWHM (")'] = bu.get_fit_vec(all_fits, "data_fwhm").value
dset["PWV (mm)"] = np.array(bu.get_split_vec(all_fits, "pwv_mean", ctx), float)
epoch_starts = np.zeros(len(all_fits), int)
epoch_ends = np.zeros(len(all_fits), int)
for epoch_start, epoch_end in cfg.epochs:
    tmsk = (all_fits["time"] >= epoch_start) * (all_fits["time"] < epoch_end)
    epoch_starts[tmsk] = epoch_start
    epoch_ends[tmsk] = epoch_end
dset["Epoch"] = np.array([f"{s}-{e}" for s, e in zip(epoch_starts, epoch_ends)])
to_scatter = [
    ("ctime (s)", "time"),
    ("Hour of Day (hr)", "hour"),
    ("Amp (pW)", "amp"),
    ('FWHM (")', "fwhm"),
    ("PWV (mm)", "pwv"),
    ("Azimuth (deg)", "az"),
    ("Elevation (deg)", "el"),
    ("Roll (deg)", "roll"),
    ("Corot (deg)", "corot"),
]
emsk = dset["Epoch"] != "0-0"
dset = {k: v[emsk] for k, v in dset.items()}
all_fits = all_fits[emsk]
fjobs = fjobs[emsk]

# Loop through splits and epochs
epoch_lines = np.unique(cfg.epochs)
epoch_lines = epoch_lines[
    (epoch_lines >= np.min(all_fits["time"])) * (epoch_lines < np.max(all_fits["time"]))
]
for split in cfg.split_by:
    print(f"Splitting by {split}")
    split_vec = bu.get_split_vec(all_fits, split, ctx, metasplits=cfg.metasplits)
    msk = np.char.find(split_vec.astype(str), "NOMATCH") == -1
    for m in range(1, 5):
        dat = dset[f"m={m} Amplitude (%)"]
        for spl in np.unique(split_vec[msk]):
            smsk = (split_vec == spl) * msk
            for epc in np.unique(dset["Epoch"][smsk]):
                emsk = smsk * (dset["Epoch"] == epc)

                ds = dat[emsk]
                median = np.median(ds)
                mad = np.median(np.abs(ds - median))
                lower = median - 9.0 * mad
                upper = median + 9.0 * mad
                msk[emsk] *= (ds >= lower) & (ds <= upper)

                emsk *= msk

                ds = dat[emsk]
                Q1 = np.percentile(ds, 25)
                Q3 = np.percentile(ds, 75)
                IQR = Q3 - Q1
                msk[emsk] *= (ds >= Q1 - (1.5 * IQR)) * (ds <= Q3 + (1.5 * IQR))
    if np.sum(msk) == 0:
        print("None found!")
        continue
    spls = np.array(np.char.split(split_vec[msk], "+").tolist()).T
    srt = np.lexsort(spls.tolist() + [dset["Epoch"][msk]])
    dset_filt = {k: v[msk][srt] for k, v in dset.items()}
    for i, s in enumerate(split.split("+")):
        dset_filt[s] = spls[i][srt]

    # Now lets make some plots
    for field, name in to_scatter:
        plot_dir_spl = os.path.join(plot_dir, split, name)
        os.makedirs(plot_dir_spl, exist_ok=True)
        for m in range(1, 5):
            to_ignore = [f for f, _ in to_scatter if f != field]
            to_merge = [[s for s in split.split("+") if s != "tube_slot"]]
            row = "+".join(to_merge[0])
            if field not in dset_filt:
                print(f"{field} missing?")
            plot = auto_relplot(
                dset_filt,
                x=field,
                y=f"m={m} Amplitude (%)",
                kind="scatter",
                col="Epoch",
                row=row,
                hue="tube_slot" if "tube_slot" in dset_filt.keys() else None,
                style="tube_slot" if "tube_slot" in dset_filt.keys() else None,
                ignore=to_ignore,
                merge=to_merge,
                auto=False,
                facet_kws={
                    "sharey": False,
                    "sharex": name != "time",
                    "margin_titles": True,
                },
                alpha=0.8,
                s=10,
            )
            for axis in plot.axes.flat:
                axis.tick_params(labelleft=True)
            plot.figure.suptitle(
                f"Fractional m={m} Amplitude by {name.upper() if field.split(' ')[0].isupper() else name.title()} ({mode})"
            )
            plt.subplots_adjust(top=(1 - 0.25 / len(plot.axes)))
            plt.savefig(
                os.path.join(plot_dir_spl, f"{mode}_{name}_amp_{m}.png"),
                bbox_inches="tight",
            )
            plt.close()

            plot = auto_relplot(
                dset_filt,
                x=field,
                y=f"m={m} Angle (deg)",
                kind="scatter",
                col="Epoch",
                row=row,
                hue="tube_slot" if "tube_slot" in dset_filt.keys() else None,
                style="tube_slot" if "tube_slot" in dset_filt.keys() else None,
                ignore=to_ignore,
                merge=to_merge,
                auto=False,
                facet_kws={
                    "sharey": False,
                    "sharex": name != "time",
                    "margin_titles": True,
                },
                alpha=0.5,
            )
            for axis in plot.axes.flat:
                axis.tick_params(labelleft=True)
            plot.figure.suptitle(
                f"Fractional m={m} Angle by {name.upper() if field.split(' ')[0].isupper() else name.title()} ({mode})"
            )
            plt.subplots_adjust(top=(1 - 0.25 / len(plot.axes)))
            plt.savefig(
                os.path.join(plot_dir_spl, f"{mode}_{name}_ang_{m}.png"),
                bbox_inches="tight",
            )
            plt.close()
