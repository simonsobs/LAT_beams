import os
import sys

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy import constants as const
from pixell import enmap
from sotodlib.core import Context
from sotodlib.io import hkdb
from tqdm import tqdm

from lat_beams import beam_utils as bu
from lat_beams.utils import get_args_cfg, make_jobdb, setup_cfg, setup_paths

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
plot_dir = os.path.join(plot_dir, "beam_summary")
os.makedirs(plot_dir, exist_ok=True)
fpath = os.path.join(data_dir, "beam_pars.h5")
jdb = make_jobdb(None, data_dir)

# Get jobs
fjobs = np.array(jdb.get_jobs(jclass="fit_map", jstate="done"))

print(f"{len(fjobs)} fits to check")
if len(fjobs) == 0:
    sys.exit(0)

# Load fits
all_fits = bu.load_beam_fits_from_jobs(fpath, fjobs.tolist(), jdb)
snr = bu.get_fit_vec(all_fits, "amp") / bu.get_fit_vec(all_fits, "noise")
solid_angle = bu.get_fit_vec(all_fits, "bessel.model_solid_angle_true")
msk = snr > 1
msk *= solid_angle > 0
all_fits = all_fits[msk]
fjobs = fjobs[msk]

print(f"{len(all_fits)} good fits to plot")
if len(fjobs) == 0:
    sys.exit(0)

# Setup list of things to plot
to_plot = [
    ("Bessel Solid Angle (str)", "bessel.model_solid_angle_true", u.steradian),
    ("Gassian Solid Angle Corrected (str)", "gauss.data_solid_angle_corr", u.steradian),
    ('FWHM (")', "data_fwhm", u.arcsec),
]


# Construct the dataset for seaborn
dset = {
    name: bu.get_fit_vec(all_fits, vec).to(unit).value for name, vec, unit in to_plot
}
dset["Time of Day (hr)"] = all_fits["hour"]
dset["Time (ctime)"] = all_fits["time"]
dset["band"] = all_fits["band"]

# Add epoch info
epoch_strs = ["unknown"]
epoch_ids = np.zeros(len(all_fits), int)
for i, (e0, e1) in enumerate(cfg.epochs):
    epoch_strs += [f"{e0}_{e1}"]
    epoch_ids[(all_fits["time"] >= e0) * (all_fits["time"] < e1)] = i + 1
dset["Epoch"] = np.array(epoch_strs)[epoch_ids]

# Loop through splits
for split in cfg.split_by:
    print(f"Splitting by {split}")
    split_vec = bu.get_split_vec(all_fits, split, ctx, metasplits=cfg.metasplits)
    spls = np.unique(split_vec)
    plot_dir_spl = os.path.join(plot_dir, split)
    os.makedirs(plot_dir_spl, exist_ok=True)
    dset[split] = split_vec

    for name, vec, unit in to_plot:
        title = f"{' '.join(name.split(' ')[:-1])}"
        prefix = title.lower().replace(" ", "_")

        # Filter out outliers by split
        dat = dset[name]
        msk = dset["Epoch"] != "unknown"
        msk *= ~np.isin(dset["band"], ["f030", "f040"])
        for spl in np.unique(split_vec):
            smsk = split_vec == spl
            ds = dat[smsk]
            Q1 = np.percentile(ds, 25)
            Q3 = np.percentile(ds, 75)
            IQR = Q3 - Q1
            msk[smsk] *= (ds > Q1 - 1.5 * IQR) * (ds < Q3 + 1.5 * IQR)
        dset_filt = {key: val[msk] for key, val in dset.items()}

        # Violin with epoch hue
        sns.violinplot(
            data=dset_filt,
            x=split,
            y=name,
            hue="Epoch",
            split=True,
            inner="quart",
            cut=0,
            order=np.sort(np.unique(dset_filt[split]))
        )
        plt.title(f"{' '.join(name.split(' ')[:-1])} Distrubution")
        # Add nominal FWHM if we are plotting FWHM
        if "FWHM" in name:
            for b in np.unique(dset_filt["band"]):
                plt.axhline(cfg.nominal_fwhm[b] * 60.0, linestyle="--")

        plt.savefig(os.path.join(plot_dir_spl, f"{prefix}_dist.png"))
        plt.close()

        # Scatter vs ctime with epoch cols
        sns.relplot(
            data=dset_filt,
            x="Time (ctime)",
            y=name,
            hue=split,
            col="Epoch",
            kind="scatter",
            facet_kws={"sharey": "row", "sharex": False},
        )
        plt.suptitle(f"{' '.join(name.split(' ')[:-1])} Over Time")
        plt.subplots_adjust(top=0.85)
        plt.savefig(os.path.join(plot_dir_spl, f"{prefix}_time.png"))
        plt.close()

        # Scatter vs time of day with epoch cols
        sns.relplot(
            data=dset_filt,
            x="Time of Day (hr)",
            y=name,
            hue=split,
            col="Epoch",
            kind="scatter",
        )
        plt.suptitle(f"{' '.join(name.split(' ')[:-1])} By Time of Day")
        plt.subplots_adjust(top=0.85)
        plt.savefig(os.path.join(plot_dir_spl, f"{prefix}_hour.png"))
        plt.close()
