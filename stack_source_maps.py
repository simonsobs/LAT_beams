import argparse
import glob
import os
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy import units as u
from pixell import enmap
from sotodlib.core import AxisManager, Context
from matplotlib.colors import SymLogNorm

from lat_beams.beam_utils import (
    crop_maps,
    estimate_solid_angle,
    get_cent,
    get_fwhm_radial_bins,
    radial_profile,
)
from lat_beams.fitting import fit_gauss_beam
from lat_beams.utils import print_once, coadd, recenter

plt.rcParams["image.cmap"] = "RdGy_r"
comps = ["T", "Q", "U"]


def plot_map(imap, title, out_file):
    [[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(
        imap.corners(corner=False)
    )
    plt_extent = (ra_min, ra_max, dec_min, dec_max)
    plt.imshow(imap, origin="lower", extent=plt_extent)
    plt.colorbar()
    plt.grid()
    plt.xlabel('Xi (")')
    plt.ylabel('Eta (")')
    plt.title(title)
    plt.savefig(out_file)
    plt.close()
    
    _norm = SymLogNorm(linthresh=1e-3 * np.max(np.abs(imap)), clip=True)
    plt.imshow(imap, origin="lower", extent=plt_extent, norm=_norm)
    plt.colorbar()
    plt.grid()
    plt.xlabel('Xi (")')
    plt.ylabel('Eta (")')
    plt.title(title)
    root, ext = os.path.splitext(out_file)
    plt.savefig(f"{root}_log{ext}")
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument("cfg", help="Path to the config file")
parser.add_argument("--source", "-s", default="all", help="Which source to stack on")
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

# Get some global setting
epochs = cfg.get("epochs", [(0, 2e10)])
pointing_type = cfg.get("pointing_type", "pointing_model")
split_by = cfg.get("split_by", ["band", "band-stream_id"])
extent = cfg.get("extent", 900)

# Setup folders
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "beams/lat")
plot_dir = os.path.join(
    root_dir, "plots", project_dir, "source_maps", pointing_type, "stacks", args.source 
)
out_dir = os.path.join( root_dir, "data", project_dir, "source_maps", pointing_type, "stacks", args.source) 
data_dir = os.path.join( root_dir, "data", project_dir, "source_maps", pointing_type)
os.makedirs(plot_dir, exist_ok=True)
fit_file = os.path.join(data_dir, "fits", "beam_pars.h5")

# Get the list of files
src = "*" if args.source == "all" else args.source
flist = sorted(glob.glob(data_dir + f"/{src}/*/*/*_solved.fits"))
obs_ids = []
times = []
stream_ids = []
bands = []
wlist = []
rlist = []
rwlist = []
for fname in flist:
    parts = os.path.basename(fname).split("_")
    obs_ids += [f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"]
    times += [float(parts[1])]
    stream_ids += [f"{parts[4]}_{parts[5]}"]
    bands += [parts[6]]
    wlist += [fname[::-1].replace("solved"[::-1], "weights"[::-1], 1)[::-1]]
    rlist += [fname[::-1].replace("solved"[::-1], "resid"[::-1], 1)[::-1]]
    rwlist += [fname[::-1].replace("solved"[::-1], "resid_weights"[::-1], 1)[::-1]]
obs_ids = np.array(obs_ids)
times = np.array(times)
stream_ids = np.array(stream_ids)
bands = np.array(bands)
flist = np.array(flist)
wlist = np.array(wlist)
rlist = np.array(rlist)
rwlist = np.array(rwlist)
dtype = [
    ("obs_id", obs_ids.dtype),
    ("stream_id", stream_ids.dtype),
    ("band", bands.dtype),
    ("time", float),
    ("fname", flist.dtype),
    ("wname", wlist.dtype),
    ("rname", rlist.dtype),
    ("rwname", rwlist.dtype),
]
all_maps = np.fromiter( zip(obs_ids, stream_ids, bands, times, flist, wlist, rlist, rwlist), dtype, count=len(flist))
msk = np.array([os.path.isfile(wpath) for wpath in all_maps["wname"]])
all_maps = all_maps[msk]
msk = np.zeros(len(all_maps), bool)
file = h5py.File(fit_file)
for i in range(len(all_maps)):
    aman_path = os.path.join(all_maps["obs_id"][i], all_maps["stream_id"][i], all_maps["band"][i])
    msk[i] = aman_path in file
file.close()
all_maps = all_maps[msk]

print(f"{len(all_maps)} to be stacked")

# Stack by splits
for split in split_by:
    print(f"Splitting by {split}")
    split_vec = np.array([all_maps[spl] for spl in split.split("-")], dtype=str).T
    for spl in np.unique(split_vec, axis=0):
        out_dir_spl = os.path.join(out_dir, split, "-".join(spl))
        plot_dir_spl = os.path.join(plot_dir, split, "-".join(spl))
        os.makedirs(out_dir_spl, exist_ok=True)
        os.makedirs(plot_dir_spl, exist_ok=True)
        print(f"{spl}")
        smaps = all_maps[np.all(split_vec == spl, axis=-1)]
        for epoch in epochs:
            print(f"\tEpoch {epoch}")
            times = smaps["time"]
            tmsk = (times >= epoch[0]) * (times < epoch[1])
            if np.sum(tmsk) == 0:
                print(f"\t\tNo maps found! Skipping...")
                continue
            maps = smaps[tmsk]

            imaps = [recenter(enmap.read_map(fname), obs_id, stream_id, band, fit_file, True, extent) for fname, obs_id, stream_id, band in zip(maps["fname"], maps["obs_id"], maps["stream_id"], maps["band"])]
            wmaps = [recenter(enmap.read_map(wname)[np.diag_indices(3)], obs_id, stream_id, band, fit_file, False, extent) for wname, obs_id, stream_id, band in zip(maps["wname"], maps["obs_id"], maps["stream_id"], maps["band"])]
            omap, oweight = coadd(imaps, wmaps)
            enmap.write_map(os.path.join(out_dir_spl, f"{'-'.join(spl)}_{epoch[0]}_{epoch[1]}_solved.fits"), omap)
            enmap.write_map(os.path.join(out_dir_spl, f"{'-'.join(spl)}_{epoch[0]}_{epoch[1]}_weights.fits"), oweight)
            for i, om in enumerate(omap.reshape((-1,) + omap.shape[-2:])):
                plot_map(om, f"{spl} {comps[i]} Map Stack", os.path.join(plot_dir_spl, f"{'-'.join(spl)}_{comps[i]}_map_stack_{epoch[0]}_{epoch[1]}.png"))


            # imaps = [enmap.read_map(rname) for rname in maps["rname"]]
            # wmaps = [enmap.read_map(rwname) for rwname in maps["rwname"]]
            # # imaps, wmaps = recenter(imaps, wmaps, extent)
            # omap, oweight = coadd(imaps, wmaps)
            # (omap,), (oweight,) = recenter([omap,], [oweight,], extent, False)
            # enmap.write_map(os.path.join(out_dir_spl, f"{spl}_{epoch[0]}_{epoch[1]}_solved.fits"), omap)
            # enmap.write_map(os.path.join(out_dir_spl, f"{spl}_{epoch[0]}_{epoch[1]}_weights.fits"), oweight)
            # for i, om in enumerate(omap.reshape((-1,) + omap.shape[-2:])):
            #     plot_map(om, f"{spl} {comps[i]} Residual Stack", os.path.join(plot_dir_spl, f"{spl}_{comps[i]}_resid_stack_{epoch[0]}_{epoch[1]}.png"))

