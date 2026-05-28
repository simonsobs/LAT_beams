import os
import sys

import astropy.units as u
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from pixell import enmap
from sotodlib.core import Context
from sotodlib.io import hkdb
from tqdm import tqdm

from lat_beams import beam_utils as bu
from lat_beams.utils import get_args_cfg, make_jobdb, setup_cfg, setup_paths
from lat_beams.plotting import auto_relplot

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

# Loop through splits
for split in cfg.split_by:
    if "source" not in split:
        continue
    print(f"Splitting by {split}")
    data_dir_spl = os.path.join(data_dir, "stack_profiles", split)
    plot_dir_spl = os.path.join(plot_dir, split)
    if not os.path.isdir(data_dir_spl):
        print("\tNo splits found!")
        continue
    os.makedirs(plot_dir_spl, exist_ok=True)
    to_plot = {s : [] for s in split.split("+")}
    to_plot["r"] = []
    to_plot["profile"] = []
    to_plot["epoch"] = []
    for spl_dir in sorted([f.path for f in os.scandir(data_dir_spl) if f.is_dir()]):
        spl_rel = os.path.relpath(spl_dir, data_dir_spl)
        prof_dir = os.path.join(data_dir, "stack_profiles", split, spl_rel)
        for epoch in cfg.epochs:
            prof_path = os.path.join(prof_dir, f"profile_{spl_rel}_{epoch[0]}_{epoch[1]}.txt")
            if not os.path.isfile(prof_path):
                continue
            profile = np.genfromtxt(prof_path)
            msk = profile[:, 0] < 1200*cfg.mask_size

            to_plot["epoch"] += [f"{epoch[0]}_{epoch[1]}"] * np.sum(msk)
            to_plot["r"] += profile[msk, 0].tolist()
            to_plot["profile"] += profile[msk, 1].tolist()
            for sc, si in zip(split.split("+"), spl_rel.split("+")):
                to_plot[sc] += [si] * np.sum(msk)
    plt.close()
    plot = auto_relplot(to_plot, x="r", y="profile", kind="line", estimator=None, hue="source", col="band")
    plot.set_axis_labels('r (")', 'Beam Profile')
    plot.set(xlim=(0, 3600*cfg.mask_size/3), yscale="log")
    plt.suptitle(f"Beam Profile by {split}")
    plt.subplots_adjust(top=0.85)
    plt.savefig(
        os.path.join(plot_dir_spl, f"source_cmp_{split}.png"), bbox_inches="tight"
    )
