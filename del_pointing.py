"""
Delete bad fits.
Eventually we should just not have bad fits...
"""

import argparse
import os

import h5py
import numpy as np
import yaml
from sotodlib.core import Context, metadata
from sotodlib.io.metadata import read_dataset, write_dataset

band_names = {"m": ["f090", "f150"], "u": ["f220", "f280"]}
fwhm = {"f090": 2, "f150": 1.3, "f220": 0.95, "f280": 0.83}  # arcmin

parser = argparse.ArgumentParser()
parser.add_argument("cfg", help="Path to the config file")
parser.add_argument("obs_id", help="The obs_id to delete")
parser.add_argument("ufm", help="The stream_id to delete (ie. ufm_mv20)")
parser.add_argument(
    "--delete",
    action="store_true",
    help="If passed delete from the h5 file rather than nulling it out",
)
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

# Get some global settings
source = cfg.get("source", "mars")

# Setup folders
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "first_light_unfocused")
plot_dir = os.path.join(root_dir, "plots", project_dir, "source_fits", source)
data_dir = os.path.join(root_dir, "data", project_dir, "source_fits")

# Get the list of observations
ctx = Context(cfg.get("context", "/so/metadata/lat/contexts/smurf_detcal.yaml"))
if ctx.obsdb is None:
    raise ValueError("No obsdb in context!")
obs = ctx.obsdb.get(args.obs_id)

# Output metadata setup
h5_path = os.path.join(data_dir, "tod_fits.h5")
h5_file = h5py.File(h5_path, "a")

if args.delete:
    print(f"Deleting entry for {obs['obs_id']} {args.ufm}")
else:
    print(f"Nulling out entry for {obs['obs_id']} {args.ufm}")
if obs["obs_id"] not in h5_file:
    raise ValueError(f"{obs['obs_id']} not in h5 file!")
obs_plot_dir = os.path.join(plot_dir, obs["obs_id"])
dset = f"{obs['obs_id']}/{args.ufm}"
if dset not in h5_file:
    raise ValueError(f"{dset} not in h5 file!")

if args.delete:
    del h5_file[dset]
else:
    # Null out the dataset
    rset = read_dataset(h5_file, dset)
    arr = rset.asarray()
    rset = metadata.ResultSet.from_friend(np.zeros(1, dtype=arr.dtype))
    write_dataset(rset, h5_file, dset, True)

# Delete fp plots
os.remove(os.path.join(obs_plot_dir, f"{args.ufm}_fp.png"))
os.remove(os.path.join(obs_plot_dir, f"{args.ufm}_fp_amp.png"))
os.remove(os.path.join(obs_plot_dir, f"{args.ufm}_fp_fwhm.png"))

# Clean up
h5_file.close()
