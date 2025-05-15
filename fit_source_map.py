from pixell import enmap
from sotodlib.core import Context, AxisManager
import h5py
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from so3g.proj import RangesMatrix
from sotodlib.core import metadata
import sys
import argparse
import yaml
from lat_beams.pointing_model import apply_pointing_model
from lat_beams.fitting import fit_gauss_beam
from mpi4py import MPI
import h5py
from astropy import units as u

plt.rcParams["image.cmap"] = "RdGy_r"


comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()


def get_cent(imap, buf=30, sigma=1):
    smoothed = gaussian_filter(imap, sigma=sigma)
    smoothed[:buf] = 0
    smoothed[-1 * buf :] = 0
    smoothed[:, :buf] = 0
    smoothed[:, -1 * buf :] = 0
    cent = np.unravel_index(np.argmax(smoothed, axis=None), smoothed.shape)

    return cent


def crop_maps(solved, weights, extent, buf=30, sigma=1):
    cent = get_cent(solved, buf, sigma)
    xmin = max(0, cent[0] - extent)
    xmax = min(solved.shape[0], cent[0] + extent)
    ymin = max(0, cent[1] - extent)
    ymax = min(solved.shape[1], cent[1] + extent)
    solved = solved[xmin:xmax, ymin:ymax]
    weights = weights[xmin:xmax, ymin:ymax]
    return solved, weights


parser = argparse.ArgumentParser()
parser.add_argument("cfg", help="Path to the config file")
parser.add_argument("--obs_ids", nargs="+", help="Pass a list of obs ids to run on")
parser.add_argument(
    "--overwrite", "-o", action="store_true", help="Overwrite an existing fit"
)
parser.add_argument(
    "--start_from", "-s", default=0, type=int, help="Skip to the nth obs (0 indexed)"
)
parser.add_argument(
    "--lookback",
    "-l",
    type=float,
    help="Amount of time to lookback for query, overides start time from config",
)
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

# Get some global settings
source = cfg.get("source", "mars")
extent = cfg.get("extent", 900)
snr_extent = cfg.get("snr_extent", 360)
min_sigma = cfg.get("min_sigma_fit", 3)
min_snr = cfg.get("min_snr", 10)

# Setup folders
root_dir = os.path.expanduser(cfg.get("root_dir", "~"))
project_dir = cfg.get("project_dir", "beams/lat")
plot_dir = os.path.join(root_dir, "plots", project_dir, "source_map_fits", source)
data_dir = os.path.join(root_dir, "data", project_dir, "source_map_fits", source)
map_dir = os.path.join(root_dir, "data", project_dir, "source_maps", source)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
# outfile = h5py.File(os.path.join(data_dir, 'beam_pars.h5'), 'a', driver='mpio', comm=comm)
if myrank == 0:
    outfile = h5py.File(os.path.join(data_dir, "beam_pars.h5"), "a")

# Get the list of files
flist = sorted(glob.glob(map_dir + "/*/*/*_solved.fits"))

# Get the obs_id, stream id, and band
obs_ids = []
stream_ids = []
bands = []
for fname in flist:
    parts = os.path.basename(fname).split("_")
    obs_ids += [f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"]
    stream_ids += [f"{parts[4]}_{parts[5]}"]
    bands += [parts[6]]
obs_ids = np.array(obs_ids)
stream_ids = np.array(stream_ids)
bands = np.array(bands)
flist = np.array(flist)

# Metadata mods must be done by all
# for fname, obs_id, stream_id, band in zip(flist, obs_ids, stream_ids, bands):
#     outfile.require_group(os.path.join(obs_id, stream_id, band))

if args.obs_ids is None:
    # Limit to the time range
    start_time = cfg["start_time"]
    if args.lookback is not None:
        start_time = time.time() - 3600 * args.lookback
    stop_time = cfg["stop_time"]
    ctx = Context(cfg.get("context", "/so/metadata/lat/contexts/smurf_detcal.yaml"))
    if ctx.obsdb is None:
        raise ValueError("No obsdb in context!")
    start_times = np.array([ctx.obsdb.get(obs_id)["start_time"] for obs_id in obs_ids])
    stop_times = np.array([ctx.obsdb.get(obs_id)["stop_time"] for obs_id in obs_ids])
    msk = (start_times > start_time) * (stop_times < stop_time)
else:
    msk = np.isin(obs_ids, args.obs_ids)
obs_ids = obs_ids[msk]
stream_ids = stream_ids[msk]
bands = bands[msk]
flist = flist[msk]

print(f"{len(flist)} maps found")
print(f"Starting from {args.start_from}")
obs_ids = obs_ids[args.start_from :]
stream_ids = stream_ids[args.start_from :]
bands = bands[args.start_from :]
flist = flist[args.start_from :]

# Remove files that we already have
already_have = []
if not args.overwrite and myrank == 0:
    for fname, obs_id, stream_id, band in zip(flist, obs_ids, stream_ids, bands):
        path = os.path.join(obs_id, stream_id, band)
        if path in outfile:
            already_have += [fname]
already_have = comm.bcast(already_have, root=0)

max_split = int(len(flist) // nproc) * nproc

lo_flist = flist[max_split:]
lo_obs_ids = obs_ids[max_split:]
lo_stream_ids = stream_ids[max_split:]
lo_bands = bands[max_split:]

flist = np.array_split(flist[:max_split], nproc)[myrank]
obs_ids = np.array_split(obs_ids[:max_split], nproc)[myrank]
stream_ids = np.array_split(stream_ids[:max_split], nproc)[myrank]
bands = np.array_split(bands[:max_split], nproc)[myrank]

nmaps = len(flist)
# Shove all the leftovers into rank 0 for now
if myrank == 0:
    flist = np.hstack([flist, lo_flist])
    obs_ids = np.hstack([obs_ids, lo_obs_ids])
    stream_ids = np.hstack([stream_ids, lo_stream_ids])
    bands = np.hstack([bands, lo_bands])

par_names = ["amp", "dec0", "ra0", "fwhm_dec", "fwhm_ra", "theta", "offset"]
par_units = [u.pW, u.arcsec, u.arcsec, u.arcsec, u.arcsec, u.radian, u.pW]
to_save = (None, None)
for i, (fname, obs_id, stream_id, band) in enumerate(
    zip(flist, obs_ids, stream_ids, bands)
):
    if i < nmaps:
        comm.barrier()
        to_save = comm.gather(to_save, root=0)
    else:
        to_save = [to_save]
    if myrank == 0:
        for aman, path in to_save:
            if aman is None:
                continue
            aman.save(outfile, path, overwrite=True)
        if i % 10 == 9:
            outfile.flush()
    sys.stdout.flush()
    print(f"Fitting {obs_id}_{stream_id}_{band} ({i+1}/{len(flist)})")
    if fname in already_have:
        print("\tAlready in output file and overwrite not set. Skipping...")
        to_save = (None, None)
        continue
    wpath = fname[::-1].replace("solved"[::-1], "weights"[::-1], 1)[::-1]
    if not os.path.isfile(wpath):
        print("\tNo weights map found! Skipping...")
        to_save = (None, None)
        continue

    # Load the maps
    solved = enmap.read_map(fname)[0]
    weights = enmap.read_map(wpath)[0][0]
    pixsize = 3600 * solved.wcs.wcs.cdelt[1]

    # Check if this is a bogus map
    if np.sum(~(weights == 0)) == 0:
        print("\tWeights all 0. Skipping...")
        to_save = (None, None)
        continue

    # Estimate SNR
    snr_extent_pix = int(snr_extent // pixsize)
    cent = get_cent(solved)
    sig = solved[cent]
    noise = solved.copy()
    xmin = max(0, cent[0] - snr_extent_pix)
    xmax = min(solved.shape[0], cent[0] + snr_extent_pix)
    ymin = max(0, cent[1] - snr_extent_pix)
    ymax = min(solved.shape[1], cent[1] + snr_extent_pix)
    noise[xmin:xmax, ymin:ymax] = np.nan
    noise = np.nanstd(noise)
    snr = sig / noise

    if snr < min_snr:
        print("\tSNR too low! Skipping")
        to_save = (None, None)
        continue

    # Slice things
    solved, weights = crop_maps(solved, weights, int(extent // pixsize))
    pixmap = enmap.pixmap(solved.shape, solved.wcs)

    # Fit
    cent = get_cent(solved)
    res = fit_gauss_beam(solved, weights, pixmap, cent, min_sigma)
    if res is None:
        print("\tFit failed! Probably not a good map?")
        to_save = (None, None)
        continue
    (
        popt,
        perr,
        model,
        data_fwhm,
        data_solid_angle,
        model_solid_angle,
        r,
        rprof,
        mprof,
    ) = res

    # Adjust shift
    dec, ra = 3600 * np.rad2deg(solved.pix2sky((0, 0)))
    popt[2] = ra - popt[2]
    popt[1] += dec

    # Plot
    os.makedirs(
        os.path.join(plot_dir, str(obs_id.split("_")[1])[:5], obs_id), exist_ok=True
    )
    [[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(
        solved.corners(corner=False)
    )
    plt_extent = [ra_min, ra_max, dec_min, dec_max]

    plt.imshow(model, origin="lower", extent=plt_extent)
    plt.xlabel('RA (")')
    plt.ylabel('Dec (")')
    plt.title(f"{obs_id}_{stream_id}_{band} Model")
    plt.colorbar()
    plt.savefig(
        os.path.join(
            plot_dir,
            str(obs_id.split("_")[1])[:5],
            obs_id,
            f"{obs_id}_{stream_id}_{band}_model.png",
        )
    )
    plt.clf()

    plt.imshow(solved - model, origin="lower", extent=plt_extent)
    plt.xlabel('RA (")')
    plt.ylabel('Dec (")')
    plt.title(f"{obs_id}_{stream_id}_{band} Residual")
    plt.colorbar()
    plt.savefig(
        os.path.join(
            plot_dir,
            str(obs_id.split("_")[1])[:5],
            obs_id,
            f"{obs_id}_{stream_id}_{band}_residual.png",
        )
    )
    plt.clf()

    plt.plot(
        r,
        rprof,
        alpha=0.6,
        label=f'Data \nFWHM={data_fwhm:.2f}", Ω={data_solid_angle:.2f} sr',
    )
    plt.plot(
        r,
        mprof,
        alpha=0.6,
        label=f'Model \nFWHM=({popt[3]:.2f}, {popt[4]:.2f})", Ω={model_solid_angle:.2f} sr',
    )
    plt.xlabel('Radius (")')
    plt.ylabel("Power (pW)")
    plt.title(f"{obs_id}_{stream_id}_{band} Profile")
    plt.legend()
    plt.savefig(
        os.path.join(
            plot_dir,
            str(obs_id.split("_")[1])[:5],
            obs_id,
            f"{obs_id}_{stream_id}_{band}_profile.png",
        )
    )
    plt.clf()

    plt.plot(r, rprof - mprof)
    plt.xlabel('Radius (")')
    plt.ylabel("Power (pW)")
    plt.title(f"{obs_id}_{stream_id}_{band} Profile Residual")
    plt.savefig(
        os.path.join(
            plot_dir,
            str(obs_id.split("_")[1])[:5],
            obs_id,
            f"{obs_id}_{stream_id}_{band}_profile_resid.png",
        )
    )
    plt.clf()

    # Save
    aman = AxisManager()
    for name, unit, par, err in zip(par_names, par_units, popt, perr):
        aman.wrap(name, par * unit)
        aman.wrap(f"{name}_err", err * unit)
    aman.wrap("data_fwhm", data_fwhm * u.arcsec)
    aman.wrap("data_solid_angle", data_solid_angle * u.sr)
    aman.wrap("model_solid_angle", model_solid_angle * u.sr)
    aman.wrap("noise", noise * u.pW)
    aman_path = os.path.join(obs_id, stream_id, band)
    to_save = (aman, aman_path)

comm.barrier()
to_save = comm.gather(to_save, root=0)
if myrank == 0:
    for aman, path in to_save:
        if aman is None:
            continue
        aman.save(outfile, path, overwrite=True)
    outfile.close()
sys.stdout.flush()
