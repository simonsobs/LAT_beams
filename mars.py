from sotodlib.core import Context, AxisManager
from lat_beams import beam as lb
import matplotlib.pyplot as plt
import numpy as np
import os
from sotodlib import tod_ops
from so3g.proj import RangesMatrix
from sotodlib.coords import planets as cp
from scipy.ndimage import gaussian_filter
from matplotlib.colors import SymLogNorm
from lat_beams.fitting import pointing_quickfit
from so3g.proj import RangesMatrix

plt.rcParams["image.cmap"] = "PRGn"


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


ws_cent = np.array([-0.023696372, -0.021090211])
ot_cent = np.array((-0.026882106, -0.015573051))
ds = 5

plot_dir = "/so/home/saianeesh/plots/first_light_unfocused/scratch/mars/mv28"
data_dir = "/so/home/saianeesh/data/first_light_unfocused/scratch/mars/mv28"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

ctx = Context("/so/metadata/lat/contexts/smurf_detcal.yaml")
obslist = ctx.obsdb.query('type=="obs" and mars', tags=["mars=1"])
obs_id = "obs_1740190513_lati1_111"
meta = ctx.get_meta(obs_id)
meta.restrict("dets", meta.det_info.stream_id == "ufm_mv28")
meta.restrict("dets", meta.det_cal.bg > -1)
meta.restrict("dets", np.isfinite(meta.det_cal.tau_eff))
bp = (meta.det_cal.bg % 4) // 2
meta.restrict("dets", bp == 0)
aman = ctx.get_obs(meta)
filt = tod_ops.filters.iir_filter(invert=True)
aman.signal = tod_ops.filters.fourier_filter(aman, filt, signal_name="signal")
filt = tod_ops.filters.timeconst_filter(timeconst=aman.det_cal.tau_eff, invert=True)
aman.signal = tod_ops.filters.fourier_filter(aman, filt, signal_name="signal")
aman = lb.downsample_obs(aman, ds)
aman.signal -= np.mean(aman.signal, axis=-1)[..., None]
ptp = np.ptp(aman.signal, axis=-1)
aman = aman.restrict("dets", ptp < 2 * np.median(ptp))
plt.plot(aman.signal[::3].T, alpha=0.3)
plt.savefig(os.path.join(plot_dir, f"tod.png"))
plt.close()
filt = tod_ops.filters.high_pass_butter4(10)
sig_filt = tod_ops.filters.fourier_filter(aman, filt)
plt.close()
plt.plot(sig_filt.T, alpha=0.3)
plt.savefig(os.path.join(plot_dir, f"tod_filt.png"))
res = (2 / 300.0) * np.pi / 180.0
x = 0
y = 0
mask = {"shape": "circle", "xyr": (x, y, 1.5)}
aman_dummy = aman.restrict("dets", [aman.dets.vals[0]], in_place=False)
fp = AxisManager(aman_dummy.dets)
fp.wrap("xi", np.zeros(1), [(0, "dets")])
fp.wrap("eta", np.zeros(1), [(0, "dets")])
fp.wrap("gamma", np.zeros(1), [(0, "dets")])
aman_dummy.wrap("focal_plane", fp)
source_flags = cp.compute_source_flags(
    tod=aman_dummy,
    P=None,
    mask=mask,
    center_on="mars",
    res=res * 10,
    max_pix=4e8,
    wrap=None,
)
start = source_flags.ranges[0].ranges()[0][0]
stop = source_flags.ranges[0].ranges()[-1][-1]
aman = aman.restrict("samps", (start, stop))
sig_filt = sig_filt[:, start:stop]
ptp = np.ptp(sig_filt, axis=-1)
thresh = 0.1 * np.percentile(ptp, 90)
aman = aman.restrict("dets", ptp > thresh)
sig_filt = sig_filt[ptp > thresh]

if os.path.isfile("mars_fp.h5"):
    focal_plane = AxisManager.load("mars_fp.h5")
    aman.restrict("dets", np.isin(aman.dets.vals, focal_plane.dets.vals))
    aman.wrap("focal_plane", focal_plane)
else:
    focal_plane = pointing_quickfit(
        aman, (4, None), fwhm=np.deg2rad(1.5 / 60.0), source="mars", bin_priors=True
    )
    aman.wrap("focal_plane", focal_plane)
    med_xi = np.median(aman.focal_plane.xi)
    med_eta = np.median(aman.focal_plane.eta)
    msk = (
        np.sqrt(
            (aman.focal_plane.xi - med_xi) ** 2
            + np.abs(aman.focal_plane.eta - med_eta) ** 2
        )
        < np.deg2rad(0.5)
    ) * (aman.focal_plane.amp > 0)
    aman.restrict("dets", msk)
    plt.close()
    plt.scatter(aman.focal_plane.xi, aman.focal_plane.eta)
    plt.savefig(os.path.join(plot_dir, f"fp.png"))
    plt.close()
    plt.hist(aman.focal_plane.amp, bins=30)
    plt.savefig(os.path.join(plot_dir, f"fp_amp.png"))
    focal_plane.save("mars_fp.h5")


aman.restrict("dets", np.isfinite(aman.det_cal.phase_to_pW))
aman.signal *= aman.det_cal.phase_to_pW[..., None]
print(aman)
orig = aman.copy()
cuts = RangesMatrix.zeros(aman.signal.shape)
res = (2 / 300.0) * np.pi / 180.0
x = 0
y = 0
mask = {"shape": "circle", "xyr": (x, y, 0.1)}
source_flags = cp.compute_source_flags(
    tod=aman, P=None, mask=mask, center_on="mars", res=res, max_pix=4e8, wrap=None
)
print(
    f"{np.sum(source_flags.get_stats()['samples'])*np.mean(np.diff(aman.timestamps))} detector seconds on source"
)
out = cp.make_map(
    aman,
    center_on="mars",
    res=res,
    cuts=cuts,
    source_flags=source_flags,
    comps="T",
    filename=os.path.join(data_dir, "{obs_id}_{map}.fits"),
    info={"obs_id": obs_id},
)
smoothed = gaussian_filter(out["solved"][0], sigma=1)

cent = np.unravel_index(np.argmax(smoothed, axis=None), smoothed.shape)

plt.close()
plt.imshow(out["solved"][0])
plt.xlim((cent[1] - 50, cent[1] + 50))
plt.ylim((cent[0] - 50, cent[0] + 50))
plt.colorbar()
plt.grid()
plt.savefig(os.path.join(plot_dir, f"map.png"))

plt.close()
l10 = np.log10(out["solved"][0] + np.min(out["solved"][0]) + 1e-5)
plt.imshow(out["solved"][0], norm=SymLogNorm(1e-6))
plt.xlim((cent[1] - 50, cent[1] + 50))
plt.ylim((cent[0] - 50, cent[0] + 50))
plt.colorbar()
plt.grid()
plt.savefig(os.path.join(plot_dir, f"map_log10.png"))

rprof = radial_profile(smoothed, cent[::-1])
rlen = len(rprof)
rprof = np.hstack([np.flip(rprof), rprof])
plt.close()
fig, axd = plt.subplot_mosaic(
    [["A", "A", "A"], ["A", "A", "A"], ["A", "A", "A"], ["B", "B", "B"]],
    layout="constrained",
    figsize=(7, 10),
)
size = 30
axd["A"].imshow(
    smoothed[cent[0] - size : cent[0] + size, cent[1] - size : cent[1] + size],
    origin="lower",
)
axd["A"].set_xticks([])
axd["A"].set_yticks([])
axd["B"].plot(rprof[rlen - size : rlen + size])
axd["B"].set_xticks([])
axd["B"].set_yticks([])
plt.savefig(os.path.join(plot_dir, f"map_smooth.png"), bbox_inches="tight")


xi_off = np.mean(aman.focal_plane.xi) - ws_cent[0]
eta_off = np.mean(aman.focal_plane.xi) - ws_cent[1]
ot_cent -= np.array([xi_off, eta_off])
dist = np.sqrt(
    (ot_cent[0] - aman.focal_plane.xi) ** 2 + (ot_cent[1] - aman.focal_plane.eta) ** 2
)
msk = dist > np.deg2rad(0.5)
msk = np.repeat(msk, int(aman.samps.count)).reshape(len(msk), -1)
ranges = RangesMatrix.from_mask(msk)
splits = {"Within .5 deg": ranges, "Outside .5 deg": ~ranges}
out = cp.make_map(
    aman,
    center_on="mars",
    res=res,
    cuts=cuts,
    data_splits=splits,
    source_flags=source_flags,
    comps="T",
)

for split in out["splits"].keys():
    dat = out["splits"][split]
    title = split.lower().replace(" ", "_")
    plt.close()
    plt.title(split)
    plt.imshow(dat["solved"][0])
    plt.xlim((cent[1] - 50, cent[1] + 50))
    plt.ylim((cent[0] - 50, cent[0] + 50))
    plt.colorbar()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, f"map_{title}.png"))

    plt.close()
    plt.title(split)
    plt.imshow(dat["solved"][0], norm=SymLogNorm(1e-6))
    plt.xlim((cent[1] - 50, cent[1] + 50))
    plt.ylim((cent[0] - 50, cent[0] + 50))
    plt.colorbar()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, f"map_{title}_log10.png"))
