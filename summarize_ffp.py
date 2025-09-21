# TODO: make this generic
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from sotodlib.coords import fp_containers as fpc

epochs = [(1744848000, 1745207940), (1745207940, 1749600000), (1749600000, 2e9)]

fpath = "/global/cfs/cdirs/sobs/users/skh/data/pointing/lat/finalize_focal_plane/results/per_obs/focal_plane.h5"
dat = fpc.Receiver.load_file(fpath)
pdir = "/global/cfs/cdirs/sobs/users/skh/plots_raw/pointing/lat/finalize_focal_plane/per_obs/summary"
os.makedirs(pdir, exist_ok=True)

times = []
ufms = []
n_dets = []
xi_scales = []
eta_scales = []
for obs_id in dat.keys():
    time = float(obs_id.split("_")[1])
    for ot in dat[obs_id].optics_tubes:
        for fp in ot.focal_planes:
            n_det = np.sum(np.isfinite(fp.weights) * (fp.weights > 0.5))
            times += [time]
            ufms += [fp.stream_id]
            n_dets += [n_det]
            xi_scales += [fp.transform.scale[0]]
            eta_scales += [fp.transform.scale[1]]
times = np.array(times)
ufms = np.array(ufms)
n_dets = np.array(n_dets)
xi_scales = np.array(xi_scales)
eta_scales = np.array(eta_scales)

for start, stop in epochs:
    msk = (times >= start) * (times < stop)
    print(f"{np.sum(msk)} fit tods from {start} to {stop}")
    for ufm in np.unique(ufms):
        u_msk = ufms == ufm
        print(f"\t{np.sum(msk * u_msk)} fit tods for {ufm}")
        n = n_dets[msk * u_msk]
        print(f"\t{np.median(n)} ± {np.std(n)} median ± std fit dets for {ufm}")

plt.scatter(times, ufms, c=n_dets.astype(float), alpha=0.4)
plt.colorbar(label="Fit Detectors")
elines = np.unique(np.array(epochs))[:-1]
for el in elines:
    plt.axvline(el)
plt.xlim((elines[0], None))
plt.xlabel("ctime")
plt.ylabel("Array")
plt.title(f"Fit Detectors By Array")
plt.savefig(os.path.join(pdir, f"ndet_by_array.png"))
plt.close()

plt.scatter(times, ufms, c=xi_scales, vmin=0.9, vmax=1.1)
elines = np.unique(np.array(epochs))[:-1]
for el in elines:
    plt.axvline(el)
plt.colorbar(label="scale")
plt.xlabel("ctime")
plt.ylabel("Array")
plt.title(f"Xi Scale")
plt.savefig(os.path.join(pdir, f"xi_scale.png"))
plt.close()

plt.scatter(times, ufms, c=eta_scales, vmin=0.9, vmax=1.1)
elines = np.unique(np.array(epochs))[:-1]
for el in elines:
    plt.axvline(el)
plt.colorbar(label="scale")
plt.xlabel("ctime")
plt.ylabel("Array")
plt.title(f"Eta Scale")
plt.savefig(os.path.join(pdir, f"eta_scale.png"))
plt.close()

plt.hist(xi_scales[times > epochs[-1][0]], bins="auto", alpha=0.5, label="Xi")
plt.hist(eta_scales[times > epochs[-1][0]], bins="auto", alpha=0.5, label="Eta")
plt.legend()
plt.xlim((1, None))
plt.xlabel("Scale factor")
plt.ylabel("Counts")
plt.title(f"Scale Factors")
plt.savefig(os.path.join(pdir, f"scale.png"))
plt.close()
