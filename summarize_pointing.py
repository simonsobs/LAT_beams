# TODO: make this generic
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

epochs = [(1744848000, 1745207940), (1745207940, 1749600000), (1749600000, 2e9)]
min_dets = 400

fpath = "/global/cfs/cdirs/sobs/users/skh/data/pointing/lat/source_fits/tod_fits.h5"
dat = h5py.File(fpath, "r")
pdir = "/global/cfs/cdirs/sobs/users/skh/plots_raw/pointing/lat/source_fits/summary"
os.makedirs(pdir, exist_ok=True)

dat['obs_1750771728_lati3_110']['ufm_mv20'].shape[0]
times = []
ufms = []
n_dets = []
for obs_id in dat.keys():
    time = float(obs_id.split("_")[1])
    for ufm in dat[obs_id].keys():
        n_det = dat[obs_id][ufm].shape[0]
        if n_det < min_dets:
            continue
        times += [time]
        ufms += [ufm]
        n_dets += [n_det]
times = np.array(times)
ufms = np.array(ufms)
n_dets = np.array(n_dets)

for start, stop in epochs:
    msk = (times >= start) * (times < stop)
    print(f"{np.sum(msk)} fit tods from {start} to {stop}")
    for ufm in np.unique(ufms):
        u_msk = ufms == ufm
        print(f"\t{np.sum(msk * u_msk)} fit tods for {ufm}")
        n = n_dets[msk * u_msk]
        print(f"\t{np.median(n)} ± {np.std(n)} median ± std fit dets for {ufm}")

plt.scatter(times, ufms, c=n_dets.astype(float), alpha=.4)
plt.colorbar(label="Fit Detectors")
elines = np.unique(np.array(epochs))[:-1]
for el in elines:
    plt.axvline(el)
plt.xlim((elines[0], None))
plt.xlabel("ctime")
plt.ylabel("Array")
plt.title(f"Fit Detectors By Array (min {min_dets} dets)")
plt.savefig(os.path.join(pdir, f"ndet_by_array.png"))
plt.close()

