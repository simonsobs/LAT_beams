from sotodlib.core import AxisManager
from sotodlib import coords
import numpy as np
from . import noise as nn

def dirty_source(aman, wcs, n_rounds=3, fwhm=20):
    P = coords.P.for_tod(aman, wcs_kernel=wcs)
    buf = np.zeros_like(aman.signal)
    omap = P.to_map(aman, signal=buf)
    for _ in range(n_rounds):
        nn.compute_noise(aman, aman.signal - P.from_map(omap, dest=omap), fwhm)
        dat_filt = nn.apply_noise(aman, None)
        weights = P.to_weights(tod=aman, signal=dat_filt)
        omap = P.remove_weights(tod=aman, signal=dat_filt, weights_map=weights)

    return omap

def dirty_source_no_pointing(aman, wcs, n_rounds=3, fwhm=20):
    if 'focal_plane' in aman:
        aman.move("focal_plane", None)
    focal_plane = AxisManager(aman.dets)
    focal_plane.wrap("xi", np.zeros(aman.dets.count), [(0, "dets")])
    focal_plane.wrap("eta", np.zeros(aman.dets.count), [(0, "dets")])
    focal_plane.wrap("gamma", np.zeros(aman.dets.count), [(0, "dets")])
    aman.wrap("focal_plane", focal_plane)

    return dirty_source(aman, wcs, n_rounds, fwhm)
