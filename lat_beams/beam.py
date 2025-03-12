from sotodlib.core import AxisManager, IndexAxis
from sotodlib import coords
import numpy as np
from pixell import utils, resample
from . import noise as nn

def downsample_obs(obs, down):
    """Downsample AxisManager obs by the integer factor down.


    This implementation is quite specific and probably needs
    generalization in the future, but it should work correctly
    and efficiently for ACT-like data at least. In particular
    it uses fourier-resampling when downsampling the detector
    timestreams to avoid both aliasing noise and introducing
    a transfer function."""
    assert down == utils.nint(down), "Only integer downsampling supported, but got '%.8g'" % down
    # Compute how many samples we will end up with
    onsamp = (obs.samps.count+down-1)//down
    # Set up our output axis manager
    res    = AxisManager(obs.dets, IndexAxis("samps", onsamp))
    # Stuff without sample axes
    for key, axes in obs._assignments.items():
        if "samps" not in axes:
            val = getattr(obs, key)
            if isinstance(val, AxisManager):
                res.wrap(key, val)
            else:
                axdesc = [(k,v) for k,v in enumerate(axes) if v is not None]
                res.wrap(key, val, axdesc)
    # The normal sample stuff
    res.wrap("timestamps", obs.timestamps[::down], [(0, "samps")])
    bore = AxisManager(IndexAxis("samps", onsamp))
    for key in ["az", "el", "roll"]:
        bore.wrap(key, getattr(obs.boresight, key)[::down], [(0, "samps")])
    res.wrap("boresight", bore)
    res.wrap("signal", resample.resample_fft_simple(obs.signal, onsamp), [(0,"dets"),(1,"samps")])

    return res
def dirty_source(aman, wcs, n_rounds=3, fwhm=20, ds=20, P=None):
    print("Downsampling")
    aman = downsample_obs(aman, ds) 
    aman.signal = aman.signal.astype(np.float32)
    if P is None:
        P = coords.P.for_tod(aman, wcs_kernel=wcs)
    buf = np.zeros_like(aman.signal)
    omap = P.to_map(aman, signal=buf)
    for i in range(n_rounds):
        print(f"Running round {i}")
        P.from_map(omap, dest=buf)
        print("\tComputing noise")
        nn.compute_noise(aman, aman.signal - buf, fwhm)
        print("\tApplying noise")
        dat_filt = nn.apply_noise(aman, None)
        print("\tBinning to map")
        omap = P.remove_weights(tod=aman, signal=dat_filt)
        buf[:] = 0

    return omap, aman

def dirty_source_no_pointing(aman, wcs, n_rounds=3, fwhm=20, ds=20):
    if 'focal_plane' in aman:
        aman.move("focal_plane", None)
    focal_plane = AxisManager(aman.dets)
    focal_plane.wrap("xi", np.zeros(aman.dets.count), [(0, "dets")])
    focal_plane.wrap("eta", np.zeros(aman.dets.count), [(0, "dets")])
    focal_plane.wrap("gamma", np.zeros(aman.dets.count), [(0, "dets")])
    aman.wrap("focal_plane", focal_plane)

    return dirty_source(aman, wcs, n_rounds, fwhm, ds)
