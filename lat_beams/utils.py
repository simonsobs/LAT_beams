import os
import sys

import numpy as np
from astropy import units as u
from pixell import enmap, utils
from sotodlib.core import AxisManager

from .beam_utils import crop_maps

try:
    import mpi4py.rc

    mpi4py.rc.threads = False
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
except:
    comm = None


def print_once(*args):
    """
    Helper function to print only once when running with MPI.
    Only the rank 0 process will print.

    Parameters
    ----------
    *args : Unpack[tuple[Any, ...]]
        Arguments to pass to print.
    """
    if comm is None or comm.Get_rank() == 0:
        print(*args)
        sys.stdout.flush()


def subpix_shift(imap, ishape, iwcs):
    crdelt = iwcs.wcs.crval - imap.wcs.wcs.crval
    cpdelt = iwcs.wcs.crpix - imap.wcs.wcs.crpix
    subpix = (crdelt / iwcs.wcs.cdelt - cpdelt + 0.5) % 1 - 0.5
    imap2 = enmap.fractional_shift(imap, -1 * subpix[::-1], nofft=False)

    return imap2


def coadd(imaps, iweights, medsub=True):
    # Get a joint geometry and init output maps
    ishape, iwcs = imaps[0].shape, imaps[0].wcs
    imaps = [subpix_shift(imap, ishape, iwcs) for imap in imaps]
    iweights = [subpix_shift(imap, ishape, iwcs) for imap in iweights]
    oshape, owcs = enmap.union_geometry([im.geometry for im in imaps])
    omap = enmap.zeros((len(imaps[0].shape),) + oshape, owcs)
    oweight = enmap.zeros((len(imaps[0].shape),) + oshape, owcs)

    # Coadd
    op = np.ndarray.__iadd__
    for im, iw in zip(imaps, iweights):
        oweight.insert(iw, op=op)
        off = 0
        # if medsub:
        #     off = utils.weighted_median(im, iw, axis=0)
        omap.insert(iw * (im - off), op=op)
    with np.errstate(divide="ignore", invalid="ignore"):
        omap /= oweight
    np.nan_to_num(omap, copy=False, nan=0, posinf=0, neginf=0)

    return omap, oweight


def recenter(imap, obs_id, stream_id, band, fit_file, norm=True, extent=None):
    aman_path = os.path.join(obs_id, stream_id, band)
    aman = AxisManager.load(fit_file, aman_path)
    cent = enmap.sky2pix(
        imap.shape,
        imap.wcs,
        np.array((aman.eta0.to(u.rad).value, aman.xi0.to(u.rad).value)),
    )
    zero = enmap.sky2pix(imap.shape, imap.wcs, np.array((0, 0)))
    imap = enmap.shift(imap, cent - zero)

    if norm:
        imap = (imap) / aman.amp.value

    if extent is not None:
        imap = crop_maps(
            [imap],
            enmap.sky2pix(imap.shape, imap.wcs, np.array((0, 0))),
            extent / np.abs(3600 * imap.wcs.wcs.cdelt[1]),
        )[0]

    return imap
