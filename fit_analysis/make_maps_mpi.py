#############################################
# WARNING: THIS FILE IS KIND OF OLD AND IT MIGHT CAUSE ERRORS IF RUN
# THIS FILE IS HERE AS AN EXAMPLE OF HOW MAPS ARE CREATED
##############################################


import warnings
import sys
import numpy as np
import argparse
import json

import logging, glob, mpi4py
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

mpi4py.rc.threads = False
from mpi4py import MPI

mpi4py.rc.threads = False
from mpi4py import MPI

from sotodlib import tod_ops
from sotodlib.core import Context, AxisManager
from sotodlib.tod_ops import flags, detrend_tod, jumps, gapfill
from sotodlib.core.flagman import has_any_cuts, has_all_cut
from sotodlib import coords
from sotodlib.coords.pointing_model import apply_pointing_model
from sotodlib.coords import planets as cp
from so3g.proj import RangesMatrix


def load_obs_from_id(
    obs_id: str,
    wafer: str,
    band: str,
    ctx: str = '/global/cfs/cdirs/sobs/users/skh/data/beams/lat/context_nersc_pointing_model.yaml',
    ) -> 'AxisManager':

    """
    Loads a specific obs data loading only a given band and wafer. Used in the map maker stage.
    """
    ctx = Context(ctx)
    obs = ctx.obsdb.get(obs_id, tags=True)
    meta = ctx.get_meta(obs_id)
    
    meta.restrict("dets", np.isfinite(meta.det_cal.tau_eff))
    db_flag = tod_ops.flags.get_det_bias_flags(meta)
    meta.restrict("dets", ~db_flag.det_bias_flags)
    meta.restrict("dets", np.isfinite(meta.det_cal.phase_to_pW))
    meta.restrict(
        "dets",
        np.isfinite(meta.focal_plane.xi)
        * np.isfinite(meta.focal_plane.eta)
        * np.isfinite(meta.focal_plane.gamma),
    )
    ufms = np.unique(meta.det_info.stream_id)
    wafer = {"ws0": 0, "ws1":1, "ws2":2}[wafer]
    meta_ufm = meta.copy().restrict("dets", meta.det_info.stream_id == ufms[wafer])
    bp = (meta_ufm.det_cal.bg % 4) // 2
    tube_band = ufms[wafer][4]
    
    if tube_band == 'm':
        band = {"f090": 0, "f150": 1}[band]
    else:
        # implement hf values
        pass
    
    meta_band = meta_ufm.copy().restrict("dets", bp == band)
    obs = ctx.get_obs(meta_band)
    return obs

def proccess_TOD(obs: 'AxisManager',
                 restrict: bool = False,
                 max_trend: float = 3.0,
                 t_piece: int = 30,
                 win_size: int = 30,
                 nsigma: int = 3,
                 nbuf: int = 30,
                 modes: int = 1,
                 use_pca: bool = False,
                 nj: int = 10,
                 ) -> 'AxisManager':
    """
    Proccesses an observation up to the point before you begin flagging sources
    for map-making.

    The order of proccessing goes:
        - Restriction of detectors
        - IIR Filter
        - Time Const Filter
        - Median Subtraction
        - Trending Flags
        - Jumps Fix
        - Glitches Fix
        - Second Median Subtraction
        - Phase to pW
    """

    ## THIS PART DIFFERS FROM SCRIPT!
    
    # Restrict Detectors
    if restrict:
        obs.restrict('dets', obs.dets.vals[obs.det_info.wafer.type == 'OPTC'])
        rfrac_range=(0.05,0.9); psat_range=(0,20)
        flags.get_det_bias_flags(obs, rfrac_range=rfrac_range, psat_range=psat_range)
        bad_dets = has_all_cut(obs.flags.det_bias_flags)
        obs.restrict('dets', obs.dets.vals[~bad_dets])
    
    # IIR Filter
    filt = tod_ops.filters.iir_filter(invert=True)
    obs.signal = tod_ops.filters.fourier_filter(obs, filt, signal_name="signal")

    # Time Constant Filter
    filt = tod_ops.filters.timeconst_filter(
                timeconst=obs.det_cal.tau_eff, invert=True
            )
    obs.signal = tod_ops.filters.fourier_filter(
                obs, filt, signal_name="signal"
            )
    # Median Subtraction
    detrend_tod(obs, method='median')

    # Trending Flags
    tf = flags.get_trending_flags(obs, max_trend=max_trend, t_piece=t_piece) 
    tdets = has_any_cuts(tf)
    obs.restrict('dets', ~tdets)

    ## Jumps fix
    jflags, _, jfix = jumps.twopi_jumps(obs, 
                                        signal=obs.signal, 
                                        fix=True, 
                                        overwrite=True,
                                        win_size=win_size,
                                        nsigma=nsigma,
                                        merge=False,
                                       )
    obs.signal = jfix
    
    cuts_fixed = has_any_cuts(jflags)

    # Glitches fix
    gfilled = gapfill.fill_glitches(obs, nbuf=nbuf, use_pca=use_pca, modes=modes,
                                        signal=obs.signal,
                                        glitch_flags=jflags)
    obs.signal = gfilled
    
    nj_arr = np.array(
                    [len(jflags.ranges[i].ranges()) for i in range(len(obs.signal))]
                )
    obs.restrict("dets", nj_arr < nj)

     #Second Median Subtraction
    obs.signal -= np.mean(np.array(obs.signal), axis=-1)[..., None]

    # Phase to pW
    obs.signal *= obs.det_cal.phase_to_pW[..., None]

    flags.get_turnaround_flags(obs)


    return obs


def make_map(obs: 'AxisManager',
             wafer: str,
             band: str,
             res: float = 10.0 / 3600.0,
             cancel2time: bool = False,
             mask_r: float = .2,
             mask_x0: int = 0,
             mask_y0: int = 0,
             max_pix: int = 4e8,
             n_modes: int = 10,
             min_snr: int = 1,
             map_size_deg = 0.5, # at least .5 (sqr. of 1/2 deg by 1/2 deg)
             comps: str = 'TQU',
             n_modes_map: int = None,
             custom_save_name: str = None,
             do_splits: bool = True,
             do_logs: bool = True,
             ) -> "maps":

    """
    Makes a map of mars and saves it accordingly.

    NOTE: All do_logs features in this function are totally depricated and will cause errors!
    Please use the make_maps_mpi.py file to make maps and then check notebooks to save which fits are
    successfull!
    """
 
    obs.boresight = apply_pointing_model(obs)
    mask = {'shape': 'circle', 'xyr': (mask_x0, mask_y0, mask_r)}
    # This accounts for frequency scaling
    mask['xyr'] = mask["xyr"][:-1] + (
                mask["xyr"][-1] * 90.0 / float(band[1:]),
    )
    source_flags = cp.compute_source_flags(
        tod=obs, P=None, mask=mask,
        center_on="mars",
        res=np.radians(res), max_pix=max_pix, wrap=None)

    n_modes = 10

    sig_filt = cp.filter_for_sources(
                    tod=obs,
                    signal=obs.signal.copy(),
                    source_flags=source_flags,
                    n_modes=2 * n_modes,
                )


    min_snr = 1
    
    smsk = source_flags.mask()
    sig_filt_src = sig_filt.copy()
    sig_filt_src[~smsk] = np.nan
    sig_filt[smsk] = np.nan
    all_src = np.all(smsk, axis=-1)
    no_src = ~np.any(smsk, axis=-1)
    sdets = ~(all_src + no_src)
    peak_snr = np.zeros(len(sig_filt))
    with np.errstate(divide="ignore"):
        peak_snr[sdets] = np.nanmax(sig_filt_src[sdets], axis=-1) / np.nanstd(
            np.diff(sig_filt[sdets], axis=-1)
        )
    to_cut = peak_snr < min_snr
    to_cut[~sdets] = False
    cuts = RangesMatrix.from_mask(
        np.zeros_like(obs.signal, bool) + to_cut[..., None]
    )

    obs_id = obs.obs_info['obs_id']

    map_info = {'obs_id': obs_id,
                    'wafer': wafer,
                    'bandpass' : band,
                    'save_dir' : r"/global/u2/a/andrs/Products/MarsZero/i1"}


    save_dir = "mars_maps"

    if n_modes_map == None:
        n_modes_map = n_modes

    def get_save_name(custom_save_name, tag=''):
        if tag != '':
            tag = '_' + tag
        if custom_save_name != None:
            fname = "{save_dir}/{obs_id}/{obs_id}_{bandpass}_{wafer}_{map}_"+str(custom_save_name)+"_"+str(tag)+".fits"
        else:
            fname = "{save_dir}/{obs_id}/{obs_id}_{bandpass}_{wafer}_{map}"+str(tag)+".fits"
        return fname


    det_secs = np.sum((source_flags * ~cuts).get_stats()["samples"]) * np.mean(
                np.diff(obs.timestamps)
            )

    if do_logs:
    
        with open("maps.json", "r") as f:
            maps_meta = json.load(f)
    
        maps_meta[obs_id][f"{band}-{wafer}"]["detsecs"] = det_secs
    
    # Min time
    if det_secs < 600:
        print("Source has below 600 seconds of exposure.")

        if do_logs:
            maps_meta[obs_id][f"{band}-{wafer}"]["nomap"] = True
            maps_meta[obs_id][f"{band}-{wafer}"]["nomap_cause"] = f"Source below min exposure time. ({600})" #Written this way in case I later decide to change 600 to variable param

        if cancel2time:
            print("Cancelling map!")
            return None
    
    full_map= cp.make_map(
                obs.copy(),
                center_on="mars",
                signal=np.float32(obs.signal.copy()),
                res=res*coords.DEG,
                cuts=cuts, # Make UNION of this and LR scans
                source_flags=source_flags,
                comps=comps,
                n_modes=n_modes_map, 
                size=np.radians(map_size_deg),
                info=map_info,
                filename=get_save_name(custom_save_name),
            )

    
    if do_logs:
        maps_meta[obs_id][f"{band}-{wafer}"]["nomap"] = False
    def try_fit_snr(out):
        [[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(
                out["solved"].corners(corner=False)
            )
        plt_extent = [ra_min, ra_max, dec_min, dec_max]
        pixsize = 3600 * out["solved"].wcs.wcs.cdelt[1]
        buf = 20
        smoothed = gaussian_filter(out["solved"][0], sigma=60 / pixsize)
        smoothed[:buf] = np.nan
        smoothed[-1 * buf :] = np.nan
        smoothed[:, :buf] = np.nan
        smoothed[:, -1 * buf :] = np.nan
        cent = np.unravel_index(np.nanargmax(smoothed, axis=None), smoothed.shape)
        peak = 1.0
        if "T" in comps:
            peak = out["solved"][0][cent]
            snr = (
                peak
                / tod_ops.jumps.std_est(
                    np.atleast_2d(out["solved"][0].ravel()), ds=1
                )[0]
            )
            print(f"\t\tMap SNR approximately {snr}")
    return snr

    try:
        snr = try_fit_snr(full_map)
    except Exception as e:
        print(e)
        snr = 0
    
    if do_logs:
        maps_meta[obs_id][f"{band}-{wafer}"]["full"]["snr"] = snr

    if do_splits:
        left_scans = cp.make_map(
                    obs.copy(),
                    center_on="mars",
                    signal=np.float32(obs.signal.copy()),
                    res=res*coords.DEG,
                    cuts=cuts+obs.flags['right_scan'].copy(), 
                    source_flags=source_flags,
                    comps=comps,
                    n_modes=n_modes_map, 
                    size=np.radians(map_size_deg),
                    info=map_info,
                    filename=get_save_name(custom_save_name, tag='left_scans'),
                )
    
        try:
            snr = try_fit_snr(left_scans)
        except Exception as e:
            print(e)
            snr = 0
    
        if do_logs:
            maps_meta[obs_id][f"{band}-{wafer}"]["lscans"]["snr"] = snr
    
        right_scans = cp.make_map(
                    obs.copy(),
                    center_on="mars",
                    signal=np.float32(obs.signal.copy()),
                    res=res*coords.DEG,
                    cuts=cuts+obs.flags['left_scan'].copy(), 
                    source_flags=source_flags,
                    comps=comps,
                    n_modes=n_modes_map, 
                    size=np.radians(map_size_deg),
                    info=map_info,
                    filename=get_save_name(custom_save_name, tag='right_scans'),
                )
            
        try:
            snr = try_fit_snr(right_scans)
        except Exception as e:
            print("SNR Fit failed!", e)
            snr = 0
    
        if do_logs:
            maps_meta[obs_id][f"{band}-{wafer}"]["rscans"]["snr"] = snr
    
    else:
        left_scans, right_scans = None, None

    if do_logs:
        encoded = msgspec.json.encode(maps_meta)

        with open("maps.json", "wb") as f:
            f.write(encoded)
    return full_map, left_scans, right_scans

def fit_snr(out):
    """
    Computes SNR for the mapmaker stage. Not really useful since we calculate SNR again from fits but we
    mantain it in order to not break the make_maps function.
    """
    [[dec_min, ra_min], [dec_max, ra_max]] = 3600 * np.rad2deg(out.corners(corner=False))
    pixsize = 3600 * out.wcs.wcs.cdelt[1]

    # Flip if map is negative-dominated
    data = out[0]
    if np.nanmean(data) < 0:
        data = -data

    smoothed = gaussian_filter(data, sigma=60 / pixsize)

    # Mask out buffer
    buf = int(2 * (60 / pixsize))
    smoothed[:buf] = np.nan
    smoothed[-buf:] = np.nan
    smoothed[:, :buf] = np.nan
    smoothed[:, -buf:] = np.nan

    # Find peak and estimate SNR
    cent = np.unravel_index(np.nanargmax(smoothed, axis=None), smoothed.shape)
    peak = np.abs(out[0][cent])  # original value, absolute for SNR

    snr = peak / tod_ops.jumps.std_est(np.atleast_2d(out[0].ravel()), ds=1)[0]
    return snr

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument("--obs_ids", nargs="+", help="Pass a list of obs ids to run on")
args = parser.parse_args()

if args.obs_ids is not None:
    obs_list = args.obs_ids
    print(obs_list)
else:
    with open("/global/u2/a/andrs/Software/LATR_Validation/pipelines/lat_beams/MAKE_SRC_MAP/maps2make.json", "r") as f:
        maps_meta = json.load(f)

    obs_list = list(maps_meta.keys())
    
obs_list = np.array_split(obs_list, nproc)[myrank]

for obs_id in obs_list:
    for i, wafer in enumerate(["ws0", "ws1", "ws2"]):
        for band in ["f090", "f150"]:
            if obs_list[-3:][i] == '0':
                continue
            try:
                obs = load_obs_from_id(obs_id, wafer, band);
                obs = proccess_TOD(obs, restrict=True)
                make_map(obs, wafer, band, cancel2time=True, do_splits=True, do_logs=False)
            except Exception as e:
                print(e)
                continue