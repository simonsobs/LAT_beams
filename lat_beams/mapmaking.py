"""
Functions for producing source maps.
"""

import glob
import logging
import os
import time
from argparse import Namespace
from logging import Logger
from typing import Optional, cast

import numpy as np
import sqlalchemy as sqy
from astropy.wcs import WCS
from mpi4py import MPI
from pixell import bunch, enmap
from so3g.proj import RangesMatrix
from sotodlib import mapmaking, tod_ops
from sotodlib.coords import planets as cp
from sotodlib.coords import pmat
from sotodlib.core import AxisManager
from sotodlib.site_pipeline import jobdb

from .beam_utils import estimate_cent
from .utils import log_lvl, set_tag


def make_cuts(
    aman: AxisManager,
    source_flags: RangesMatrix,
    n_modes: int,
    job: jobdb.Job,
    logger: Logger,
    cfg: Namespace,
) -> Optional[RangesMatrix]:
    """
    Compute cuts on a source TOD before mapping.
    This filters for the source and then calculates the peak SNR of each detector
    by dividing the max of the region in `source_flags` by the standard deviation of the region outside of `source_flags`.
    Any detector with an SNR less than `cfg.min_snr` is cut.

    Parameters
    ----------
    aman : AxisManager
        The loaded data ready to be mapped.
    source_flags : RangesMatrix
        RangesMatrix with all samples within some radius of the source flagged.
    n_modes : int
        The number of modes to use when filtering.
    job : jobdb.Job
        The job associated with making this map.
    logger : Logger
        The logger to log to.
    cfg : Namespace
        The loaded configuration.
        See `lat_beams.utils.config` for details.

    Returns
    -------
    cuts : Optional[RangesMatrix]
        The calculated cuts.
        If the number of uncut detectors is less than `cfg.min_dets` then `None` is returned.
    """
    sig_filt = cp.filter_for_sources(
        tod=aman,
        signal=aman.signal.copy(),
        source_flags=source_flags,
        n_modes=n_modes,
    )
    smsk = source_flags.mask()
    sig_filt_src = sig_filt.copy()
    sig_filt_src[~smsk] = np.nan
    sig_filt[smsk] = np.nan
    all_src = np.all(smsk, axis=-1)
    no_src = ~np.any(smsk, axis=-1)
    sdets = ~(all_src + no_src)
    peak_snr = np.zeros(len(sig_filt))
    if np.sum(sdets) > 0:
        with np.errstate(divide="ignore"):
            peak_snr[sdets] = np.nanmax(sig_filt_src[sdets], axis=-1) / np.nanstd(
                np.diff(sig_filt[sdets], axis=-1)
            )
    to_cut = peak_snr < cfg.min_snr  # + ~np.isfinite(peak_snr)
    to_cut[~sdets] = False
    cuts = RangesMatrix.from_mask(np.zeros_like(aman.signal, bool) + to_cut[..., None])
    logger.debug("\tCutting %s detectors from map", np.sum(to_cut))
    if np.sum(~to_cut) < cfg.min_dets:
        msg = f"Not enough detectors after source flag cuts!"
        logger.error("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = cast(sqy.Column[str], jobdb.JState.failed)

        return None
    return cuts


def make_map(
    aman: AxisManager,
    src_to_map: str,
    res: float,
    cuts: RangesMatrix,
    source_flags: RangesMatrix,
    comps: str,
    n_modes: int,
    pixsize: float,  # TODO: This doesn't need to exist
    filename: str,
    min_det_secs: float,
    info: dict[str, str],
    job: jobdb.Job,
    map_str: str,
    logger: Logger,
    cfg: Namespace,
) -> tuple[Optional[dict], Optional[tuple[int, int]]]:
    """
    Make a filter-bin map of a source and estimate the center.
    The map will be in source-scan coordinates and uses `domdir` threading.
    If the mapping fails the associated job will have its message updated with an explanation.
    If mapping succeeds then paths to the map's files will be added to the job.

    Once mapped the center of the map and its SNR are estimated.
    If the SNR is below `cfg.min_snr` and `cfg.del_map` is set then the files
    associated with the map are deleted.

    Parameters
    ----------
    aman : AxisManager
        The loaded data ready to be mapped.
    src_to_map : str
        The name of the source to map.
    res : float
        The desired map resolution in radians.
    cuts : RangesMatrix
        The output of `make_cuts`.
    source_flags : RangesMatrix
        RangesMatrix with all samples within some radius of the source flagged.
    comps : str
        The maps to compute, should be `T` or `TQU`.
    n_modes : int
        The number of modes to use when filtering.
    pixsize : float
        `res` in arcseconds
    filename : str
        The pattern for the output map filename.
        See `sotodlib.coords.planets.make_map` for details.
    min_det_secs : float
        The minimum number of detector seconds.
        If we are below this a map is not made.
    info : dict[str, str]
        The information used to fill in `filename`.
        See `sotodlib.coords.planets.make_map` for details.
    job : jobdb.Job
        The job associated with making this map.
    map_str : str
        A short string to describe the map in the logs and job (ie. "initial").
    logger : Logger
        The logger to log to.
    cfg : Namespace
        The loaded configuration.
        See `lat_beams.utils.config` for details.

    Returns
    -------
    outmap : Optional[dict]
        The output of `sotodlib.coords.planets.make_map`.
        `None` is returned if we are below `min_det_secs` or `cfg.min_snr`.
    cent : Optional[tuple[int, int]]]
        The estimated center of the map.
        `None` is returned if we are below `min_det_secs` or `cfg.min_snr`.
    """
    # Get time on source
    sf_uncut = source_flags * ~cuts
    if sf_uncut is None:
        raise ValueError("RangesMatrix somehow became none...")
    det_secs = np.sum(sf_uncut.get_stats()["samples"]) * np.mean(
        np.diff(np.array(aman.timestamps))
    )
    logger.debug("\t%s detector seconds on source in %s mask", det_secs, map_str)
    if det_secs < min_det_secs:
        msg = f"\tNot enough time on source in {map_str} mask."
        logger.error("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = cast(sqy.Column[str], jobdb.JState.failed)
        return None, None

    # Initial map
    with log_lvl(logger, logging.WARNING):
        out = cp.make_map(
            aman.copy(),
            thread_algo="domdir",  # type: ignore
            center_on=src_to_map,
            res=res,
            cuts=cuts,
            source_flags=source_flags,
            comps=comps,
            filename=filename,
            n_modes=n_modes,
            info=info,
        )

    # Smooth and find the center
    cent = estimate_cent(out["solved"][0], cfg.smooth_kern / pixsize, cfg.buf)

    # Estimate SNR
    peak = out["solved"][0][cent]
    snr = peak / tod_ops.jumps.std_est(np.atleast_2d(out["solved"][0].ravel()), ds=1)[0]
    ndets = np.sum(np.all(~cuts.mask(), axis=-1))
    logger.debug("\t%s map SNR approximately %s", map_str.title(), snr)
    if snr < cfg.min_snr * np.sqrt(ndets) / 2:
        msg = f"{map_str.title()} map SNR too low."
        logger.error("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = cast(sqy.Column[str], jobdb.JState.failed)
        if cfg.del_map and filename is not None:
            logger.debug("\tDeleting map files")
            glob_path = os.path.splitext(filename)[0] + "*.*"
            flist = glob.glob(glob_path)
            for fname in flist:
                if os.path.isfile(fname):
                    os.remove(fname)
            for name in ["binned", "detweights", "solved", "weights"]:
                set_tag(job, name, "")
        return None, None
    return out, cent


def get_passes(cfg: Namespace) -> list[bunch.Bunch]:
    """
    Setup passes for making an ML mapmaker.
    The output will have `cfg.mlpass` elements with all of them
    having a downsampling factor of 1.
    The last pass will hade bilinear interpolations and the rest will be nearest neighbor.
    The i'th pass will have `max(1, cfg.cgiters//2, cfg.cgiters//(i + 1))` CG iters.

    Parameters
    ----------
    cfg : Namespace
        The loaded configuration.
        See `lat_beams.utils.config` for details.

    Returns
    -------
    passes : list[bunch.Bunch]
        The passes for the sotodlib ML mapmaker.
        See this function's docstring for details on the contents.
    """
    passes = []
    if cfg.mlpass > 0:
        dsstr = "1"
        maxiter = str(cfg.cgiters)
        interpol = "bilinear"
        for i in range(1, cfg.mlpass):
            interpol = "nearest," + interpol
            maxiter = f"{max(1, cfg.cgiters//2, cfg.cgiters//(i + 1))}," + maxiter
        passes = mapmaking.setup_passes(
            downsample=dsstr, maxiter=maxiter, interpol=interpol
        )

    return passes


def add_obs_to_mapmaker(
    aman: AxisManager,
    sub_id: str,
    mapmaker: mapmaking.MLMapmaker,
    ipass: int,
    passinfo: bunch.Bunch,
    P: pmat.P,
    guess: Optional[enmap.ndmap],
    eval_prev: Optional[mapmaking.MLEvaluator],
    mapmaker_prev: Optional[mapmaking.MLMapmaker],
    logger: Logger,
):
    """
    Add an observation to the MLMapmaker.
    This makes sure that the correct weather and site are included and will use
    an input map or a previous iteration of the mapmaker to estimate the signal
    when computing the noise model if they are provided.

    Parameters
    ----------
    aman : AxisManager
        The TOD after preprocessing.
        The cut samples should be in the `glitch` flag.
    sub_id : str
        The `sub_id` of the TOD.
        Should have format `{obs_id}:{ws}:{band}`.
    mapmaker : mapmaking.MLMapmaker
        The mapmaker instance to add the observation to.
    ipass : int
        The pass number.
    passinfo : bunch.Bunch
        The pass we are on. See `setup_passes` for details.
    P : pmat.P
        The `sotodlib` projectioe matrix.
    guess : Optional[enmap.ndmap]
        A guess at what the map is.
        If this is not `None` and `ipass == 0` then this is
        used to estimate the signal when building the noise model.
    eval_prev : Optional[mapmakin.MLEvaluator]
        Evaluator for the previous pass of the mapmaker.
        If `ipass > 0` and both this and `mapmaker_prev` are not `None`
        then they ary used to estimate the signal when building the noise model.
    mapmaker_prev : Optional[mapmakin.MLMapmaker]
        Mapmaker instance for the previous pass of the mapmaker.
        If `ipass > 0` and both this and `eval_prev` are not `None`
        then they ary used to estimate the signal when building the noise model.
    logger : Logger
        The logger to log to.
    """
    if passinfo.downsample != 1:
        aman = mapmaking.downsample_obs(aman, passinfo.downsample)
        raise ValueError("downsampling not properly implemented currently")
    aman.signal = np.array(aman.signal).astype(np.float32)
    if "weather" not in aman:
        aman.wrap("weather", np.full(1, "typical"))
    if "site" not in aman:
        aman.wrap("site", np.full(1, "so_lat"))

    # Estimate noise
    if ipass == 0 and guess is not None:
        signal_estimate = P.from_map(guess)
    elif eval_prev is not None and mapmaker_prev is not None:
        signal_estimate = eval_prev.evaluate(mapmaker_prev.data[len(mapmaker.data)])
    else:
        signal_estimate = np.zeros_like(aman.signal)
    if signal_estimate is not None:
        signal_estimate = mapmaking.resample.resample_fft_simple(
            signal_estimate, aman.samps.count
        )
    with log_lvl(logger, logging.WARNING):
        mapmaker.add_obs(sub_id, aman, signal_estimate=signal_estimate, pmap=P)
    del signal_estimate
    del aman


def make_ml_map(
    amans: dict[str, tuple[AxisManager, pmat.P]],
    passes: list[bunch.Bunch],
    shape: tuple[int, int],
    wcs: WCS,
    prefix: str,
    out_dir: str,
    comm: MPI.Comm,
    logger: Logger,
    cfg: Namespace,
    guess: Optional[enmap.ndmap] = None,
) -> tuple[Optional[enmap.ndmap], tuple[str, str, str, str]]:
    """
    Make a multipass ML source map using the sotodlib ML mapmaker.
    May be worth adding a `sogma` option down the line.

    Parameters
    ----------
    amans : dict[str, tuple[AxisManager, pmat.P]]
        The AxisManagers to map.
        Each entry should map a sub_id (with format `{obs_id}:{ws}:{band}`)
        to a tuple consisting of an AxisManager that is preprocessed and ready to map
        and the corresponding projection matrix.
    passes : list[bunch.Bunch]
        The output of `setup_passes`.
    shape : tuple[int, int]
        The desired shape of the output map.
    wcs : WCS
        The WCS to be used for the output map.
    prefix : str
        The prefix to be preprended to the map filenames.
    out_dir : str
        The directory to save maps to.
    comm : MPI.Comm
        The MPI communicator to use when mapmaking.
        All processes must have a non-zero sized `amans`.
    logger : Logger
        The logger to log to.
    cfg : Namespace
        The loaded configuration.
        See `lat_beams.utils.config` for details.
    guess : Optional[enmap.ndmap], default: None
        A map to use to estimate the signal when constructing
        the noise model in the first pass of mapmaking.
        Pass `None` if you don't have a good starting guess.

    Returns
    -------
    outmap : Optional[enmap.ndmap]
        The signal map from the final pass.
        Will be `None` if no passes are run.
    paths : tuple[str, str, str, str]
        The paths to the signal, rhs, div, and bin maps from the final pass.
    """
    mlmap_path = ""
    rhs_path = ""
    div_path = ""
    bin_path = ""
    outmap = None
    eval_prev = None
    mapmaker_prev = None
    for ipass, passinfo in enumerate(passes):
        if comm.Get_rank() == 0:
            logger.debug(
                "Starting pass %d/%d maxit %d down %d interp %s"
                % (
                    ipass + 1,
                    len(passes),
                    passinfo.maxiter,
                    passinfo.downsample,
                    passinfo.interpol,
                )
            )
        pass_prefix = os.path.join(out_dir, f"{prefix}pass{ipass+1}_")
        noise_model = mapmaking.NmatDetvecs(verbose=False)
        signal_cut = mapmaking.SignalCut(comm, dtype=np.float32)
        signal_map = mapmaking.SignalMap(
            shape,
            wcs,
            comm,
            comps=cfg.comps,
            dtype=np.float64,  # type: ignore
            tiled=False,
            interpol=passinfo.interpol,
        )
        signals = [signal_cut, signal_map]
        mapmaker = mapmaking.MLMapmaker(
            signals, noise_model=None, dtype=np.float32, verbose=True
        )

        for sub_id, (aman, P) in amans.items():
            P.interpol = passinfo.interpol
            add_obs_to_mapmaker(
                aman.copy(),
                sub_id,
                mapmaker,
                ipass,
                passinfo,
                P,
                guess,
                eval_prev,
                mapmaker_prev,
                logger,
            )

        # Write the starting maps
        mapmaker.prepare()
        rhs_path = signal_map.write(out_dir + "/", "rhs", signal_map.rhs, unit="pW^-1")
        div_path = signal_map.write(out_dir + "/", "div", signal_map.div, unit="pW^-2")
        bin_path = signal_map.write(
            out_dir + "/",
            "bin",
            enmap.map_mul(signal_map.idiv, signal_map.rhs),
            unit="pW",
        )
        if comm.Get_rank() == 0:
            logger.debug("\tWrote rhs, div, bin")

        # Set up initial condition
        x0 = (
            None
            if ipass == 0 or eval_prev is None or mapmaker_prev is None
            else mapmaker.translate(mapmaker_prev, eval_prev.x_zip)
        )

        # Solve
        t1 = time.time()
        step = None
        for step in mapmaker.solve(maxiter=passinfo.maxiter, x0=x0):
            t2 = time.time()
            dump = step.i % 10 == 0
            if comm.Get_rank() == 0:
                (logger.debug if dump else logger.debug)(
                    "\tCG step %4d %15.7e %8.3f %s"
                    % (step.i, step.err, t2 - t1, "" if not dump else "(write)")
                )
            if dump:
                for signal, val in zip(signals, step.x):
                    if signal.output:
                        mlmap_path = signal.write(pass_prefix, "map%04d" % step.i, val)
            t1 = time.time()

        logger.debug("Done with ML map")
        if step is None:
            raise ValueError("Mapmaker ran 0 steps!")
        for signal, val in zip(signals, step.x):
            if signal.output:
                outmap = cast(enmap.ndmap, val)
                mlmap_path = signal.write(pass_prefix, "map", val, unit="pW")

        mapmaker_prev = mapmaker
        eval_prev = mapmaker.evaluator(step.x_zip)

    mlmap_path = "" if mlmap_path is None else mlmap_path
    rhs_path = "" if rhs_path is None else rhs_path
    div_path = "" if div_path is None else div_path
    bin_path = "" if bin_path is None else bin_path

    return outmap, (mlmap_path, rhs_path, div_path, bin_path)
