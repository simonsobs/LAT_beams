import glob
import logging
import os
import time

import numpy as np
from pixell import enmap
from so3g.proj import RangesMatrix
from sotodlib import mapmaking, tod_ops
from sotodlib.coords import planets as cp

from .beam_utils import estimate_cent
from .utils import log_lvl, set_tag


def make_cuts(aman, source_flags, n_modes, job, logger, cfg):
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
        job.jstate = "failed"
        return None
    return cuts


def make_map(
    aman,
    src_to_map,
    res,
    cuts,
    source_flags,
    comps,
    n_modes,
    pixsize,
    filename,
    min_det_secs,
    info,
    job,
    map_str,
    logger,
    cfg,
    det_splits={},
):
    # Get time on source
    det_secs = np.sum((source_flags * ~cuts).get_stats()["samples"]) * np.mean(
        np.diff(aman.timestamps)
    )
    logger.debug("\t%s detector seconds on source in %s mask", det_secs, map_str)
    if det_secs < min_det_secs:
        msg = f"\tNot enough time on source in {map_str} mask."
        logger.error("\t%s", msg)
        set_tag(job, "message", msg)
        job.jstate = "failed"
        return None, None

    with log_lvl(logger, logging.WARNING):
        # Full map
        out = cp.make_map(
            aman.copy(),
            thread_algo="domdir",
            center_on=src_to_map,
            res=res,
            cuts=cuts,
            source_flags=source_flags,
            comps=comps,
            filename=filename,
            n_modes=n_modes,
            info=info,
        )

        # Splits, being a litte inefficient by fitering again here 
        if len(det_splits):
            _ = cp.make_map(
                aman.copy(),
                thread_algo="domdir",
                center_on=src_to_map,
                res=res,
                cuts=cuts,
                source_flags=source_flags,
                comps=comps,
                filename=filename,
                n_modes=n_modes,
                info=info,
                data_splits=det_splits,
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
        job.jstate = "failed"
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


def get_passes(cfg):
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
    aman, sub_id, mapmaker, ipass, passinfo, P, guess, eval_prev, mapmaker_prev, logger
):
    if passinfo.downsample != 1:
        aman = mapmaking.downsample_obs(aman, passinfo.downsample)
        raise ValueError("downsampling not properly implemented currently")
    aman.signal = aman.signal.astype(np.float32)
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
    amans, passes, shape, wcs, prefix, out_dir, comm, logger, cfg, guess=None
):
    mlmap_path = ""
    rhs_path = ""
    div_path = ""
    bin_path = ""
    outmap = None
    eval_prev = None
    mapmaker_prev = None
    for ipass, passinfo in enumerate(passes):
        if comm.Get_rank() == 0:
            logger.info(
                "Starting pass %d/%d maxit %d down %d interp %s"
                % (
                    ipass + 1,
                    len(passes),
                    passinfo.maxiter,
                    passinfo.downsample,
                    passinfo.interpol,
                )
            )
            logger.flush()
        pass_prefix = os.path.join(out_dir, f"{prefix}pass{ipass+1}_")
        noise_model = mapmaking.NmatDetvecs(verbose=False)
        signal_cut = mapmaking.SignalCut(comm, dtype=np.float32)
        signal_map = mapmaking.SignalMap(
            shape,
            wcs,
            comm,
            comps=cfg.comps,
            dtype=np.float64,
            tiled=False,
            interpol=passinfo.interpol,
        )
        signals = [signal_cut, signal_map]
        mapmaker = mapmaking.MLMapmaker(
            signals, noise_model=noise_model, dtype=np.float32, verbose=True
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
        x0 = None if ipass == 0 else mapmaker.translate(mapmaker_prev, eval_prev.x_zip)

        # Solve
        t1 = time.time()
        step = None
        for step in mapmaker.solve(maxiter=passinfo.maxiter, x0=x0):
            t2 = time.time()
            dump = step.i % 10 == 0
            if comm.Get_rank() == 0:
                (logger.info if dump else logger.debug)(
                    "\tCG step %4d %15.7e %8.3f %s"
                    % (step.i, step.err, t2 - t1, "" if not dump else "(write)")
                )
                logger.flush()
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
                outmap = val
                mlmap_path = signal.write(pass_prefix, "map", val, unit="pW")

        mapmaker_prev = mapmaker
        eval_prev = mapmaker.evaluator(step.x_zip)
        logger.flush()

    return outmap, (mlmap_path, rhs_path, div_path, bin_path)
