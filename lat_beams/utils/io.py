"""
Utilities for reading and writing data to disk.
"""

import logging
from typing import Optional
from copy import deepcopy

import numpy as np
from sotodlib.core import AxisManager, Context
from sotodlib.preprocess.preprocess_util import preproc_or_load_group
from sotodlib.site_pipeline import jobdb
from sotodlib.preprocess import Pipeline

from .jobs import ErrCode, fail, set_tag
from .log import LoggerLike, log_lvl


def load_aman(
    obs_id: str,
    preprocess_cfg: dict,
    dets: dict,
    job: jobdb.Job,
    min_dets: int,
    logger: LoggerLike,
    fp_flag: bool = False,
    save: bool = False,
    debug_dets: Optional[int|str] = None
) -> Optional[AxisManager]:
    """
    Load and preprocess an observation.

    Parameters
    ----------
    obs_id : str
        The `obs_id` to load.
    preprocess_cfg : dict
        The loaded preprocess configuration.
    dets : dict
        Detector selections dictionairy.
        Check your preprocess config to see what the minimum set of selections needed here are.
    job : jobdb.Job
        The `Job` that we are loading this observation for.
        If we fail to load it then the job is marked as failed and the reason we couldn't load
        will be saved in the `message` tag.
    min_dets : int
        The minimum number of detectors allowed after preprocessing that we want.
        If fewer than `min_dets` detectors remain then the job is marked as failed.
        and `None` is returned.
    logger : LoggerLike
        Logger to log to when preprocessing.
        Note that the log level will be set to `ERROR` for preprocess.
    fp_flag : bool, default: False
        If `True` then keep only detectors with valid pointing.
    save : bool, default: False
        If `True` then try to save the preprocess result.
    debug_dets : Optional[int|str], default: None
        If `int` then will load first N dets from meta.dets.vals
        If string of comma-separated readout_ids, will load only those. 

    Returns
    -------
    aman : Optional[AxisManager]
        If we loaded and preprocessed successfully this is the loaded observation.
        If something failed this is `None`.
    """
    if debug_dets is not None:
        save = False  #Don't save preprocess if a subset of dets.
        with log_lvl(logger, logging.ERROR):
            ctx = Context(preprocess_cfg["context_file"])
            all_dets = ctx.get_det_info(obs_id, dets=dets)
        if isinstance(debug_dets, int):
            if debug_dets <= 0:
                  raise ValueError ("Non-positive number of debug dets passed")
            detlist = all_dets[:min(len(all_dets), debug_dets)]
        elif isinstance(debug_dets, str):
            detlist = all_dets[np.isin(np.asarray(all_dets["readout_id"]), debug_dets.split(","))]
        else:
             raise ValueError("Invalid debug dets") # or something
        if len(detlist) == 0:
            raise ValueError("No debug dets in obs")
        min_dets = int(len(detlist)//2)
        dets = deepcopy(dets)
        dets["readout_id"] = detlist["readout_id"]

    try:
        with log_lvl(logger, logging.ERROR):
            aman, _, _, err = preproc_or_load_group(
                obs_id,
                preprocess_cfg,
                dets=dets,
                save_archive=save,
                save_proc_aman=save,
                overwrite=True,
                logger=logger,
            )
    except Exception as e:
        msg = f"Failed to load or preprocess with error {e}"
        fail(job, ErrCode.PREPROC, msg, logger)
        return None
    if aman is None:
        msg = f"Preprocess failed with error {err}"
        fail(job, ErrCode.PREPROC, msg, logger)
        return None

    if fp_flag:
        aman.restrict(
            "dets",
            np.isfinite(aman.focal_plane.xi)
            * np.isfinite(aman.focal_plane.eta)
            * np.isfinite(aman.focal_plane.gamma),
        )

    if aman.dets.count < min_dets:
        msg = f"Only {aman.dets.count} dets!"
        fail(job, ErrCode.MIN_DETS, msg, logger)
        return None
    return aman
