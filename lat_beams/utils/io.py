import logging

import numpy as np
from sotodlib.preprocess.preprocess_util import preproc_or_load_group


def load_aman(obs_id, preprocess_cfg, dets, job, min_dets, L, fp_flag=False):
    try:
        with log_lvl(L, logging.ERROR):
            err, _, _, aman = preproc_or_load_group(
                obs_id,
                preprocess_cfg,
                dets=dets,
                save_archive=False,
                overwrite=True,
                logger=L,
            )
    except:
        msg = "Failed to load or preprocess!"
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        return None
    if aman is None:
        msg = f"Preprocess failed with error {err}"
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
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
        L.error(f"\t{msg}")
        set_tag(job, "message", msg)
        job.jstate = "failed"
        return None
    return aman
