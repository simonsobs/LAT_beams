import argparse
import os
import time

import numpy as np
import yaml


def get_args_cfg():
    # Only the config is necessary; the rest are just for ease of use.
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="Path to the config file")
    parser.add_argument(
        "--plot_only",
        "-p",
        action="store_true",
        help="Don't do any fitting or mamaking, just plot TODs or existing maps",
    )
    # Control which obs are used
    parser.add_argument("--obs_ids", nargs="+", help="Pass a list of obs ids to run on")
    parser.add_argument(
        "--lookback",
        "-l",
        type=float,
        help="Amount of time to lookback for query, overides start time from config",
    )
    # JobDB stuff
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Overwrite an existing fit"
    )
    parser.add_argument(
        "--retry_failed", "-r", action="store_true", help="Retry failed jobs"
    )
    parser.add_argument(
        "--job_memory",
        "-m",
        type=float,
        help="If job was run within this many hours of this script starting then don't rerun even if overwrite or retry_failed is passed",
    )
    parser.add_argument(
        "--job_memory_buffer",
        "-mb",
        default=0,
        type=float,
        help="If job was run within this many minutes of this script starting then rerun even if job_memory is passed",
    )
    # Shared useful stuff
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run a profile (only for fit_pointing and make_source_mask)",
    )
    # fit_pointing exclusive args
    parser.add_argument(
        "--forced_ws",
        "-ws",
        nargs="+",
        help="Force these wafer slots into the fit (only for fit_pointing)",
    )
    parser.add_argument(
        "--parallel_factor",
        "-f",
        default=4,
        type=int,
        help="Per-obs parallelization factor (only for fit_pointing)",
    )
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    return args, cfg


def setup_cfg(args, cfg, replace={}, apply_ds=False):
    # TODO: Make a default config yaml file and only do modifications here

    # What data to use
    cfg["tel"] = cfg.get("tel", "lat")
    cfg["forced_ws"] = args.forced_ws if args.forced_ws is not None else []
    if cfg.get("try_all", False):
        cfg["forced_ws"] = ["ws0", "ws1", "ws2"]
    cfg["fit_source_list"] = cfg.get("fit_source_list", ["mars", "saturn"])
    cfg["map_source_list"] = cfg.get("map_source_list", ["mars", "saturn"])
    cfg["start_time"] = cfg.get("start_time", 0)
    if args.lookback is not None:
        cfg["start_time"] = time.time() - 3600 * args.lookback
    cfg["stop_time"] = cfg.get("stop_time", 20000000000)
    cfg["max_dur"] = cfg.get("max_dur", 2)

    # Get paths to stuff
    cfg["preprocess_cfg"] = cfg.get("preprocess_cfg", None)
    cfg["ctx_path"] = cfg.get(
        "ctx_path",
        f"/global/cfs/cdirs/sobs/metadata/{cfg['tel']}/contexts/smurf_detcal_local.yaml",
    )
    cfg["nominal_path"] = os.path.expanduser(
        cfg.get("nominal_path", f"~/data/pointing/{cfg['tel']}/nominal/focal_plane.h5")
    )
    cfg["root_dir"] = os.path.expanduser(cfg.get("root_dir", "~"))
    cfg["append"] = cfg.get("append", "")

    # Source masking and projection settings
    cfg["res"] = cfg.get("res", (10 / 3600.0) * np.pi / 180.0)
    cfg["pointing_mask"] = cfg.get(
        "pointing_mask", {"shape": "circle", "xyr": (0, 0, 0.75)}
    )
    cfg["map_mask_size"] = cfg.get("map_mask_size", 0.1)
    cfg["search_mask"] = cfg.get("search_mask", {"shape": "circle", "xyr": (0, 0, 0.5)})

    # Cuts and processing info
    cfg["ds"] = cfg.get("ds", 5)
    ds = cfg["ds"] if apply_ds else 1
    cfg["hp_fc"] = cfg.get("hp_fc", 4)
    cfg["lp_fc"] = cfg.get("lp_fc", 30)
    cfg["n_med"] = cfg.get("n_med", 5)
    cfg["n_std"] = cfg.get("n_std", 10)
    cfg["min_samps"] = cfg.get("min_samps", 1000) / ds
    cfg["block_size"] = cfg.get("block_size", 5000) / ds
    cfg["min_dets"] = cfg.get("min_dets", 30)
    cfg["trim_samps"] = cfg.get("time_samps", 200) // ds
    cfg["min_hits"] = cfg.get("min_hits", 1)
    cfg["fwhm_tol_pointing"] = cfg.get("fwhm_tol_pointing", 0.2)
    cfg["fwhm_tol_map"] = cfg.get("fwhm_tol_map", 3)
    cfg["max_chisq"] = cfg.get("max_chisq", 2.5)
    cfg["min_det_secs"] = cfg.get("min_det_secs", 600)
    cfg["min_snr"] = cfg.get("min_snr", 5)
    cfg["relcal_range"] = cfg.get("relcal_range", [0.3, 2])
    cfg["min_sigma"] = cfg.get("min_sigma", 3)
    cfg["ufm_rad"] = cfg.get("ufm_rad", 0.01)
    cfg["miscenter_thresh"] = cfg.get("miscenter_thresh", 5)

    # Geometry
    cfg["extent"] = cfg.get("extent", 600)
    cfg["snr_extent"] = cfg.get("snr_extent", 500)
    cfg["buf"] = cfg.get("buf", 30)
    cfg["buf_cropped"] = cfg.get("buf_cropped", 5)
    cfg["smooth_kern"] = cfg.get("smooth_kern", 60)

    # Mapping
    cfg["n_modes"] = cfg.get("n_modes", 10)
    cfg["del_map"] = cfg.get("del_map", True)
    cfg["cgiters"] = cfg.get("cgiters", 30)
    cfg["mlpass"] = cfg.get("mlpass", 3)
    cfg["comps"] = cfg.get("comps", "TQU")

    # Map fits
    cfg["gauss_multipole"] = cfg.get("gauss_multipole", True)
    cfg["bessel_beam"] = cfg.get("bessel_beam", True)
    cfg["n_multipoles"] = cfg.get("n_multipoles", 3)
    cfg["n_bessel"] = cfg.get("n_bessel", 10)
    cfg["force_bessel_cent"] = cfg.get("force_bessel_cent", False)
    cfg["bessel_wing"] = cfg.get("bessel_wing", False)
    cfg["sym_gauss"] = cfg.get("sym_gauss", True)

    # Hardware info
    cfg["nominal_fwhm"] = cfg.get(
        "nominal_fwhm", {"f090": 2.0, "f150": 1.3, "f220": 0.95, "f280": 0.83}
    )
    cfg["aperature"] = cfg.get("aperature", 6)
    cfg["corr_primary"] = cfg.get("corr_primary", 280)
    cfg["eps_primary"] = cfg.get("eps_primary", 17)

    # TOD fits
    cfg["fit_pars"] = cfg.get("fit_pars", {})
    cfg["pad"] = cfg.get("pad", True)
    cfg["src_msk"] = cfg.get("src_msk", True)
    cfg["blind_search"] = cfg.get("blind_search", True)

    # Misc
    cfg["log_thresh"] = cfg.get("log_thresh", 1e-3)
    cfg["pointing_type"] = cfg.get("pointing_type", "pointing_model")
    cfg["epochs"] = cfg.get("epochs", [(0, 2e10)])
    cfg["split_by"] = cfg.get(
        "split_by", ["band", "tube_slot+band", "source+band", "source+tube_slot+band"]
    )
    cfg["lmax"] = cfg.get("lmax", 20000)
    cfg["r_step"] = cfg.get("r_step", 1)

    # Rename for our scope
    for o, n in replace.items():
        if o not in cfg:
            continue
        cfg[n] = cfg[o]
        del cfg[o]

    cfg_str = yaml.dump(cfg)

    return argparse.Namespace(**cfg), cfg_str


def setup_paths(root_dir, project, tel, append=""):
    plot_dir = os.path.join(root_dir, "plots", project, tel, append)
    data_dir = os.path.join(root_dir, "data", project, tel, append)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    return plot_dir, data_dir
