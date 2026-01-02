import argparse

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
    parser.add_argument(
        "--profile", action="store_true", help="Run a profile (only for fit_pointing)"
    )
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    return args, cfg
