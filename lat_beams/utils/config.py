"""
Module for handling all configuration of scripts.

## Config File Fields

## General pipeline settings

??? info "cfg.root_dir"
    Root directory for output products.

??? info "cfg.tel"
    Telescope identifier used when constructing the output directory tree.

??? info "cfg.pointing_type"
    Pointing type used to distinguish beam-analysis products.
    ie. pointing_model, raw, etc.

??? info "cfg.append"
    Optional suffix appended to the output directory name.

??? info "cfg.test_append"
    Optional suffix used when naming directories and jobdb. Mostly used for testing and one-offs.

??? info "cfg.copy_fits_test"
    If True when when `test_append` is not "" then make a copy of the beam fits.
    This will overwrite an existing file.

??? info "cfg.single_det"
    Whether the analysis is operating on single-detector data.
    When True,`_single_det` is appended to the output directory name.

??? info "cfg.ctx_path"
    Path to the sotodlib context.

??? info "cfg.preprocess_cfg"
    Path to the preprocessing configuration used to load and preprocess
    the data before fitting or mapmaking.

??? info "cfg.source_list"
    List of source names to process.
    Note that this has the following aliases:

    * `map_source_list`: used in `make_source_map`.
    * `fit_source_list`: used in `fit_source_map`.

    This distiction is because there are sources we want to map
    that we do not want to fit in the standard pipeline (ie. TauA).

??? info "cfg.start_time"
    Lower bound on the observation timestamp used when selecting jobs.
    If `args.lookback` is passed then this becomes the current time minus the lookback.

??? info "cfg.stop_time"
    Upper bound on the observation timestamp used when selecting jobs.
    If `args.lookback` is passed then this becomes the current time.

??? info "cfg.fwhm_tol"
    Fractional tolerance between the measured radial FWHM and the
    nominal band FWHM. A fit is rejected when: `abs(1 - data_fwhm / nominal_fwhm) > fwhm_tol`
    This is aliased to `fwhm_tol_map` for map fitting
    and `fwhm_tol_pointing` for pointing fits..

??? info "cfg.nominal_fwhm"
    Mapping from observing band to nominal beam FWHM. These values are
    used for the initial Gaussian fit, FWHM quality cuts, noise
    estimation, and stacking diagnostics. These should be in arcmins
    and should be a dict where each key is a bandname (ie. "f090").

??? info "cfg.min_samps"
    Minimum number of source-flagged samples required for a pointing fit
    to proceed. It is also used when deciding which detectors have enough
    source-flagged samples to remain in the fit.

??? info "cfg.min_dets"
    Minimum number of detectors required after cuts.

### Mapmaking

??? info "cfg.extent"
    Angular extent of the map region used for beam fitting and stacking.
    Also used when generating fitted-model and residual diagnostic plots.
    Should be in arcseconds.

??? info "cfg.res"
    Target pixel resolution used when constructing the common
    tangent-plane WCS for beam maps and high-resolution profile
    calculations. Should be in radians.

??? info "cfg.mask_size"
    Angular size of the mask used during mapmaking and beam-model fitting.
    When `cfg.apply_fscale` is enabled, the fitting stage scales this value
    according to the observing frequency before converting it to radians.
    This is aliased by `map_mask_size`.

??? info "cfg.apply_fscale"
    Whether beam-mask is adjusted according to observing frequency.
    When enabled, the mask is scaled by `90 / frequency_GHz`.

??? info "cfg.aperature"
    Aperture size used by the Bessel beam model. The fitting stage
    converts this value to a `Quantity` in meters before passing it
    to the Bessel fitting routine.

??? info "cfg.buf"
    Buffer used when estimating the beam center on the original map.
    In units of pixels.

??? info "cfg.buf_cropped"
    Buffer used when estimating the beam center after the map has been
    cropped, again in pixels.

??? info "cfg.smooth_kern"
    Angular smoothing scale used when estimating the beam center.

??? info "cfg.snr_extent"
    Angular extent around the estimated beam center excluded when
    estimating map noise for the initial SNR calculation.

??? info "cfg.extent_highres"
    Angular extent of the high-resolution map used for calculating the
    final profile and covariance.

??? info "cfg.pixsize_highres"
    Pixel size for the high-resolution final profile.

??? info "cfg.search_mask"
    Mask definition used to search for the source in an initial map.

??? info "cfg.del_map"
    If True delete maps that don't pass cuts in mapmaking.

??? info "cfg.cgiters_single"
    Number of CG iters used when making a single obs ML map.

??? info "cfg.cgiters_full"
    Number of CG iters used when making a full ML map.

??? info "cfg.mlpass"
    Number of passes to run the ML mapmaker for.

??? info "cfg.comps"
    Which comps to mapmake. Should be "T" or "TQU".

??? info "cfg.force_zero_cent"
    Whether map-fitting workflows force the beam center to zero
    instead of fitting for a recenter.

??? info "cfg.n_modes"
    Number of modes to remove when mapmaking.

??? info "cfg.relcal_range"
    Allowed relative calibration range.

??? info "cfg.min_det_secs"
    Minimum number of detector seconds in the source mask needed to mapmake.

### Pointing fits

??? info "cfg.forced_ws"
    Wafer-slot identifiers that are forced to be processed even when they
    are not present in the observation's source tags. The pointing-fit
    script uses these values when constructing the set of wafer slots
    eligible for fitting.


??? info "cfg.try_all"
    If True then try all wafer slots.
    This will override forced_ws.

??? info "cfg.max_dur"
    Maximum allowed observation duration, in hours, when selecting
    pointing-fit observations from the observation database.

??? info "cfg.nominal_path"
    Path to the nominal focal-plane pointing model. The pointing-fit script
    loads this HDF5 file and uses it to obtain nominal detector positions,
    calculate the UFM radius, and provide nominal pointing information for
    source masking.

??? info "cfg.pointing_mask"
    Mask definition used when generating source flags with the centered
    source flagger. The pointing-fit script passes this configuration to
    `sotodlib.coords.planets.compute_source_flags` to identify samples
    containing the astronomical source.

??? info "cfg.ds"
    Downsampling factor applied to the TOD before filtering and fitting.

??? info "cfg.hp_fc"
    High-pass filter cutoff frequency used when filtering the TOD before
    the pointing fit. It is also passed to fit_tod_pointing as part of the
    filter configuration.

??? info "cfg.lp_fc"
    Low-pass filter cutoff frequency used when filtering the TOD before
    the pointing fit. It is also passed to fit_tod_pointing as part of the
    filter configuration.

??? info "cfg.n_med"
    Multiplier applied to the median detector noise when rejecting
    unusually noisy detectors within each frequency band.

??? info "cfg.n_std"
    Number of standard deviations used by source-flagging logic. It
    controls the threshold in the blind and SVD source flaggers.

??? info "cfg.block_size"
    Time/sample block size used by source-flagging logic. It controls the
    minimum extent and separation of flagged source regions and the
    buffering applied to source flags.

??? info "cfg.trim_samps"
    Number of samples trimmed from each edge of the downsampled TOD to
    avoid Fourier-filter ringing.

??? info "cfg.min_hits"
    Minimum number of source hits required for an individual detector fit
    to be considered acceptable.

??? info "cfg.high_hits"
    Higher hit-count threshold used when identifying a sufficiently
    well-sampled set of detectors for estimating the center of the array.

??? info "cfg.max_chisq"
    Maximum allowed reduced chi-squared for an individual pointing fit.
    Detectors with reduced chi-squared above this threshold are marked as
    bad.

??? info "cfg.min_R2"
    Minimum acceptable R2 value for a pointing fit. Fits below this
    threshold are excluded from the focal-plane diagnostic plot and
    treated as bad fits.

??? info "cfg.svd_modes"
    Number of SVD modes used by the SVD-based source flagger. When source
    filtering is enabled, the same value is also passed to
    cp.filter_for_sources.

??? info "cfg.svd_iters"
    Number of iterations used by the SVD-based source flagger.

??? info "cfg.iter_svd_sub"
    Whether the SVD-derived common mode is subtracted from the TOD after
    SVD source identification.

??? info "cfg.filter_for_sources"
    Whether the pointing-fit TOD is additionally filtered using the source
    flags and SVD modes before fitting.

??? info "cfg.source_flag_exp"
    Expression defining how source flags are combined. The default
    expression is `(svd + blind) * cent`. The pointing-fit script
    evaluates this expression using source flags supplied by the SVD,
    blind, and centered source flaggers.


??? info "cfg.fit_pars"
    Additional keyword arguments passed directly to fit_tod_pointing.

??? info "cfg.pad"
    Whether the pointing-fit result is padded with detectors that were
    present in the observation metadata but did not produce a fitted
    result. When enabled, missing detectors are added with NaN values for
    floating-point fit fields.

??? info "cfg.src_msk"
    Whether samples identified by the source-flag expression are used to
    restrict the TOD to the source-crossing region and remove detectors
    with insufficient source-flagged samples.

### Beam-fit configuration

??? info "cfg.sym_gauss"
    Whether the Gaussian beam fit is constrained to be symmetric.

??? info "cfg.min_snr"
    Minimum SNR required for an individual beam map to proceed through the fitting stage.

??? info "cfg.bessel_beam"
    Whether to fit the Bessel-based beam model after the Gaussian fit.

??? info "cfg.min_sigma"
    Minimum allowed beam-model width used when validating and processing
    fitted Gaussian and Bessel model parameters.
    Set to a negetive value to use the whole map.

??? info "cfg.n_bessel"
    Number of Bessel terms/components used by the Bessel beam fit.

??? info "cfg.n_multipoles"
    Number of multipoles included in the Bessel beam model. This also
    controls the number of non-axisymmetric beam modes shown in fitting diagnostics.

??? info "cfg.skip_multipoles"
    Multipoles excluded from the Bessel beam fit.

??? info "cfg.bessel_wing_n_sigma"
    Controls the extent of the Bessel-model wing relative to the fitted
    beam. When frequency scaling is enabled, the fitting stage scales
    this value by the same factor used for the beam mask.

??? info "cfg.gauss_multipole"
    If True fit for the multipole expansion of the Gauss fit.

??? info "cfg.corr_primary"
    Error correlation scale of the mirror in mm.

??? info "info" "cfg.eps_primary"
    RMS error of the mirror in um-rms.

### Stacking quality cuts

??? info "cfg.min_stack_snr"
    Minimum fitted beam SNR required for an observation to contribute to a stack.

??? info "cfg.max_pwv"
    Maximum allowed PWV/elevation-corrected atmospheric loading.
    Fits are retained only when: `pwv / sin(elevation) <= max_pwv`.

??? info "cfg.max_cut_pix_frac"
    Maximum allowed fraction of pixels masked or removed from a candidate
    beam map before it is rejected from a stack.

??? info "cfg.min_irat"
    Minimum acceptable inverse-variance median-to-variance ratio. Used to
    reject maps with poorly behaved or highly structured inverse
    variance.

??? info "cfg.max_cn"
    Threshold on the logarithm of the correlated/white noise levels used
    during map-quality selection.

??? info "cfg.corr_ratio_cut"
    Maximum allowed correlated-to-white-noise ratio, subject to the
    adjustment based on the absolute noise levels.

??? info "cfg.miscenter_thresh"
    Maximum allowed displacement, in pixels, between the estimated beam
    center and the expected center of the reprojected map.

### Noise and diagnostic configuration

??? info "cfg.n_lmin"
    Lower multipole bound used when estimating map noise.

??? info "cfg.n_lmax"
    Upper multipole bound used when estimating map noise.

??? info "cfg.log_thresh"
    Logarithmic threshold used when generating beam-map diagnostic plots.

??? info "cfg.empir_cov"
    Whether to calculate empirical covariance information from the
    individual beam fits. In the fitting stage, empirical covariance is
    loaded when more than five contributing fits are available. It also
    controls whether empirical scatter based summary plots are generated.

??? info "cfg.lmax"
    Maximum multipole used when calculating the beam window function and
    Bessel profile covariance.

??? info "cfg.cov_modes"
    Number or configuration of covariance modes retained when calculating
    the Bessel profile covariance.

### Split and epoch configuration

??? info "cfg.det_split_dir"
    Directory associated with detector splits. This field is initialized
    by setup_cfg but is not directly used by the pointing-fit script.


??? info "cfg.det_splits"
    Detector split names to process. Each script automatically adds
    `"full"` to this list when selecting stack-map jobs.

??? info "cfg.split_by"
    Split dimensions used to select stack jobs for fitting.
    These can be anything that `beam_utils.get_split_vec` can
    understand.

??? info "cfg.metasplits"
    Metadata split definitions passed to the beam-processing utilities
    when constructing split vectors.

??? info "cfg.epochs"
    Sequence of `(start, end)` time ranges over which stack jobs are
    constructed or selected. The fitting stage only processes jobs whose
    epoch range matches one of these configured ranges.


### Abscal

??? info "cfg.abscal_r_frac"
    Value to scale the mapmaking radius by when selecting a portion of the profile to
    use when estimating amplitude for abscal. A value of 1 will use the full mapmaking region.
    You probably never want this to be > 1.

??? info "cfg.abscal_lmin"
    Minimum ell to use when estimating amplitude for abscal.
    If <0 the estimating is all done in real space. If this is >= 0
    then a second estimation is done from the window function at ells
    greater than or equal to this.

??? info "cfg.abscal_from_model"
    If True use the model profile when estimating abscal.
    If False use the data profile instead.
"""

import argparse
import os
import time
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import yaml


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values from `b` take precedence over values from `a`. When a key
    exists in both dictionaries and both corresponding values are
    dictionaries, those dictionaries are merged recursively. All other
    values from `b` replace the corresponding values from `a`.
    Values are deep-copied when inserted into the result, so mutable values
    in the input dictionaries are not shared with the returned dictionary.

    Parameters
    ----------
    a : dict[str, Any]
        The base dictionary.
    b : dict[str, Any]
        The dictionary whose values take precedence.

    Returns
    -------
    dict[str, Any]
        A new dictionary containing the recursively merged values. Neither
        input dictionary is modified.
    """
    result = deepcopy(a)
    for bk, bv in b.items():
        av = result.get(bk)
        if isinstance(av, dict) and isinstance(bv, dict):
            result[bk] = deep_merge(av, bv)
        else:
            result[bk] = deepcopy(bv)
    return result


def load_config(start_cfg: dict[str, Any], cfg_path: str) -> dict[str, Any]:
    """
    Load a configuration file and recursively merge its base configuration.

    The configuration at `cfg_path` is loaded and merged with
    `start_cfg`. If the loaded configuration contains a `base` key then
    the referenced base configuration is loaded recursively and is merged in.

    Values from the more specific configuration take precedence over values
    from its base configuration.

    Note that relative ``"base"`` paths are resolved relative to the directory
    containing the configuration file that references them.

    Parameters
    ----------
    start_cfg : dict[str, Any]
        Configuration values that take precedence over values loaded from `cfg_path`.
    cfg_path : str
        Path to the YAML configuration file to load.

    Returns
    -------
    dict[str, Any]
        The fully merged configuration.

    Raises
    ------
    FileNotFoundError
        If `cfg_path` or a referenced base configuration does not exist.
    yaml.YAMLError
        If a configuration file contains invalid YAML.
    """
    with open(cfg_path) as file:
        new_cfg = yaml.safe_load(file)

    cfg = deep_merge(new_cfg, start_cfg)
    if "base" in new_cfg:
        base_path = new_cfg["base"]
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(cfg_path), base_path)
        return load_config(cfg, base_path)

    return cfg


def get_args_cfg() -> tuple[argparse.Namespace, dict[str, Any]]:
    """
    Parse command-line arguments and load the configuration file.
    Run the script with `--help` for details.

    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    cfg : dict[str, Any]
        Configuration loaded from the YAML file.
        This is loaded recursively, see `load_config` for details.
    """
    # Only the config is necessary; the rest are just for ease of use.
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", help="Path to the config file")
    parser.add_argument(
        "--plot_only",
        "-p",
        action="store_true",
        help="Don't do any fitting or mamaking, just plot TODs or existing maps",
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help="Don't do any fitting or mamaking, just plot a summary of results",
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
    cfg = load_config({}, args.cfg)

    return args, cfg


def setup_cfg(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    replace: Optional[dict[str, str]] = None,
    apply_ds: bool = False,
) -> tuple[argparse.Namespace, str]:
    """
    Apply defaults and command-line overrides to a loaded configuration.
    This also lets you rename things. When loading from `cfg_str` you
    don't need to apply any processing, you can just convert directly
    to a dict to get the final config.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    cfg : dict[str, Any]
        Configuration dictionary to modify.
    replace : Optional[dict[str, str]], default: None
        Mapping of configuration keys to rename. Keys present in ``cfg`` are
        copied to their new names and removed from their old names.
    apply_ds : bool, default: False
        Whether downsampling should be applied when calculating
        sample-dependent configuration values.

    Returns
    -------
    cfg : argparse.Namespace
        Configuration converted to an attribute-accessible namespace.
    cfg_str : str
        YAML representation of the final configuration.
    """
    # TODO: Make a default config yaml file and only do modifications here

    if replace is None:
        replace = {}

    # General pipeline settings
    cfg["root_dir"] = os.path.expanduser(cfg.get("root_dir", "~"))
    cfg["tel"] = cfg.get("tel", "lat")
    cfg["pointing_type"] = cfg.get("pointing_type", "pointing_model")
    cfg["append"] = cfg.get("append", "")
    cfg["test_append"] = cfg.get("test_append", "")
    cfg["copy_fits_test"] = cfg.get("copy_fits_test", True)
    cfg["single_det"] = cfg.get("single_det", False)
    cfg["ctx_path"] = cfg.get(
        "ctx_path",
        f"/global/cfs/cdirs/sobs/metadata/{cfg['tel']}/contexts/"
        "smurf_detcal_local.yaml",
    )
    cfg["preprocess_cfg"] = cfg.get("preprocess_cfg", None)
    cfg["map_source_list"] = cfg.get("map_source_list", ["mars", "saturn"])
    cfg["fit_source_list"] = cfg.get("fit_source_list", ["mars", "saturn"])
    cfg["start_time"] = cfg.get("start_time", 0)
    if args.lookback is not None:
        cfg["start_time"] = time.time() - 3600 * args.lookback
    cfg["stop_time"] = cfg.get("stop_time", 20000000000)
    if args.lookback is not None:
        cfg["stop_time"] = time.time()
    cfg["nominal_fwhm"] = cfg.get(
        "nominal_fwhm",
        {
            "f030": 7.4,
            "f040": 5.1,
            "f090": 2.0,
            "f150": 1.3,
            "f220": 0.95,
            "f280": 0.83,
        },
    )
    cfg["min_samps"] = cfg.get("min_samps", 1000)
    cfg["min_dets"] = cfg.get("min_dets", 30)

    # Mapmaking
    cfg["extent"] = cfg.get("extent", 600)
    cfg["res"] = cfg.get("res", (10 / 3600.0) * np.pi / 180.0)
    cfg["map_mask_size"] = cfg.get("map_mask_size", 0.1)
    cfg["apply_fscale"] = cfg.get("apply_fscale", True)
    cfg["aperature"] = cfg.get("aperature", 6)
    cfg["buf"] = cfg.get("buf", 30)
    cfg["buf_cropped"] = cfg.get("buf_cropped", 10)
    cfg["smooth_kern"] = cfg.get("smooth_kern", 60)
    cfg["snr_extent"] = cfg.get("snr_extent", 500)
    cfg["extent_highres"] = cfg.get("extent_highres", 3600)
    cfg["pixsize_highres"] = cfg.get("pixsize_highres", 1)
    cfg["search_mask"] = cfg.get(
        "search_mask",
        {"shape": "circle", "xyr": (0, 0, 0.5)},
    )
    cfg["del_map"] = cfg.get("del_map", True)
    cfg["cgiters_single"] = cfg.get("cgiters_single", 30)
    cfg["cgiters_full"] = cfg.get("cgiters_full", 400)
    cfg["mlpass"] = cfg.get("mlpass", 3)
    cfg["comps"] = cfg.get("comps", "TQU")
    cfg["force_zero_cent"] = cfg.get("force_zero_cent", False)
    cfg["n_modes"] = cfg.get("n_modes", 10)
    cfg["relcal_range"] = cfg.get("relcal_range", [0.3, 2])
    cfg["min_det_secs"] = cfg.get("min_det_secs", 600)

    # Pointing fits
    cfg["forced_ws"] = args.forced_ws if args.forced_ws is not None else []
    if cfg.get("try_all", False):
        cfg["forced_ws"] = ["ws0", "ws1", "ws2", "ws."]
    cfg["max_dur"] = cfg.get("max_dur", 2)
    cfg["nominal_path"] = os.path.expanduser(
        cfg.get(
            "nominal_path",
            f"~/data/pointing/{cfg['tel']}/nominal/focal_plane.h5",
        )
    )
    cfg["pointing_mask"] = cfg.get(
        "pointing_mask",
        {"shape": "circle", "xyr": (0, 0, 0.75)},
    )
    cfg["ds"] = cfg.get("ds", 5)
    ds = cfg["ds"] if apply_ds else 1
    cfg["hp_fc"] = cfg.get("hp_fc", 4)
    cfg["lp_fc"] = cfg.get("lp_fc", 30)
    cfg["n_med"] = cfg.get("n_med", 5)
    cfg["n_std"] = cfg.get("n_std", 10)
    cfg["block_size"] = int(cfg.get("block_size", 200) // ds)
    cfg["trim_samps"] = cfg.get("trim_samps", 200) // ds
    cfg["min_samps"] = cfg["min_samps"] / ds
    cfg["min_hits"] = cfg.get("min_hits", 1)
    cfg["high_hits"] = cfg.get("high_hits", 5)
    cfg["max_chisq"] = cfg.get("max_chisq", 2.5)
    cfg["min_R2"] = cfg.get("min_R2", 0.01)
    cfg["svd_modes"] = cfg.get("svd_modes", 10)
    cfg["svd_iters"] = cfg.get("svd_iters", 5)
    cfg["iter_svd_sub"] = cfg.get("iter_svd_sub", False)
    cfg["filter_for_sources"] = cfg.get("filter_for_sources", False)
    cfg["source_flag_exp"] = cfg.get("source_flag_exp", "(svd + blind) * cent")
    cfg["fit_pars"] = cfg.get("fit_pars", {})
    cfg["pad"] = cfg.get("pad", True)
    cfg["src_msk"] = cfg.get("src_msk", True)

    # Beam-fit configuration
    cfg["sym_gauss"] = cfg.get("sym_gauss", True)
    cfg["min_snr"] = cfg.get("min_snr", 5)
    cfg["bessel_beam"] = cfg.get("bessel_beam", True)
    cfg["min_sigma"] = cfg.get("min_sigma", 3)
    cfg["n_bessel"] = cfg.get("n_bessel", 10)
    cfg["n_multipoles"] = cfg.get("n_multipoles", 3)
    cfg["skip_multipoles"] = cfg.get("skip_multipoles", [])
    cfg["bessel_wing_n_sigma"] = cfg.get("bessel_wing_n_sigma", 5)
    cfg["gauss_multipole"] = cfg.get("gauss_multipole", True)
    cfg["corr_primary"] = cfg.get("corr_primary", 280)
    cfg["eps_primary"] = cfg.get("eps_primary", 17)

    # Stacking quality cuts
    cfg["min_stack_snr"] = cfg.get("min_stack_snr", 10)
    cfg["max_pwv"] = cfg.get("max_pwv", 2.5)
    cfg["max_cut_pix_frac"] = cfg.get("max_cut_pix_frac", 0.15)
    cfg["min_irat"] = cfg.get("min_irat", 3)
    cfg["max_cn"] = cfg.get("max_cn", -2.5)
    cfg["corr_ratio_cut"] = cfg.get("corr_ratio_cut", 20)
    cfg["miscenter_thresh"] = cfg.get("miscenter_thresh", 5)

    # Noise and diagnostic configuration
    cfg["n_lmin"] = cfg.get("n_lmin", 2000)
    cfg["n_lmax"] = cfg.get("n_lmax", 60000)
    cfg["log_thresh"] = cfg.get("log_thresh", 1e-3)
    cfg["empir_cov"] = cfg.get("empir_cov", False)
    cfg["lmax"] = cfg.get("lmax", 20000)
    cfg["cov_modes"] = cfg.get("cov_modes", 20)

    # Split and epoch configuration
    cfg["det_split_dir"] = cfg.get("det_split_dir", "")
    cfg["det_splits"] = cfg.get("det_splits", [])

    cfg["split_by"] = cfg.get(
        "split_by",
        [
            "band",
            "tube_slot+band",
            "source+band",
            "source+tube_slot+band",
        ],
    )
    cfg["metasplits"] = cfg.get("metasplits", {})
    cfg["epochs"] = cfg.get("epochs", [(0, 2e10)])

    # Abscal
    cfg["abscal_r_frac"] = cfg.get("abscal_r_frac", 0.3)
    cfg["abscal_lmin"] = cfg.get("abscal_lmin", -1)
    cfg["abscal_from_model"] = cfg.get("abscal_from_model", False)

    # Rename for our scope
    for old_name, new_name in replace.items():
        if old_name not in cfg:
            continue
        cfg[new_name] = cfg[old_name]
        del cfg[old_name]

    cfg_str = yaml.dump(cfg)

    return argparse.Namespace(**cfg), cfg_str


def setup_paths(
    root_dir: str,
    project: str,
    tel: str,
    append: str = "",
) -> tuple[str, str]:
    """
    Create and return the plot and data directories.

    Parameters
    ----------
    root_dir : str
        Root directory under which the project directories are created.
    project : str
        Project name used to construct the directory paths.
    tel : str
        Telescope name used to construct the directory paths.
    append : str, optional
        Additional path component appended to the project/telescope paths.

    Returns
    -------
    plot_dir : str
        Path to the plot directory.
    data_dir : str
        Path to the data directory.
    """
    plot_dir = os.path.join(root_dir, "plots", project, tel, append)
    data_dir = os.path.join(root_dir, "data", project, tel, append)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    return plot_dir, data_dir
