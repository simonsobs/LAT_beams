"""
Tools for fitting detector pointing from source-observation TODs.

The main fitting routine, `fit_tod_pointing`, estimates each detector's
offset, beam width, and amplitude by fitting a symmetric 2D Gaussian to
the source response in source-centered (xi, eta) coordinates. The TOD
residuals are filtered before fitting, and optional binned priors are
used to improve the initial pointing estimate.

The module also provides helpers for converting boresight coordinates
to source-centered coordinates, constructing binned position priors,
and filtering TOD residuals. See the individual function docstrings for
details.
"""

import logging
import sys
import warnings
from typing import Optional, cast

import numpy as np
import so3g
import sotodlib.coords.planets as planets
from astropy.convolution import (
    Gaussian1DKernel,
    Gaussian2DKernel,
    convolve,
    convolve_fft,
)
from jaxtyping import Float
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.signal import detrend
from scipy.stats import binned_statistic, binned_statistic_2d
from so3g.proj import Ranges, quat
from sotodlib import core
from sotodlib.core import AxisManager
from sotodlib.tod_ops.fft_ops import (
    RFFTObj,
    find_inferior_integer,
    find_superior_integer,
)
from sotodlib.tod_ops.filters import (
    FilterChain,
    fourier_filter,
    high_pass_sine2,
    identity_filter,
)
from sotodlib.tod_ops.filters import logger as flog
from sotodlib.tod_ops.filters import low_pass_sine2
from tqdm.auto import tqdm
from typing_extensions import Optional, cast

from .core import gaussian2d

flog.setLevel(logging.ERROR)


def get_xieta_src_centered(
    ctime: Float[NDArray, "n"],
    az: Float[NDArray, "n"],
    el: Float[NDArray, "n"],
    roll: Float[NDArray, "n"],
    sso_name: str,
) -> tuple[Float[NDArray, "n"], Float[NDArray, "n"]]:
    """Get source-centered xi and eta coordinates.

    The source position is evaluated using the center of the input time
    range. This is a good approximation for a slowly moving source, but
    can become inaccurate for a long time range or one spanning both sides
    of a transit.

    Parameters
    ----------
    ctime : Float[NDArray, "n"]
        Observation times.
    az : Float[NDArray, "n"]
        Boresight azimuth angles in radians.
    el : Float[NDArray, "n"]
        Boresight elevation angles in radians.
    roll : Float[NDArray, "n"]
        Boresight roll angles in radians.
    sso_name : str
        Name of the solar-system object.

    Returns
    -------
    xi : Float[NDArray, "n"]
        Source-centered xi coordinates in radians.
    eta : Float[NDArray, "n"]
        Source-centered eta coordinates in radians.
    """
    csl = so3g.proj.CelestialSightLine.az_el(
        ctime, az, el, roll=roll, weather="typical"
    )
    q_bore = csl.Q  # type: ignore

    # planet position
    planet = planets.SlowSource.for_named_source(sso_name, ctime[int(len(ctime) / 2)])
    ra0, dec0 = planet.pos(ctime)
    q_obj = so3g.proj.quat.rotation_lonlat(ra0, dec0)

    q_total = ~q_bore * q_obj
    xi, eta, _ = quat.decompose_xieta(q_total)

    return xi, eta


def _empty_fp(aman: AxisManager) -> AxisManager:
    """
    Create an empty focal-plane result for an observation.

    Parameters
    ----------
    aman : AxisManager
        Input observation containing the detector and boresight axes.

    Returns
    -------
    focal_plane : AxisManager
        Empty focal-plane ``AxisManager`` with one entry per detector.
    """
    ndet = cast(int, aman.dets.count)
    focal_plane = core.AxisManager(aman.dets)

    focal_plane.wrap("xi", np.zeros(ndet, dtype=float), [(0, "dets")])
    focal_plane.wrap("eta", np.zeros(ndet, dtype=float), [(0, "dets")])
    focal_plane.wrap("gamma", np.zeros(ndet, dtype=float), [(0, "dets")])
    focal_plane.wrap("fwhm", np.zeros(ndet, dtype=float), [(0, "dets")])
    focal_plane.wrap("amp", np.zeros(ndet, dtype=float), [(0, "dets")])
    focal_plane.wrap("dist", np.zeros(ndet, dtype=float), [(0, "dets")])
    focal_plane.wrap("hits", np.zeros(ndet, dtype=int), [(0, "dets")])
    focal_plane.wrap("az", np.zeros(ndet, dtype=float), [(0, "dets")])
    focal_plane.wrap("el", np.zeros(ndet, dtype=float), [(0, "dets")])
    focal_plane.wrap(
        "roll",
        np.zeros(ndet, dtype=float)
        + np.mean(cast(Float[np.ndarray, "ndet"], aman.boresight.roll)),
        [(0, "dets")],
    )
    focal_plane.wrap(
        "reduced_chisq",
        np.zeros(ndet, dtype=float),
        [(0, "dets")],
    )
    focal_plane.wrap("R2", np.zeros(ndet, dtype=float), [(0, "dets")])

    return focal_plane


def _bin_priors_1d(
    fit_am: AxisManager,
    xi0: float,
    eta0: float,
    fwhm: float,
) -> tuple[float, float]:
    """
    Estimate the beam center by independently binning xi and eta.

    Parameters
    ----------
    fit_am : AxisManager
        Detector TOD containing `xi`, `eta`, and filtered residual data.
    xi0 : float
        Initial xi position in radians.
    eta0 : float
        Initial eta position in radians.
    fwhm : float
        Approximate beam FWHM in radians.

    Returns
    -------
    xi0 : float
        Estimated xi center in radians.
    eta0 : float
        Estimated eta center in radians.
    """
    xi = np.asarray(fit_am.xi)
    eta = np.asarray(fit_am.eta)
    xi_binned, edges, _ = binned_statistic(
        xi, fit_am.resid_filt[0], bins=int(np.ptp(xi) / (1 * fwhm))
    )
    xi_cents = 0.5 * (edges[:-1] + edges[1:])
    xi_binned = convolve(
        xi_binned, Gaussian1DKernel((fwhm / 2.3548) / np.mean(np.diff(edges)))
    )
    if not np.all(np.isnan(xi_binned)):
        xi0 = xi_cents[np.nanargmax(xi_binned)]
    eta_binned, edges, _ = binned_statistic(
        eta, fit_am.resid_filt[0], bins=int(np.ptp(eta) / (1 * fwhm))
    )
    eta_cents = 0.5 * (edges[:-1] + edges[1:])
    eta_binned = convolve(
        eta_binned, Gaussian1DKernel((fwhm / 2.3548) / np.mean(np.diff(edges)))
    )
    if not np.all(np.isnan(eta_binned)):
        eta0 = eta_cents[np.nanargmax(eta_binned)]

    return xi0, eta0


def _bin_priors_2d(
    fit_am: AxisManager,
    xi0: float,
    eta0: float,
    fwhm: float,
) -> tuple[float, float]:
    """
    Estimate the beam center from a smoothed 2D binned TOD.

    Parameters
    ----------
    fit_am : AxisManager
        Detector TOD containing `xi`, `eta`, and filtered residual data.
    xi0 : float
        Initial xi position in radians.
    eta0 : float
        Initial eta position in radians.
    fwhm : float
        Approximate beam FWHM in radians.

    Returns
    -------
    xi0 : float
        Estimated xi center in radians.
    eta0 : float
        Estimated eta center in radians.
    """
    xi = np.asarray(fit_am.xi)
    eta = np.asarray(fit_am.eta)
    binned, x_edges, y_edges, _ = binned_statistic_2d(
        xi,
        eta,
        fit_am.resid_filt[0],
        bins=(int(np.ptp(xi) / (1 * fwhm)), int(np.ptp(eta) / (1 * fwhm))),  # type: ignore
    )
    warnings.filterwarnings("ignore", category=UserWarning, append=True)
    binned = convolve_fft(
        binned,
        Gaussian2DKernel(
            1 * (fwhm / 2.3548) / np.mean(np.diff(x_edges)),
            1 * (fwhm / 2.3548) / np.mean(np.diff(y_edges)),
        ),
    )
    xi_cents = 0.5 * (x_edges[:-1] + x_edges[1:])
    eta_cents = 0.5 * (y_edges[:-1] + y_edges[1:])
    if not np.all(np.isnan(binned)):
        max_idx = np.unravel_index(np.nanargmax(binned), binned.shape)
        xi0 = xi_cents[max_idx[0]]
        eta0 = eta_cents[max_idx[1]]
    return xi0, eta0


def filter_tod(
    am: AxisManager,
    filt: FilterChain,
    signal_name: str = "resid",
    rfft: Optional[RFFTObj] = None,
) -> AxisManager:
    """
    Apply a Fourier-domain filter to a signal in an AxisManager.

    Parameters
    ----------
    am : AxisManager
        AxisManager containing the signal and time axes.
    filt : FilterChain
        sotodlib Fourier filter to apply.
    signal_name : str, default="resid"
        Name of the signal field to filter.
    rfft : RFFTObj, optional
        Precomputed FFT configuration. If `None`, the filter constructs
        the required configuration.

    Returns
    -------
    AxisManager
        The input AxisManager with `{signal_name}_filt` containing the
        filtered signal.
    """
    sig_filt_name = f"{signal_name}_filt"
    am[sig_filt_name] = am[signal_name].copy()
    filt_kw = dict(
        detrend=None,
        resize=None,
        axis_name="samps",
        signal_name=sig_filt_name,
        time_name="timestamps",
        rfft=rfft,
    )
    am[sig_filt_name] = fourier_filter(am, filt, **filt_kw)  # type: ignore
    return am


def fit_tod_pointing(
    aman: AxisManager,
    bandpass_range: tuple[Optional[float], Optional[float]] = (None, None),
    fwhm: float = np.deg2rad(0.5),
    max_rad: Optional[float] = None,
    source: str = "mars",
    bin_priors: bool = True,
    bin_2d: bool = True,
    pos_priors: Optional[Float[NDArray, "ndets 2"]] = None,
    show_tqdm: bool = False,
    min_snr: float = 5.0,
) -> AxisManager:
    """Fit detector beam positions from a source-observation TOD.

    Each detector is fit independently with a symmetric 2D Gaussian. The
    source-centered xi/eta trajectory is used as the detector position,
    and the TOD is filtered before fitting. Initial positions can optionally
    be estimated from binned TOD data or supplied as positional priors.

    Parameters
    ----------
    aman : AxisManager
        Observation containing detector TOD and boresight data. Must contain
        `dets`, `samps`, `timestamps`, `signal`, and `boresight` fields.
        The boresight must contain `az`, `el`, and `roll`.
    bandpass_range : tuple[Optional[float], Optional[float]], default: (None, None)
        Low and high frequency cutoffs in Hz. A value of `None` disables
        the corresponding cutoff.
    fwhm : float, default: np.deg2rad(0.5)
        Initial beam FWHM in radians.
    max_rad : float, optional
        Maximum radius in radians around the initial beam position to include
        in the fit. If `None`, uses `20 * fwhm`.
    source : str, default: "mars"
        Solar-system source being observed.
    bin_priors : bool, default: True
        If `True`, estimate detector positions by binning and smoothing the
        TOD unless a positional prior is supplied.
    bin_2d : bool, default: True
        If `True`, estimate the position from a 2D binned map. Otherwise,
        estimate xi and eta independently.
    pos_priors : Float[NDArray, "ndets 2"], optional
        Initial positional priors in radians. Each row contains `(xi, eta)`
        for one detector. A row of `(nan, nan)` disables the prior for that
        detector.
    show_tqdm : bool, default: False
        If `True`, display a progress bar while fitting detectors.
    min_snr : float, default: 5.0
        Signal-to-noise threshold used when calculating the number of hits.

    Returns
    -------
    focal_plane : AxisManager
        Fitted detector parameters. Fields include:

        - `xi`: fitted xi position in radians.
        - `eta`: fitted eta position in radians.
        - `gamma`: placeholder rotation angle, currently zero.
        - `fwhm`: fitted symmetric beam FWHM in radians.
        - `amp`: fitted beam amplitude in the units of `aman.signal`.
        - `dist`: distance between the fitted position and initial position
          in radians.
        - `az`: azimuth at the detector beam crossing in radians.
        - `el`: elevation at the detector beam crossing in radians.
        - `roll`: roll at the detector beam crossing in radians.
        - `reduced_chisq`: reduced chi-squared of the fit.
        - `R2`: coefficient of determination for the fitted TOD.
        - `hits`: number of scan samples above `min_snr`.
    """
    # TODO: Can use full detector map to fit the array at once. This hasnt been written.
    # Right now det maps for LAT not good; SAT will be ok to test with though. Should provide both options here,
    # since doing full det maps fit is faster than per det.
    # Either way, want to do per TOD because per TOD would be a refinement to the full det maps fit.
    if pos_priors is not None and len(pos_priors) != aman.dets.count:
        raise ValueError(
            f"{len(pos_priors)} positional priors given for {aman.dets.count} detectors"
        )
    if pos_priors is None:
        pos_priors = np.ones((cast(int, aman.dets.count), 2)) * np.nan
    sigma = fwhm / 2.3548
    if max_rad is None:
        max_rad = 20 * fwhm

    focal_plane = _empty_fp(aman)
    mean_el = np.mean(np.asarray(aman.boresight.el))

    # getting xi eta in a coordinate system where (0, 0) is the planet youre fitting. Expecting trimmed TOD for source.
    # Cannot include both rising and setting (ie a sign change). Note that a transit is flat -- so is ok.
    xi, eta = get_xieta_src_centered(
        np.asarray(aman.timestamps),
        np.asarray(aman.boresight.az),
        np.asarray(aman.boresight.el),
        np.asarray(aman.boresight.roll),
        source,
    )
    aman.wrap("xi", xi, [(0, "samps")])
    aman.wrap("eta", eta, [(0, "samps")])

    az_d = detrend(np.asarray(aman.boresight.az))
    az_v = np.diff(az_d, prepend=az_d[0])
    d_az = np.sign(az_v)
    scan_samps = np.ptp(az_d) / (np.median(np.abs(az_v)))
    turnarounds = np.diff(d_az, prepend=d_az[0]) != 0
    turnarounds = ~(
        Ranges.from_mask(turnarounds).buffer(int(0.05 * scan_samps))
    )  # Invert for convenience

    # 0 is the highpass part, 1 lowpass part.
    filt = identity_filter()
    if bandpass_range[0] is not None:
        filt *= high_pass_sine2(cutoff=bandpass_range[0])
    if bandpass_range[1] is not None:
        filt *= low_pass_sine2(cutoff=bandpass_range[1])

    def fit_func(x, fit_am, filt, rfft, scales):
        xi0, eta0, amp, fwhm, offset = x * scales
        model = gaussian2d(
            np.asarray((fit_am.eta, fit_am.xi)), amp, xi0, eta0, fwhm, fwhm, 0, offset
        )
        fit_am.resid = (fit_am.signal.ravel() - model).reshape(fit_am.resid.shape)
        fit_am = filter_tod(fit_am, filt, signal_name="resid", rfft=rfft)
        return (
            np.sum(np.asarray(fit_am.resid) * np.asarray(fit_am.resid_filt)) * fit_am.wn
        )

    # Loop through all detectors and fit them one at a time.
    it = np.asarray(aman.dets.vals)
    if show_tqdm:
        it = tqdm(np.asarray(aman.dets.vals))
    aman.signal = np.asarray(aman.signal, dtype=np.float32)
    for i, det in enumerate(it):
        if show_tqdm:
            sys.stderr.flush()
        # Make a temporary restricted axis manager with just the one desired detector.
        fit_am = aman.restrict("dets", [det], in_place=False)
        # Containers for residual of fit and the filtered residual.
        fit_am.wrap("resid", fit_am.signal.copy(), [(0, "dets"), (1, "samps")])
        fit_am.wrap(
            "resid_filt", np.zeros_like(fit_am.signal), [(0, "dets"), (1, "samps")]
        )
        # Estimateing white noise by taking the standard deviation of the filtered TOD.
        fit_am = filter_tod(fit_am, filt)
        std = np.std(np.asarray(fit_am.resid_filt))
        if std == 0:
            focal_plane.amp[i] = -np.inf
            continue
        fit_am.wrap("wn", 1.0 / std.item() ** 2)

        # Get starting guess for where the detector is looking. Bad guess = bad pointing final fit. Descends into a local instead of global minima.
        # Get xi, eta at maximum sample, but this is not a very good guess.
        max_idx = np.argmax(np.asarray(fit_am.resid_filt[0]))
        xi_max = xi[max_idx]
        eta_max = eta[max_idx]
        xi0, eta0 = float(xi_max), float(eta_max)
        _bin_priors = bin_priors
        # If it has a det map fit then can use given positional prior
        if np.all(np.isfinite(pos_priors[i])):
            xi0, eta0 = pos_priors[i]
            _bin_priors = False

        # Determine if 1d or 2d binning used.
        # Bin in xi and eta
        if _bin_priors and not bin_2d:
            xi0, eta0 = _bin_priors_1d(fit_am, xi0, eta0, fwhm)
        elif _bin_priors and bin_2d:
            xi0, eta0 = _bin_priors_2d(fit_am, xi0, eta0, fwhm)
        msk_samps = np.where((xi - xi0) ** 2 + (eta - eta0) ** 2 < max_rad**2)[
            0
        ].astype(float)
        if len(msk_samps) < 10 and _bin_priors:
            xi0, eta0 = xi_max, eta_max
            msk_samps = np.where((xi - xi0) ** 2 + (eta - eta0) ** 2 < max_rad**2)[
                0
            ].astype(float)
        if len(msk_samps) < 10:
            print(f"Not enouth samples flagged for {det}")
            msk_samps = np.arange(cast(int, aman.samps.count))

        start = np.percentile(msk_samps, 5)
        stop = np.percentile(msk_samps, 95)
        cent = int(0.5 * (start + stop))
        nsamps = find_superior_integer(stop - start)
        if nsamps > cast(int, fit_am.samps.count):
            nsamps = find_inferior_integer(fit_am.samps.count)
        start = cent - nsamps // 2
        stop = cent + nsamps // 2 + nsamps % 2
        if start < 0:
            stop += start
            start = 0
        if stop > cast(int, fit_am.samps.count):
            start -= cast(int, fit_am.samps.count) - stop
            stop = cast(int, fit_am.samps.count)
        sl = slice(
            start + cast(int, fit_am.samps.offset),
            stop + cast(int, fit_am.samps.offset),
        )

        fit_am.restrict("samps", sl)
        rfft = RFFTObj.for_shape(1, cast(int, fit_am.samps.count), "BOTH")

        ptp = np.ptp(np.asarray(fit_am.resid_filt))
        init_pars = [xi0, eta0, ptp, fwhm, 0]
        scales = np.array([fwhm, fwhm, ptp, fwhm, ptp])
        bounds = [
            (xi0 - max_rad, xi0 + max_rad),
            (eta0 - max_rad, eta0 + max_rad),
            (0, np.inf),
            (0.1 * fwhm, 10 * fwhm),
            (-ptp, ptp),
        ]
        init_pars = np.array(init_pars) / scales
        bounds = np.array(bounds) / scales[..., None]

        # Nelder-Mead is only non-gradient method. The gradient ones get stuck on edges and find local minima. This find central global minima.
        res = minimize(
            fit_func,
            init_pars,
            bounds=bounds,
            args=(fit_am, filt, rfft, scales),
            # method="Nelder-Mead",
        )
        if res.success:
            res_nm = minimize(
                fit_func,
                res.x,
                bounds=bounds,
                args=(fit_am, filt, rfft, scales),
                method="Nelder-Mead",
                options={"maxiter": 50},
            )
            if res_nm.fun <= res.fun:
                res_nm.success = res.success
                res = res_nm
        res.x *= scales
        focal_plane.xi[i] = res.x[0]
        focal_plane.eta[i] = res.x[1]
        focal_plane.amp[i] = res.x[2]
        focal_plane.fwhm[i] = res.x[3]

        if not res.success:
            focal_plane.R2[i] = 0.0
        else:
            fit_am.resid = (
                np.asarray(fit_am.signal).ravel() - np.mean(np.asarray(fit_am.signal))
            ).reshape(fit_am.resid.shape)
            fit_am = filter_tod(fit_am, filt, signal_name="resid", rfft=rfft)
            ss_tot = (
                np.sum(np.asarray(fit_am.resid) * np.asarray(fit_am.resid_filt))
                * fit_am.wn
            )
            focal_plane.R2[i] = 1 - (res.fun / ss_tot)

        focal_plane.dist[i] = np.sqrt(
            (np.asarray(focal_plane.xi[i]) - xi0) ** 2
            + (np.asarray(focal_plane.eta[i]) - eta0) ** 2
        )
        delta_xi = xi - np.asarray(focal_plane.xi[i])
        delta_eta = eta - np.asarray(focal_plane.eta[i])

        # Lets calculate hits
        model = gaussian2d(
            np.asarray([eta, xi]),
            cast(float, focal_plane.amp[i]),
            cast(float, focal_plane.xi[i]),
            cast(float, focal_plane.eta[i]),
            cast(float, focal_plane.fwhm[i]),
            cast(float, focal_plane.fwhm[i]),
            0,
            0,
        )

        hits = Ranges.from_mask(model / std >= min_snr) * turnarounds
        focal_plane.hits[i] = len(hits.ranges())

        # Azel crossings
        xi_weights = np.exp(-0.5 * ((delta_xi / sigma) ** 2)) / (
            sigma * np.sqrt(2 * np.pi)
        )
        eta_weights = np.exp(-0.5 * ((delta_eta / sigma) ** 2)) / (
            sigma * np.sqrt(2 * np.pi)
        )
        weights = xi_weights * eta_weights
        tot_weight = np.sum(weights)
        if tot_weight == 0:
            focal_plane.R2[i] = 0
        else:
            focal_plane.az[i] = np.sum(aman.boresight.az * weights) / tot_weight
            focal_plane.el[i] = np.sum(aman.boresight.el * weights) / tot_weight
            focal_plane.roll[i] = (
                cast(float, focal_plane.roll[i])
                + cast(float, focal_plane.el[i])
                - mean_el
            )

        # Chisq
        focal_plane.reduced_chisq[i] = res.fun / (
            cast(int, fit_am.samps.count) - len(res.x)
        )

    return focal_plane
