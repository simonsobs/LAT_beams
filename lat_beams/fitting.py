"""
Just fitting a single TOD for now, can genralize later (but maybe just do a WITCH interface for that)
Maybe should add priors
"""
import numpy as np
from sotodlib.tod_ops.filters import high_pass_sine2, low_pass_sine2, fourier_filter
from tqdm.auto import tqdm
from scipy.optimize import minimize
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d
from so3g.proj import coords, quat
import sotodlib.coords.planets as planets
from sotodlib import core
import so3g
from . import noise as nn


# TODO get these working with our data
def gauss_grad(x, y, x0, y0, amp, sigma):
    dx = x - x0
    dy = y - y0
    var = sigma**2
    gauss = amp*np.exp(-.5*(dx**2 + dy**2)/(var))

    dfdx = -(dx/var)*gauss
    dfdy = -(dy/var)*gauss
    dady = gauss/amp
    dsdy = ((dx**2 + dy**2)/sigma**3)*gauss 
    grad = np.array([dfdx, dfdy, dady, dsdy])

    return gauss, grad

def objective(aman, pars):
    npar = len(pars)
    chisq = np.array(0)
    grad = np.zeros(npar)
    curve = np.zeros((npar, npar))

    pred_dat, grad_dat = gauss_grad(aman.xi, aman.eta, *pars)
    
    resid = aman.signal - pred_dat
    resid_filt = nn.apply_noise(aman, resid) 
    chisq = np.sum(resid * resid_filt)
    grad_filt = np.zeros_like(grad_dat)
    for i in range(npar):
        grad_filt[i] =  nn.apply_noise(aman, grad_dat[i])
    grad_filt = np.reshape(grad_filt, (npar, -1))
    grad_dat = np.reshape(grad_dat, (npar, -1))
    resid = resid.ravel()
    grad = np.dot(grad_filt, np.transpose(resid)) 
    curve = np.dot(grad_filt, np.transpose(grad_dat))

    return chisq, grad, curve

def get_xieta_src_centered_new( ctime, az, el, roll, sso_name,):
    """
    Modified from analyze_bright_ptsrc
    """
    csl = so3g.proj.CelestialSightLine.az_el(ctime, az, el, roll=roll, weather="typical")
    q_bore = csl.Q

    # planet position
    planet = planets.SlowSource.for_named_source(sso_name, ctime[int(len(ctime)/2)])
    ra0, dec0 = planet.pos(ctime)
    q_obj = so3g.proj.quat.rotation_lonlat(ra0, dec0)


    q_total = ~q_bore * q_obj
    xi, eta, _ = quat.decompose_xieta(q_total)

    return xi, eta

def gaussian2d(xi, eta, a, xi0, eta0, fwhm_xi, fwhm_eta, phi):
    """
    Stolen from analyze_bright_ptsrc

    Simulate a time stream with an Gaussian beam model
    Args
    ------
    xi, eta: cordinates in the detector's system
    a: float
        amplitude of the Gaussian beam model
    xi0, eta0: float, float
        center position of the Gaussian beam model
    fwhm_xi, fwhm_eta, phi: float, float, float
        fwhm along the xi, eta axis (rotated)
        and the rotation angle (in radians)

    Ouput:
    ------
    sim_data: 1d array of float
        Time stream at sampling points given by xieta
    """
    xi_rot = xi * np.cos(phi) - eta * np.sin(phi)
    eta_rot = xi * np.sin(phi) + eta * np.cos(phi)
    factor = 2 * np.sqrt(2 * np.log(2))
    xi_coef = -0.5 * (xi_rot - xi0) ** 2 / (fwhm_xi / factor) ** 2
    eta_coef = -0.5 * (eta_rot - eta0) ** 2 / (fwhm_eta / factor) ** 2
    sim_data = a * np.exp(xi_coef + eta_coef)
    return sim_data

def pointing_quickfit(aman, bandpass_range=(None, None), fwhm=np.deg2rad(0.5), max_rad=None, source='mars', bin_priors=False):
    """
    Modified from analyze_bright_ptsrc
    """
    if max_rad is None:
        max_rad = 5 * fwhm

    ts = aman.timestamps
    az = aman.boresight.az
    el = aman.boresight.el
    roll = aman.boresight.roll

    focal_plane = core.AxisManager(aman.dets)
    focal_plane.wrap("xi", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("eta", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("gamma", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("fwhm", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("amp", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])
    focal_plane.wrap("dist", np.zeros(len(aman.dets.vals), dtype=float), [(0, "dets")])

    xi, eta = get_xieta_src_centered_new(ts, az, el, roll, source)

    def filter_tod(am, signal_name='resid'):
        sig_filt_name = f"{signal_name}_filt"
        am[sig_filt_name] = am[signal_name].copy()
        filt_kw = dict(
            detrend='linear', resize='zero_pad', axis_name='samps',
            signal_name=sig_filt_name, time_name='timestamps'
        )
        if bandpass_range[0] is not None:
            highpass = high_pass_sine2(cutoff=bandpass_range[0])
            am[sig_filt_name] = fourier_filter(am, highpass, **filt_kw)
        if bandpass_range[1] is not None:
            lowpass = low_pass_sine2(cutoff=bandpass_range[1])
            am[sig_filt_name] = fourier_filter(am, lowpass, **filt_kw)
        return am

    def fit_func(x, fit_am, xi, eta):
        xi0, eta0, amp, fwhm = x
        model = gaussian2d(
            xi, eta, amp, xi0, eta0, fwhm, fwhm, 0
        )
        fit_am.resid = (fit_am.signal.ravel() - model).reshape(fit_am.resid.shape)
        fit_am = filter_tod(fit_am, signal_name='resid')
        return np.sum(fit_am.resid * fit_am.resid_filt)

    for i, det in enumerate(tqdm(aman.dets.vals)):
        fit_am = aman.restrict("dets", [det], in_place=False)
        fit_am.wrap("resid", fit_am.signal.copy(), [(0, "dets"), (1, "samps")])
        fit_am.wrap("resid_filt", np.zeros_like(fit_am.signal), [(0, "dets"), (1, "samps")])
        fit_am = filter_tod(fit_am) 

        max_idx = np.argmax(fit_am.resid_filt[0])
        xi_max = xi[max_idx]
        eta_max = eta[max_idx]
        xi0, eta0 = xi_max, eta_max
        # Bin in xi and eta
        # Should I do this in 2d as a crappy map?
        if bin_priors:
            xi_binned, edges, _ = binned_statistic(xi, fit_am.resid_filt[0], bins = int(np.ptp(xi)/(.1*fwhm)))
            xi_cents = .5*(edges[:-1] + edges[1:])
            xi_binned = gaussian_filter1d(xi_binned, (fwhm/2.3548)/np.mean(np.diff(edges)))
            xi0 = xi_cents[np.nanargmax(xi_binned)]
            eta_binned, edges, _ = binned_statistic(eta, fit_am.resid_filt[0], bins = int(np.ptp(eta)/(.1*fwhm)))
            eta_cents = .5*(edges[:-1] + edges[1:])
            eta_binned = gaussian_filter1d(eta_binned, (fwhm/2.3548)/np.mean(np.diff(edges)))
            eta0 = eta_cents[np.nanargmax(eta_binned)]
        amp = np.ptp(fit_am.signal) * 3
        msk_samps = np.where((xi - xi0)**2 + (eta - eta0)**2 < max_rad**2)[0].astype(float)
        if len(msk_samps) < 10 and bin_priors:
            xi0, eta0 = xi_max, eta_max
            msk_samps = np.where((xi - xi0)**2 + (eta - eta0)**2 < max_rad**2)[0].astype(float)
        if len(msk_samps) < 10:
            print(f"Not enouth samples flagged for {det}")
            msk_samps = np.arange(aman.samps.count)
        sl = slice(int(np.percentile(msk_samps, 5)), int(np.percentile(msk_samps, 95)))
        fit_am.restrict("samps", sl)

        res = minimize(fit_func, [xi0, eta0, amp, fwhm], bounds=[(xi0-max_rad, xi0+max_rad), (eta0-max_rad, eta0+max_rad), (-1, 10*amp), (fwhm/4, 4*fwhm)], args=(fit_am, xi[sl], eta[sl]), method="Nelder-Mead")

        focal_plane.xi[i] = res.x[0]
        focal_plane.eta[i] = res.x[1]
        focal_plane.amp[i] = res.x[2]
        focal_plane.fwhm[i] = res.x[3]
        focal_plane.dist[i] = np.sqrt((res.x[0] - xi0)**2 + (res.x[1] - eta0)**2)

        if not res.success:
            focal_plane.amp[i] = -np.inf

    return focal_plane 
