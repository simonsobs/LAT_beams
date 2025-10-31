import numpy as np
import os
from pixell import enmap
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
import astropy.units as u
from itertools import product
from scipy.special import j1
import inspect
import msgspec
from scipy.ndimage import shift
from scipy.ndimage import distance_transform_edt
from scipy.special import factorial

maps_path = "/global/u2/a/andrs/Products/Mars/i1/"
maps_dict = "/global/u2/a/andrs/Products/Mars/i1/maps.json"
clean_map_min_snr = 15
clean_map_max_chi2_red = 5
clean_map_max_ellip = 0.2


class obsmap:
    """
    Custom class that allows for easy management of data tied to a specifc map.
    A map is simply a given observation under a specific wafer and band.
    You load objects of this class via the load_maps function.
    
    Attributes
    ----------
    obs_id: str
    band : str
        Allowed values: ['f090', 'f150']
    wafer: str
        Allowed values: ['ws0', 'ws1', 'ws2']
    kind: str
        Specifies whether it's the full data or a split in scan directions. 
        Allowed values: ['full', 'lscans', 'rscans']
    meta : dict
        Dictionary data from the maps.json file tied to the specific observation.
    time_class: str
        Indicates the time of day classification the observation corresponds to.
        Allowed values: ['day', 'night', 'transition']
    season_class: str
        Indicates what alignment period the observation corresponds to.
        Allowed values: ['first_alignment', 'second_alignment']
    solved: enmap array
        Loads the associated solved fits file to the given map. Only loads
        data when called for the first time in order to avoid wasting resources.
    weights: enmap array
        Loads the associated weights fits file to the given map. Only loads
        data when called for the first time in order to avoid wasting resources.
    snr: float
        SNR of map calculated as the fitted peak over the std of a masked region beyond a given radius.
        See 'fit_maps.ipynb' for more detail on how this is calculated.
    lscans: obsmap
        Another instance of obsmap but using only data for the left direction scans.
    rscans: obsmap
        Another instance of obsmap but using only data for the right direction scans.

    Dynamic Attributes
    ------------------
    <model_name> : Modidfied BaseModel class
        If the dictionary data tied to the map specifies that a fit exists for a given model name, 
        and that model name is tied to an actual model class, then that model is added as an attribute to
        this obsmap class.
    
    Methods
    -------
    load_fits():
        Adds the models for which the map has fits for as attributes. The models are added as instances
        and this function also adds the fitted parameters, chi2_red value and any relevant profile stats
        derived from the fits as attributes to the instances.
        This method is executed on init so it's not neccesary to call.
    """
    
    def __init__(self, 
                 obs_id: str,
                 band: str,
                 wafer: str,
                 kind: str = 'full',
                ):
        self.obs_id = obs_id
        self.band = band
        self.wafer = wafer
        self.kind = kind
       

        with open(maps_dict, "rb") as f:
            encoded = f.read()

        maps_meta = msgspec.json.decode(encoded)

        self.meta = maps_meta[self.obs_id][f"{self.band}-{self.wafer}"][self.kind]
        self.time_class = maps_meta[self.obs_id]['time_info']['time_class']
        self.season_class = maps_meta[self.obs_id]['time_info']['season_class']

        self.snr = self.meta["snr"]
        
        self._solved = None
        self._weights = None

        self.lscans = None
        self.rscans = None

        if self.kind == 'full':
            self.lscans= obsmap(self.obs_id, self.band, self.wafer, "lscans")
            self.rscans= obsmap(self.obs_id, self.band, self.wafer, "rscans")
        self.load_fits()

    @property
    def solved(self):
        if self._solved is None:
            if self.kind == 'full':
                file = os.path.join(maps_path, self.obs_id, f"{self.obs_id}_{self.band}_{self.wafer}_solved.fits")
            elif self.kind == 'lscans':
                file = os.path.join(maps_path, self.obs_id, f"{self.obs_id}_{self.band}_{self.wafer}_solved_left_scans.fits")
            elif self.kind == 'rscans':
                file = os.path.join(maps_path, self.obs_id, f"{self.obs_id}_{self.band}_{self.wafer}_solved_right_scans.fits")
            data = enmap.read_map(file)
            self._solved = data
            
        return self._solved

    @property
    def weights(self):
        if self._weights is None:
            if self.kind == 'full':
                file = os.path.join(maps_path, self.obs_id, f"{self.obs_id}_{self.band}_{self.wafer}_weights.fits")
            elif self.kind == 'lscans':
                file = os.path.join(maps_path, self.obs_id, f"{self.obs_id}_{self.band}_{self.wafer}_weights_left_scans.fits")
            elif self.kind == 'rscans':
                file = os.path.join(maps_path, self.obs_id, f"{self.obs_id}_{self.band}_{self.wafer}_weights_right_scans.fits")
            self._weights = enmap.read_map(file)
        return self._weights

    def load_fits(self):
        
        models = {
            "EllipticGaussian": EllipticGaussian,
        }

        for model, fit_data in self.meta["fits"].items():
            if model in models:
                model_class = models[model]
                model_instance = model_class()
                
                params = fit_data.get("params", {})
                profile = fit_data.get("profile", {})
                chi2_red = fit_data.get("chi2_red", 1e9)
                setattr(model_instance, "chi2_red", chi2_red)
                
                for key, val in {**params, **profile}.items():
                    setattr(model_instance, key, val)
       
                setattr(self, model, model_instance)


def load_maps(
    blacklist: list = [],
    min_snr: float = -1,
    has_fits_for: str = '',
    clean_only: bool = False,
    **kwargs
    ):

    """
    Loads and filters maps as obsmap instances. 
    See 'procedure.md' for an explanation of what is considered as a "clean map".
    
    Parameters
    ----------
    blacklist: list(str)
        List of strings corresponding to obs_id's that you don't want read.
    min_snr: float
        Minimum SNR to load map. Useless if clean_only is set to True.
    has_fits_for: str
        Selects only maps with a valid fit under the specified model. Useless if clean_only is set to True.
    clean_only: bool
        Specifies whether or not to only load maps that are considered clean.
    wafer: str
        Include only the given wafer slot. The slots are relative and not absolute.
        Default value: '*' --> Selects all slots
        Allowed values: ['ws0', 'ws1', 'ws2', '*']
    band: str
        Include only the given band.
        Default value: '*' --> Selects all bands
        Allowed values: ['f090', 'f150', ...]
    time_class: str
        Include only the given time_class classification.
        Default value: '*' --> Selects all classifications.
        Allowed values: ['day', 'night', 'transition']
    season_class: str
        Include only the given alignment season classification. 
        Default value: '*' --> Selects maps from any season.
        Allowed values: ['first_alignment', 'second_alignment']
    
    Returns
    -------
    list of obsmap instances

    Notes
    -------
    In order to allow potential future code scaling, there are no restrictions on the values of 
    band, time_class, season_class or source. It's up to the user to make sure the values
    passed actually exist within the dictionary file. Despite this, the current documentation reflects 
    the current setup.

    The way of loading relies heavily on how the files are structured. If you change this structure, be prepared
    to make changes to this loading function.

    Currently clean_fits demands we have a good EllipticGaussian model.
    """

    wafer = kwargs.get("wafer", "*")
    band = kwargs.get("band", "*")
    time_class = kwargs.get("time_class", "*")
    season_class = kwargs.get("season_class", "*")

    with open(maps_dict, "rb") as f:
        encoded = f.read()

    maps_meta = msgspec.json.decode(encoded)
    
    obs_list = [d for d in os.listdir(maps_path) if os.path.isdir(os.path.join(maps_path, d)) if '.' not in d and d not in blacklist]
    maps = [] 

    if wafer == "*":
        wafer_options = ["ws0", "ws1", "ws2"]
    else:
        wafer_options = [wafer]

    if band == "*":
        band_options = ["f090", "f150"]
    else:
        band_options = [band]
    
    for obs_id in obs_list:

        if obs_id in blacklist:
            continue
    
        if time_class != '*':
            if maps_meta[obs_id]['time_info']['time_class'] != time_class:
                continue

        if season_class != '*':
            if maps_meta[obs_id]['time_info']['season_class'] != season_class:
                continue
                
                
        for case in product(wafer_options, band_options):
            wafer, band = case
            
            if maps_meta[obs_id][f"{band}-{wafer}"]["nomap"] == True:
                continue

            # Check if the wafer is supposed to have good data and that the snr isn't too low!
            if clean_only:
                min_snr = 15
                has_fits_for = 'EllipticGaussian'
                wafers = obs_id.split("_")[-1]
                waf = {"ws0": 0, "ws1":1, "ws2":2}
                if wafers[waf[wafer]] == '1':
                    if maps_meta[obs_id][f"{band}-{wafer}"]["full"]["snr"] < clean_map_min_snr:
                        continue
                    if 'EllipticGaussian' in list(maps_meta[obs_id][f"{band}-{wafer}"]["full"]["fits"].keys()):
                        if maps_meta[obs_id][f"{band}-{wafer}"]["full"]["fits"]['EllipticGaussian']["valid"] == True:
                            meta = maps_meta[obs_id][f"{band}-{wafer}"]["full"]["fits"]['EllipticGaussian']
                            ellip = np.abs(meta["profile"]["ellipticity"][0])
                            if ellip <= clean_map_max_ellip and meta["chi2_red"] <= clean_map_max_chi2_red:
                                maps.append(obsmap(obs_id, band, wafer))
                        else:
                            continue
                                
                    else:
                        continue
            else:
                wafers = obs_id.split("_")[-1]
                waf = {"ws0": 0, "ws1":1, "ws2":2}
                if wafers[waf[wafer]] == '1':
                    if maps_meta[obs_id][f"{band}-{wafer}"]["full"]["snr"] >= min_snr:
                        if has_fits_for != '':
                            if maps_meta[obs_id][f"{band}-{wafer}"]["full"]["fits"][has_fits_for]["valid"] == False:
                                continue

                
                        maps.append(obsmap(obs_id, band, wafer))
    return maps


def stack_maps(maps, 
               stackres=False, 
               stackres_mw = True, 
               no_weights = False, 
               pix2pix_weights = True, 
               mask = None, 
               abs_val=False):
    """
    Stack a set of 2D maps (obsmap instances) using
    optional weighing and optional residual-based stacking.

    Parameters
    ----------
    maps : list
        List of obsmap instances, each containing a valid `EllipticGaussian` attr.
    stackres : bool, default=False
        If True, stack residual maps (data - Gaussian fit) instead of the
        original maps.
    stackres_mw : bool, default=True
        When stacking residuals, use measurement weights (`1/W`) rather than
        the combined variance from fit uncertainty.
    no_weights : bool, default=False
        If True, stack all maps using equal weights.
    pix2pix_weights : bool, default=True
        If True, use pixel-level weights. If False, use the mean weight of
        each map.
    mask : ndarray of bool, optional
        Boolean mask array. Pixels where `mask` is False are set to NaN before
        stacking.
    abs_val : bool, default=False
        If True, take the absolute value of residuals before stacking.

    Returns
    -------
    stacked : ndarray
        Weighted (or unweighted) average of all centered maps, shape (180, 180).

    Notes
    -----
    - Each map is centered on its fitted Gaussian peak before stacking.
    - The residues are normalized based on the fitted Gaussian params.
    - Weight maps are scaled by amplitude squared and offset-corrected.
    - Zero-weight pixels are replaced by a small value (1e-12) to avoid division by zero.
    """
    maps2stack = []
    weights2stack = []
    for map_ in maps:
        T = map_.solved[0].copy()
        W = map_.weights[0][0].copy()
        
        W[W == 0] = 1e-12 
        ny, nx = T.shape
        y, x = np.mgrid[0:ny, 0:nx]
        coords = (y, x)
        popt = map_.EllipticGaussian.get_popt()
        pcov = map_.EllipticGaussian.get_pcov()
        center = (map_.EllipticGaussian.muy[0], map_.EllipticGaussian.mux[0])
        
        if not stackres:
            CT = center_map(T, center)
            CW = center_map(W, center)
        else:
            fit = EllipticGaussian().func(coords, *popt).reshape((180,180))
            fit_err = EllipticGaussian().func_err(coords, *popt, pcov)
            fit_err[fit_err == 0] = 1e-12
            if abs_val:
                residual = np.abs(T - fit)
            else:
                residual = T - fit
                #residual -= np.nanmin(residual)
            if stackres_mw:
                res_var = 1/W
            else:
                res_var = np.abs((1/W).reshape((180,180)) + fit_err.reshape((180,180))**2)
            
            res_w = 1/res_var
            CT = center_map(residual, center)
            CW = center_weights(res_w, center)

        if mask is not None:
            CT[~mask] = np.nan
            CW[~mask] = np.nan
            
        A = map_.EllipticGaussian.amp[0]
        CW -= map_.EllipticGaussian.offset[0]
        CW *= A**2
        CT /= A

        if not pix2pix_weights:
            CW = np.nanmean(CW.ravel())*np.ones(CW.shape)
        
        maps2stack.append(CT)
        weights2stack.append(CW)
    if no_weights:
        stacked = np.average(maps2stack, axis=0)
    else:
        stacked = np.average(maps2stack, axis=0, weights=weights2stack)
    return stacked
    

def fill_invalid_with_nearest(image):
    """
    Function to fill nan values in an array with values that are clones
    of the nearest non-nan pixel.

    Parameters
    ----------
    image : ndarray

    Returns
    -------
    filled : ndarray
    """
    # Identify invalid pixels
    mask = ~np.isfinite(image)
    
    # If all are valid, return as is
    if not np.any(mask):
        return image

    # Coordinates of nearest valid pixel for each invalid one
    nearest_valid_y, nearest_valid_x = np.where(~mask)
    nearest_y, nearest_x = distance_transform_edt(mask,
                                                  return_distances=False,
                                                  return_indices=True)
    filled = image[tuple(nearest_y), tuple(nearest_x)]
    return filled



def center_map(map2d, center_pixel):
    """
    Centers an array on the given center pixel through interpolation.

    Parameters
    ----------
    map2d : ndarray
    center_pixel: tuple

    Returns
    -------
    recentered_map : ndarray

    Notes
    -----
    - More testing is needed to check the effects of the interpolation.
    """
    ny, nx = map2d.shape
    cy, cx = center_pixel
    
    target_cy = (ny - 1) / 2
    target_cx = (nx - 1) / 2
    
    shift_y = target_cy - cy
    shift_x = target_cx - cx
    
    recentered_map = shift(map2d, shift=(shift_y, shift_x), order=3, mode='nearest')
    
    return recentered_map

def center_weights(weights, center_pixel):
    """
    Centers an array of weights on the given center pixel through interpolation.
    Identical to center_map but uses zero order interpolation to avoid
    changing the weight values.

    Parameters
    ----------
    weights : ndarray
    center_pixel: tuple

    Returns
    -------
    recentered_weights : ndarray

    Notes
    -----
    - More testing is needed to check the effects of (possible) interpolation.
    """
    ny, nx = weights.shape
    cy, cx = center_pixel

    target_cy = (ny - 1) / 2
    target_cx = (nx - 1) / 2

    shift_y = target_cy - cy
    shift_x = target_cx - cx

    # use order=0 to avoid interpolating weights!
    recentered_weights = shift(weights, shift=(shift_y, shift_x), order=0, mode='nearest')
    return recentered_weights
    

def sunrise_sunset_utc(lat, lon, height, date):
    """
    Calculate sunrise and sunset times in UTC for a given observatory location.

    Parameters
    ----------
    lat : float
        Latitude in degrees (+N)
    lon : float
        Longitude in degrees (+E)
    height : float
        Height above sea level in meters
    date : str
        Date in 'YYYY-MM-DD' format

    Returns
    -------
    sunrise_utc : astropy.time.Time
    sunset_utc : astropy.time.Time
    """
    
    # Observatory location
    location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)

    # Start and end times in UTC for that date
    t0 = Time(date + " 00:00:00")
    t1 = Time(date + " 23:59:59")
    
    # Sample times at 1-minute intervals
    delta_minutes = np.linspace(0, 24*60, 24*60)*u.min
    times = t0 + delta_minutes

    # AltAz frame for the location
    altaz = AltAz(obstime=times, location=location)

    # Compute Sun altitudes
    sun_altaz = get_sun(times).transform_to(altaz)
    altitudes = sun_altaz.alt

    # Find where Sun crosses horizon (0 deg)
    above_horizon = altitudes > 0*u.deg
    crossing = np.diff(above_horizon.astype(int))
    sunrise_idx = np.where(crossing == 1)[0]
    sunset_idx = np.where(crossing == -1)[0]

    if len(sunrise_idx) == 0 or len(sunset_idx) == 0:
        raise ValueError("Sun does not rise or set on this date at this location.")

    # Use first sunrise/sunset
    sunrise_utc = times[sunrise_idx[0]]
    sunset_utc = times[sunset_idx[0]]

    return sunrise_utc, sunset_utc


def get_obs_suntimes(ctime):

    """
    Helper function for the proccess of calculating suntimes
    """   
    dt_utc = datetime.utcfromtimestamp(ctime)
    dt_santiago = dt_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/Santiago"))
    
    # General coordinates of obs site
    lat = -22.9606052
    lon = -67.7905487
    height = 5190  # meters
    date = dt_santiago.strftime("%Y-%m-%d")

    sunrise_utc, sunset_utc = sunrise_sunset_utc(lat, lon, height, date)
    santiago_tz = ZoneInfo("America/Santiago")
    sunrise_santiago = sunrise_utc.to_datetime(timezone=santiago_tz)
    sunset_santiago = sunset_utc.to_datetime(timezone=santiago_tz)
    return sunrise_santiago, sunset_santiago, dt_santiago

def classify_obs_time(start, end, t, transition_hours=1):
    """
    Classify a time as 'day', 'night', or 'transition' based on start/end times.

    Parameters
    ----------
    start : datetime
        Start time (e.g., sunrise)
    end : datetime
        End time (e.g., sunset)
    t : datetime
        Time to classify
    transition_hours : float
        Window before/after boundaries considered 'transition' (default 1 hour)

    Returns
    -------
    str
        'day', 'night', or 'transition'
    """
  
    delta = timedelta(hours=transition_hours)
    
    # Transition windows
    start_transition_start = start - delta
    start_transition_end   = start + delta
    end_transition_start   = end - delta
    end_transition_end     = end + delta

    if start_transition_start <= t <= start_transition_end or \
       end_transition_start <= t <= end_transition_end:
        return 'transition'
    elif start < t < end:
        return 'day'
    else:
        return 'night'

def radial_profile(map2d, center=None, nbins=50, rmax=None, statistic='mean'):
    """
    Compute the radial profile of a 2D map by averaging values in radial bins.
    
    Parameters
    ----------
    map2d : 2D array
        Input map (intensity, temperature, etc.).
    center : tuple (y0, x0), optional
        Center coordinates. Default = center of the map.
    nbins : int, optional
        Number of radial bins.
    rmax : float, optional
        Maximum radius to consider (in pixels). Default = largest circle fitting in map.
    statistic : str, optional
        'mean' (default) or 'median' radial value per bin.
    
    Returns
    -------
    r_centers : 1D array
        Radial bin centers (in pixels).
    profile : 1D array
        Radial intensity profile (same units as map2d).
    """
    ny, nx = map2d.shape
    y, x = np.indices((ny, nx))

    if center is None:
        center = (ny / 2, nx / 2)
    y0, x0 = center

    # Compute distance of each pixel from the center
    r = np.sqrt((x - x0)**2 + (y - y0)**2)

    # Define bins
    if rmax is None:
        rmax = np.min([x0, y0, nx - x0, ny - y0])
    bins = np.linspace(0, rmax, nbins + 1)

    # Digitize distances
    bin_idx = np.digitize(r, bins)

    # Compute statistic per bin
    r_centers = 0.5 * (bins[1:] + bins[:-1])
    profile = np.zeros(nbins)

    for i in range(1, nbins + 1):
        vals = map2d[bin_idx == i]
        if len(vals) > 0:
            if statistic == 'median':
                profile[i - 1] = np.median(vals)
            else:
                profile[i - 1] = np.mean(vals)
        else:
            profile[i - 1] = np.nan

    return r_centers, profile

def zernike_radial(n, m, r):
    """
    Radial component of Zernike polynomial.
    DOCUMENTATION PENDING.
    """
    R = np.zeros_like(r)
    for k in range((n - abs(m)) // 2 + 1):
        c = (-1)**k * factorial(n - k)
        c /= factorial(k) * factorial((n + abs(m))//2 - k) * factorial((n - abs(m))//2 - k)
        R += c * r**(n - 2*k)
    return R

def zernike(n, m, r, theta):
    """
    Full Zernike polynomial Z_n^m(r, theta).
    DOCUMENTATION PENDING.
    """
    if m > 0:
        return zernike_radial(n, m, r) * np.cos(m * theta)
    elif m < 0:
        return zernike_radial(n, -m, r) * np.sin(-m * theta)
    else:
        return zernike_radial(n, 0, r)

def zernike_decompose(map2d, r, theta, mask, nmax=6):
    """
    Decompose map into zernike modes
    DOCUMENTATION PENDING.
    """
    coeffs = []
    for n in range(nmax+1):
        for m in range(-n, n+1, 2):
            Z = zernike(n, m, r, theta)*mask
            num = np.sum(map2d * Z)
            den = np.sum(Z * Z)
            a_nm = num / den
            coeffs.append((n, m, a_nm))

    ### GET CORRELATIONS HERE (TODO)
    return coeffs

    
class BaseModel:
    """
    Abstract base class for fit models.
    DOCUMENTATION PENDING.
    """

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def func(self, x, *params):
        """Override in subclass."""
        raise NotImplementedError

    def guess(self, mapdata):
        """Override in subclass to provide initial guesses."""
        raise NotImplementedError

    def bounds(self, mapdata):
        """Override in subclass to provide bounds."""
        raise NotImplementedError

    def profile(self, popt, pcov):
        """Override in subclass to provide profile related to model."""
        raise NotImplementedError

    def fit(self, mapdata, weights=None, weights_type="invvar", p0=None, bounds=None, mask=None):
        """Fit model to data. Don't override in subclass"""
        ny, nx = mapdata.shape
        y, x = np.mgrid[0:ny, 0:nx]
        coords = (y, x)
        
        if p0 is None:
            p0 = self.guess(mapdata)
        if bounds is None:
            bounds = self.bounds(mapdata)

        if mask is not None:
            mapdata = mapdata[mask]
            if weights is not None:
                weights = weights[mask]
            y = y[mask]
            x = x[mask]
            coords = (y,x)
        
        data = mapdata.ravel()
        tiny = 1e-12
        def weights_to_sigma(weights, weights_type="invvar"):
            if weights is None:
                return None
            w = np.array(weights, dtype=float)
            if weights_type == "invvar":
                sigma = 1.0 / np.sqrt(np.maximum(w, tiny))
            elif weights_type == "sigma":
                sigma = np.maximum(w, tiny)
            elif weights_type == "relative":
                sigma = 1.0 / np.maximum(w, tiny)
            else:
                raise ValueError("weights_type must be 'invvar', 'sigma', or 'relative'")
            return sigma
            
        sigma_map = weights_to_sigma(weights, weights_type)
        sigma = None if sigma_map is None else sigma_map.ravel()
        abs_sigma = True
        popt, pcov, infodict, mesg, ier  = curve_fit(
            self.func, coords, data,
            p0=p0, bounds=bounds, sigma=sigma,
            absolute_sigma=abs_sigma, maxfev=20000, full_output=True
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, pcov, infodict, mesg, ier 

    def eval_fit(self, coords):
        sig = inspect.signature(self.func)
        kwargs = {}
        for name in sig.parameters:
            if hasattr(self, name) and name != 'coords':
                kwargs[name] = getattr(self, name)[0] # 0 represents the actual value and 1 is error
        return self.func(coords,**kwargs).reshape((180,180))

    def get_popt(self):
        sig = inspect.signature(self.func)
        popt = []
        for name in sig.parameters:
            if hasattr(self, name) and name != 'coords':
                popt.append(getattr(self, name)[0])
        return popt

    def get_perr(self):
        sig = inspect.signature(self.func)
        perr = []
        for name in sig.parameters:
            if hasattr(self, name) and name != 'coords':
                perr.append(getattr(self, name)[1])
        return perr

class EllipticGaussian(BaseModel):
    """
    EllipticGaussian model for fitting. Best fit for primary lobes so far.
    DOCUMENTATION PENDING.

    Model is: 
    G(x,y) = A * exp(-(a*dx**2 + c*dy**2 + 2b*dx*dy))
    dx = x - mux
    dy = y - muy
    a = (cos(theta)^2 / 2 sigx^2) + (sin(theta)^2 / 2 sigy^2)
    b = (-sin(theta) cos(theta) / 2 sigx^2) + (sin(theta) cos(theta) / 2 sigy^2)
    a = (sin(theta)^2 / 2 sigx^2) + (cos(theta)^2 / 2 sigy^2)
    theta = 0.5 * arctan( 2b / (a -c)), (between -45 and 45 deg)
    
    """
    def func(self, coords, amp, mux, muy, sigx, sigy, theta, offset):
        y, x = coords
        fit = amp*np.exp(-(-mux + x)**2*(np.sin(theta)**2/(2*sigy**2) + np.cos(theta)**2/(2*sigx**2)) - (-mux + x)*(-muy + y)*(np.sin(theta)*np.cos(theta)/sigy**2 \
            - np.sin(theta)*np.cos(theta)/sigx**2) - (-muy + y)**2*(np.cos(theta)**2/(2*sigy**2) + np.sin(theta)**2/(2*sigx**2))) + offset
        return fit.ravel()

    def func_err(self, coords, amp, mux, muy, sigx, sigy, theta, offset, pcov):
        """
        Propagated error of the model based on the params it has.
        The calculation should be correct as it's been tested thoroughly.
        Returns pixel by pixel error and considers correlation.
        """
        y, x = coords
        dA = np.exp(-(-mux + x)**2*(np.sin(theta)**2/(2*sigy**2) + np.cos(theta)**2/(2*sigx**2)) - (-mux + x)*(-muy + y)*(np.sin(theta)*np.cos(theta)/sigy**2 - np.sin(theta)*np.cos(theta)/sigx**2) - (-muy + y)**2*(np.cos(theta)**2/(2*sigy**2) + np.sin(theta)**2/(2*sigx**2)))
        dMux = amp*(-(2*mux - 2*x)*(np.sin(theta)**2/(2*sigy**2) + np.cos(theta)**2/(2*sigx**2)) + (-muy + y)*(np.sin(theta)*np.cos(theta)/sigy**2 - np.sin(theta)*np.cos(theta)/sigx**2))*np.exp(-(-mux + x)**2*(np.sin(theta)**2/(2*sigy**2) + np.cos(theta)**2/(2*sigx**2)) - (-mux + x)*(-muy + y)*(np.sin(theta)*np.cos(theta)/sigy**2 - np.sin(theta)*np.cos(theta)/sigx**2) - (-muy + y)**2*(np.cos(theta)**2/(2*sigy**2) + np.sin(theta)**2/(2*sigx**2)))
        dMuy = amp*((-mux + x)*(np.sin(theta)*np.cos(theta)/sigy**2 - np.sin(theta)*np.cos(theta)/sigx**2) - (2*muy - 2*y)*(np.cos(theta)**2/(2*sigy**2) + np.sin(theta)**2/(2*sigx**2)))*np.exp(-(-mux + x)**2*(np.sin(theta)**2/(2*sigy**2) + np.cos(theta)**2/(2*sigx**2)) - (-mux + x)*(-muy + y)*(np.sin(theta)*np.cos(theta)/sigy**2 - np.sin(theta)*np.cos(theta)/sigx**2) - (-muy + y)**2*(np.cos(theta)**2/(2*sigy**2) + np.sin(theta)**2/(2*sigx**2)))
        dSigx = amp*((-mux + x)**2*np.cos(theta)**2/sigx**3 - 2*(-mux + x)*(-muy + y)*np.sin(theta)*np.cos(theta)/sigx**3 + (-muy + y)**2*np.sin(theta)**2/sigx**3)*np.exp(-(-mux + x)**2*(np.sin(theta)**2/(2*sigy**2) + np.cos(theta)**2/(2*sigx**2)) - (-mux + x)*(-muy + y)*(np.sin(theta)*np.cos(theta)/sigy**2 - np.sin(theta)*np.cos(theta)/sigx**2) - (-muy + y)**2*(np.cos(theta)**2/(2*sigy**2) + np.sin(theta)**2/(2*sigx**2)))
        dSigy = amp*((-mux + x)**2*np.sin(theta)**2/sigy**3 + 2*(-mux + x)*(-muy + y)*np.sin(theta)*np.cos(theta)/sigy**3 + (-muy + y)**2*np.cos(theta)**2/sigy**3)*np.exp(-(-mux + x)**2*(np.sin(theta)**2/(2*sigy**2) + np.cos(theta)**2/(2*sigx**2)) - (-mux + x)*(-muy + y)*(np.sin(theta)*np.cos(theta)/sigy**2 - np.sin(theta)*np.cos(theta)/sigx**2) - (-muy + y)**2*(np.cos(theta)**2/(2*sigy**2) + np.sin(theta)**2/(2*sigx**2)))
        dTheta = amp*(-(-mux + x)**2*(np.sin(theta)*np.cos(theta)/sigy**2 - np.sin(theta)*np.cos(theta)/sigx**2) - (-mux + x)*(-muy + y)*(-np.sin(theta)**2/sigy**2 + np.cos(theta)**2/sigy**2 + np.sin(theta)**2/sigx**2 - np.cos(theta)**2/sigx**2) - (-muy + y)**2*(-np.sin(theta)*np.cos(theta)/sigy**2 + np.sin(theta)*np.cos(theta)/sigx**2))*np.exp(-(-mux + x)**2*(np.sin(theta)**2/(2*sigy**2) + np.cos(theta)**2/(2*sigx**2)) - (-mux + x)*(-muy + y)*(np.sin(theta)*np.cos(theta)/sigy**2 - np.sin(theta)*np.cos(theta)/sigx**2) - (-muy + y)**2*(np.cos(theta)**2/(2*sigy**2) + np.sin(theta)**2/(2*sigx**2)))
        dOffset = 1*np.ones((180, 180))
        f_grad = [dA, dMux, dMuy, dSigx, dSigy, dTheta, dOffset]
        f_grad = [x.ravel() for x in f_grad]
        f_grad = np.array(list(map(list, zip(*f_grad))))
        var = [f_grad_x @ pcov @ f_grad_x.T for f_grad_x in f_grad]
        return np.sqrt(var)

    def get_pcov(self):
        if hasattr(self, "pcov"):
            pcov = np.array(getattr(self, "pcov")).reshape((7,7))
        else:
            print('No pcov found!')
        return pcov


    def guess(self, mapdata):
        ny, nx = mapdata.shape
        y, x = np.mgrid[0:ny, 0:nx]
        
        # subtract min so weights are nonnegative
        data = mapdata - np.min(mapdata)
        total = data.sum()
        if total <= 0:
            raise ValueError("Map has no positive content for moment estimation.")
        
        # weighted centroid
        mux = (x * data).sum() / total
        muy = (y * data).sum() / total
        
        # central second moments
        dx = x - mux
        dy = y - muy
        var_xx = (data * dx * dx).sum() / total
        var_yy = (data * dy * dy).sum() / total
        var_xy = (data * dx * dy).sum() / total
        
        sigx = np.sqrt(var_xx)
        sigy = np.sqrt(var_yy)
        theta = 0
        
        offset = np.min(mapdata)
        amp    = np.max(mapdata) - offset
        
        return [amp, mux, muy, sigx, sigy, theta, offset]

    def bounds(self, mapdata):
        ny, nx = mapdata.shape
        bounds = (
            [0, 0, 0, 1e-12, 1e-12, -np.pi/4, -np.inf], # we try to avoid fitting rho too close to bounds
            [np.inf, nx, ny, 324000, 324000, np.pi/4, np.inf]
            )
        return bounds

    def profile(self, popt, pcov):
        # FIRST ORDER APPROXIMATIONS
        amp, mux, muy, sigx, sigy, theta, offset = popt
        amp_err, mux_err, muy_err, sigx_err, sigy_err, theta_err, offset_err = np.sqrt(np.diag(pcov))
        var_sigx = sigx**2
        var_sigy = sigy**2

        def ellipticity_and_error_analytic(popt, pcov, idx_sigx=3, idx_sigy=4):
            """
            Analytic first-order error propagation for e = 1 - min/maj.
            popt: full parameter vector [amp, mux, muy, sigx, sigy, theta, offset]
            pcov: full covariance matrix from curve_fit
            idx_sigx, idx_sigy: indices of sigx and sigy in popt/pcov (default 3,4)
            Returns: (e, sigma_e, a, b)
              - e: ellipticity (1 - b/a) with a = major, b = minor
              - sigma_e: propagated 1-sigma error (0 if negative var numerical)
              - a,b: major, minor (for reference)
            """
            sigx = float(popt[idx_sigx])
            sigy = float(popt[idx_sigy])
        
            # pick major/minor and their pcov indices
            if sigx >= sigy:
                a, b = sigx, sigy
                ia, ib = idx_sigx, idx_sigy
            else:
                a, b = sigy, sigx
                ia, ib = idx_sigy, idx_sigx
        
            # 2x2 cov block
            var_a = pcov[ia, ia]
            var_b = pcov[ib, ib]
            cov_ab = pcov[ia, ib]
        
            # derivatives
            de_da = b / (a**2)
            de_db = -1.0 / a
        
            # var(e)
            var_e = (de_da**2) * var_a + (de_db**2) * var_b + 2 * de_da * de_db * cov_ab
        
            sigma_e = np.sqrt(var_e) if var_e > 0 else 0.0
            e = 1.0 - b / a if a != 0 else np.nan
            return e, sigma_e, a, b, var_a, var_b
        
        ellip, ellip_err, minor, major, var_min, var_maj = ellipticity_and_error_analytic(popt, pcov)
        
        
        fwhm_maj = 2 * np.sqrt(2 * np.log(2)) * major
        fwhm_min = 2 * np.sqrt(2 * np.log(2)) * minor
        fwhm_maj_err =  2 * np.sqrt(2 * np.log(2)) *np.sqrt(var_maj)
        fwhm_min_err =  2 * np.sqrt(2 * np.log(2)) *np.sqrt(var_min)
        
        results = {
            "ellipticity": (float(ellip), float(ellip_err)),
            "fwhm_maj": (float(fwhm_maj), float(fwhm_maj_err)),
            "fwhm_min": (float(fwhm_min), float(fwhm_min_err)),
            "fwhm_maj_phys": (float(fwhm_maj*0.002778), float(fwhm_maj_err*0.002778)), # Assuming the scale for maps remains the same...
            "fwhm_min_phys": (float(fwhm_min*0.002778), float(fwhm_min_err*0.002778)), # Assuming the scale for maps remains the same...
            "pcov": [float(x) for x in pcov.flatten()],
        }
        return results
    
    def chi2(self, mapshape, infodict):
        ny, nx = mapshape
        chi2 = np.sum(infodict["fvec"]**2)
        nu = nx*ny - 7
        chi2_red = chi2/nu
        return chi2, chi2_red


#################################################### OLD MODELS ######################################

class Parametric2DGaussian(BaseModel):
    """
    OLD MODEL THAT IS NO LONGER USED
    """
    def func(self, coords, amp, mux, muy, a, b, c, offset):
        y, x = coords
        dx = x - mux
        dy = y - muy
        gauss = amp * np.exp(-1*(a*dx**2 + 2*b*dx*dy + c*dy**2)) + offset
        return gauss.ravel()

    def func_err(self, coords, amp, mux, muy, a, b, c, offset):
        """
        It's assumed that all of the params come in tuples of (val, err)
        """
        y, x = coords
        
        err = np.sqrt((amp[0]**2*a[1]**2*(mux[0] - x)**4 + 4*amp[0]**2*b[1]**2*(mux[0] - x)**2*(muy[0] - y)**2 + amp[0]**2*c[1]**2*(muy[0] - y)**4 + 4*amp[0]**2*mux[1]**2*(a[0]*(mux[0] - x) + b[0]*(muy[0] - y))**2 + 4*amp[0]**2*muy[1]**2*(b[0]*(mux[0] - x) + c[0]*(muy[0] - y))**2 + amp[1]**2 + offset[1]**2*np.exp(2*a[0]*(-mux[0] + x)**2 + 4*b[0]*(-mux[0] + x)*(-muy[0] + y) + 2*c[0]*(-muy[0] + y)**2))*np.exp(-2*a[0]*(-mux[0] + x)**2 - 4*b[0]*(-mux[0] + x)*(-muy[0] + y) - 2*c[0]*(-muy[0] + y)**2))
        return err

    def guess(self, mapdata):
        ny, nx = mapdata.shape
        y, x = np.mgrid[0:ny, 0:nx]
        
        # subtract min so weights are nonnegative
        data = mapdata - np.min(mapdata)
        total = data.sum()
        if total <= 0:
            raise ValueError("Map has no positive content for moment estimation.")
        
        # weighted centroid
        mux = (x * data).sum() / total
        muy = (y * data).sum() / total
        
        # central second moments
        dx = x - mux
        dy = y - muy
        var_xx = (data * dx * dx).sum() / total
        var_yy = (data * dy * dy).sum() / total
        var_xy = (data * dx * dy).sum() / total
        
        sigx = np.sqrt(var_xx)
        sigy = np.sqrt(var_yy)
        theta = 0
        a = (np.cos(theta)**2 / (2*sigx**2)) + (np.sin(theta)**2)/(2*sigy**2)
        b = ((-np.cos(theta)*np.sin(theta)) / (2*sigx**2)) + (np.sin(theta)*np.cos(theta))/(2*sigy**2)
        c = (np.sin(theta)**2 / (2*sigx**2)) + (np.cos(theta)**2 / (2*sigy**2))
        
        
        offset = np.min(mapdata)
        amp = np.max(mapdata) - offset

        return [amp, mux, muy, a, b, c, offset]
    

    def bounds(self, mapdata):
        ny, nx = mapdata.shape
        bounds = (
            [0, 0, 0, 1e-12, 0, 1e-12, -np.pi/4],
            [np.inf, nx, ny, 324000, 324000,324000, np.pi/4] 
            )
        return bounds

    def profile(self, popt, pcov):
        amp, mux, muy, a, b, c, offset = popt
        amp_err, mux_err, muy_err, a_err, b_err, c_err, offset_err = np.sqrt(np.diag(pcov))
        theta = 0.5*np.arctan2(2*b,(a-c))
        sigx = 1/(np.sqrt(2)*(a*np.cos(theta)**2 + 2*b*np.cos(theta)*np.sin(theta)+c*np.sin(theta)**2))
        sigy = 1/(np.sqrt(2)*(a*np.sin(theta)**2 - 2*b*np.cos(theta)*np.sin(theta)+c*np.cos(theta)**2))

        theta_err = (1/((a-c)**2+4*b))*np.sqrt(b**2*(a_err**2+c_err**2)+(a-c)**2*b_err**2)
        var_sigx = (a_err**2*np.cos(theta)**4 + b_err**2*(1 - np.cos(4*theta))/2 + c_err**2*np.sin(theta)**4 + theta_err**2*(a*np.sin(2*theta) \
                    - 2*b*np.cos(2*theta) - c*np.sin(2*theta))**2)/(2*(a*np.cos(theta)**2 + b*np.sin(2*theta) + c*np.sin(theta)**2)**4)
        var_sigy = (a_err**2*np.sin(theta)**4 + b_err**2*(1 - np.cos(4*theta))/2 + c_err**2*np.cos(theta)**4 + theta_err**2*(a*np.sin(2*theta) \
                    - 2*b*np.cos(2*theta) - c*np.sin(2*theta))**2)/(2*(a*np.sin(theta)**2 - b*np.sin(2*theta) + c*np.cos(theta)**2)**4)
        
        ellip = 1 - sigy/sigx
        ellip_err = np.sqrt((sigy**2*var_sigx + sigx**2*var_sigy)/sigx**4)
        fwhm_x = 2 * np.sqrt(2 * np.log(2)) * sigx
        fwhm_y = 2 * np.sqrt(2 * np.log(2)) * sigy
        fwhm_x_err =  2 * np.sqrt(2 * np.log(2)) *np.sqrt(var_sigx)
        fwhm_y_err =  2 * np.sqrt(2 * np.log(2)) *np.sqrt(var_sigy)
        results = {
            "theta": (theta, theta_err),
            "sigx": (sigx, np.sqrt(var_sigx)),
            "sigy": (sigy, np.sqrt(var_sigy)),
            "ellipticity_al": (ellip, ellip_err),
            "fwhm_maj": (fwhm_x, fwhm_x_err),
            "fwhm_min": (fwhm_y, fwhm_y_err),
        }
        return results

    def chi2(self, mapshape, infodict):
        ny, nx = mapshape
        chi2 = np.sum(infodict["fvec"]**2)
        nu = nx*ny - 7
        chi2_red = chi2/nu
        return chi2, chi2_red
    
class AiryDisk(BaseModel):
    """
    OLD MODEL THAT IS NO LONGER USED
    """
    def func(self, coords, amplitude, x0, y0, radius, background):
        x, y = coords
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        
        # Avoid division by zero at the center
        z = np.pi * r / radius
        z[z == 0] = 1e-12
        
        airy = (2 * j1(z) / z)**2
        return (amplitude * airy + background).ravel()

    def guess(self, mapdata):
        ny, nx = mapdata.shape
        y, x = np.indices(mapdata.shape)
        
        # --- Background estimate (use edges or median) ---
        background = np.median(mapdata)
        
        # --- Peak location ---
        peak_idx = np.unravel_index(np.argmax(mapdata), mapdata.shape)
        y0, x0 = peak_idx
        
        # --- Peak amplitude (subtract background) ---
        amplitude = mapdata[y0, x0] - background
        
        # --- Estimate radius (first dark ring) ---
        # Use radial profile to find first minimum
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        r_flat = r.ravel()
        prof = (mapdata - background).ravel()
        
        # Bin radially
        nbins = 50
        r_bins = np.linspace(0, r_flat.max(), nbins)
        prof_mean = np.zeros(nbins - 1)
        for i in range(nbins - 1):
            mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
            if np.any(mask):
                prof_mean[i] = np.mean(prof[mask])
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        
        # Find first minimum beyond the center
        from scipy.signal import argrelextrema
        minima = argrelextrema(prof_mean, np.less)[0]
        if len(minima) > 0:
            radius = r_centers[minima[0]]
        else:
            # Fallback: assume first zero about 5% of image width
            radius = 0.05 * np.mean([nx, ny])
    
        return amplitude, x0, y0, radius, background
    
    def bounds(self, mapdata):
        ny, nx = mapdata.shape
        bounds = (
            [0, 0, 0, 0.5, np.min(mapdata)],
            [np.max(mapdata) * 2, nx, ny, min(nx, ny)/2, np.max(mapdata)]
            )
        return bounds

    def chi2(self, mapshape, infodict):
        ny, nx = mapshape
        chi2 = np.sum(infodict["fvec"]**2)
        nu = nx*ny - 5
        chi2_red = chi2/nu
        return chi2, chi2_red

class EllipticalAiryDisk(BaseModel):
    """
    OLD MODEL THAT IS NO LONGER USED
    """
    def func(self, coords, I0, mux, muy, sigma_maj, sigma_min, theta, R0, offset):
        y, x = coords
        dx = x - mux
        dy = y - muy

        # rotate coordinates
        x_rot =  dx * np.cos(theta) + dy * np.sin(theta)
        y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
    
        # scale for ellipticity
        x_scaled = x_rot / sigma_maj
        y_scaled = y_rot / sigma_min
    
        # radial coordinate
        r = np.sqrt(x_scaled**2 + y_scaled**2) / R0
        r[r == 0] = 1e-12  # avoid division by zero at center
    
        # Airy function
        airy = (2 * j1(np.pi * r) / (np.pi * r))**2

        return (I0 * airy + offset).ravel()

    def chi2(self, mapshape, infodict):
        ny, nx = mapshape
        chi2 = np.sum(infodict["fvec"]**2)
        nu = nx*ny - 8
        chi2_red = chi2/nu
        return chi2, chi2_red

    def guess(self, mapdata):
        ny, nx = mapdata.shape
        y, x = np.indices(mapdata.shape)
        
        # --- Background estimate (use edges or median) ---
        background = np.median(mapdata)
        
        # --- Peak location ---
        peak_idx = np.unravel_index(np.argmax(mapdata), mapdata.shape)
        y0, x0 = peak_idx
        
        # --- Peak amplitude (subtract background) ---
        amplitude = mapdata[y0, x0] - background
        
        # --- Estimate radius (first dark ring) ---
        # Use radial profile to find first minimum
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        r_flat = r.ravel()
        prof = (mapdata - background).ravel()
        
        # Bin radially
        nbins = 50
        r_bins = np.linspace(0, r_flat.max(), nbins)
        prof_mean = np.zeros(nbins - 1)
        for i in range(nbins - 1):
            mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
            if np.any(mask):
                prof_mean[i] = np.mean(prof[mask])
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        
        # Find first minimum beyond the center
        from scipy.signal import argrelextrema
        minima = argrelextrema(prof_mean, np.less)[0]
        if len(minima) > 0:
            radius = r_centers[minima[0]]
        else:
            # Fallback: assume first zero about 5% of image width
            radius = 0.05 * np.mean([nx, ny])

        sigx, sigy, theta = 1, 1, 0
        return [amplitude, x0, y0, sigx, sigy, theta, radius, background]
    
    def bounds(self, mapdata):
        ny, nx = mapdata.shape
        bounds = (
            [0, 0, 0, 0, 0,0, 1e-6, -np.inf],
            [np.inf, nx, ny, nx,ny, np.pi, np.inf, np.inf]
            )
        return bounds