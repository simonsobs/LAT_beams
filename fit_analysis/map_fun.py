import numpy as np
import h5py
from pixell import enmap
import os
from scipy.optimize import curve_fit
import models
from tqdm.notebook import tqdm
import inspect
import traceback
from scipy.ndimage import fourier_shift
from scipy.ndimage import distance_transform_edt
from scipy.special import factorial


def load_h5(filename):
    out = {}
    with h5py.File(filename, "r") as f:
        for name in f:
            g = f[name]
            d = dict(g.attrs)

            for key in g:
                d[key] = g[key][:]

            out[name] = d   # <-- key = item name
    return out


class obsmap:
    def __init__(self, meta: dict):
        self.__dict__.update(meta)
        self.meta_name = f"obs_{self.obs_id}_{self.tube_slot}_{self.wafers_present}_{self.band}_{self.wafer}"
        self._solved = None
        self._weights = None
        self._binned = None

    @property
    def solved(self):
        if self._solved is None:
            try:
                data = enmap.read_map(self.path_pattern+'_solved.fits')
                self._solved = data
            except TypeError:
                print(f"DATA LOAD FAILED: {self.path_pattern} has invalid data!")
                return None
                
        return self._solved.copy()
        

    @solved.setter
    def solved(self, value):
        raise AttributeError("data is read-only")

    @property
    def weights(self):
        if self._weights is None:
            data = enmap.read_map(self.path_pattern+'_weights.fits')
            self._weights = data
        return self._weights.copy()

    @weights.setter
    def weights(self, value):
        raise AttributeError("data is read-only")

    @property
    def binned(self):
        if self._binned is None:
            data = enmap.read_map(self.path_pattern+'_binned.fits')
            self._binned = data
        return self._binned.copy()

    @binned.setter
    def binned(self, value):
        raise AttributeError("data is read-only")

    def load_fit(self, model, fitmeta):
        model_instance = model()
        fitinfo = fitmeta[self.meta_name]
        popt = fitinfo['params']
        pcov = fitinfo['pcov']
        perr = fitinfo['perr']
        fvec = fitinfo['fvec']

        params = list(inspect.signature(model().func).parameters.keys())[1:]
        dikt = {}
        for i, param in enumerate(params):
            dikt.update({param: (float(popt[i]), float(perr[i]))})
        for key, val in {**dikt}.items():
            setattr(model_instance, key, val)
        setattr(model_instance, "pcov", pcov)
        setattr(model_instance, "fvec", fvec)
        setattr(model_instance, "popt", popt)
        setattr(model_instance, "perr", perr)
        setattr(self, model.__name__, model_instance)

def fit_maps(maps, model: models.BaseModel, mask_method = None, sigma_method = None, comp = 'T'):
    
    no_good = 0
    results = []
    for i, map2use in enumerate(tqdm(maps, desc=f"Fitting maps w/ {model.__name__} model")):
        compid = {"T":0 ,"Q":1, "U":2}[comp]
        try:
           
            T = map2use.solved[compid]
        except Exception as e:
            print(e)
            no_good +=1 
            results.append([None, None, None, None, "Invalid data!", None])
            continue
            
        sigma = None
        sigma_mask = None
        mask = None
        
        if sigma_method is not None:
            # Method must return mask of all the pixels to use
            sigma, sigma_mask = sigma_method(map2use)
        if mask_method is not None:
            mask = mask_method(map2use)

        if sigma_mask is not None:
            if mask is not None:
                mask = mask & sigma_mask
            else:
                mask = sigma_mask
    
        ny,nx = T.shape
        y, x = np.mgrid[0:ny, 0:nx]
        coords = (y, x)

        try:
            popt, perr, pcov, infodict, mesg, ier = model().fit(mapdata=T, sigma=sigma, mask=mask)
        except Exception as e:
            print(traceback.format_exc())
            no_good += 1
            results.append([None, None, None, None, str(e), None])
            continue
        
        if int(ier) not in [1,2,3,4]:
            no_good +=1 
            results.append([popt, perr, pcov, infodict, mesg, ier])
            continue

        results.append([popt, perr, pcov, infodict, mesg, ier])
        
    print(f"{len(maps) - no_good} maps have been fitted out of {len(maps)} total maps.")
    return results

def fit_signals(signals, model: models.BaseModel, mask_method = None, sigma_method = None):
    
    no_good = 0
    results = []
    for i, signal in enumerate(tqdm(signals, desc=f"Fitting maps w/ {model.__name__} model")):

        sigma = None
        sigma_mask = None
        mask = None
        
        if sigma_method is not None:
            # Method must return mask of all the pixels to use
            sigma, sigma_mask = sigma_method(signal)
        if mask_method is not None:
            mask = mask_method(signal)

        if sigma_mask is not None:
            if mask is not None:
                mask = mask & sigma_mask
            else:
                mask = sigma_mask
    
        ny,nx = signal.shape
        y, x = np.mgrid[0:ny, 0:nx]
        coords = (y, x)

        try:
            popt, perr, pcov, infodict, mesg, ier = model().fit(mapdata=signal, sigma=sigma, mask=mask)
        except Exception as e:
            print(traceback.format_exc())
            no_good += 1
            results.append([None, None, None, None, str(e), None])
            continue
        
        if int(ier) not in [1,2,3,4]:
            no_good +=1 
            results.append([popt, perr, pcov, infodict, mesg, ier])
            continue

        results.append([popt, perr, pcov, infodict, mesg, ier])
        
    print(f"{len(signals) - no_good} signals have been fitted out of {len(signals)} total maps.")
    return results


def weights_sigma_method(map2use):
    W = map2use.weights[0][0]
    sigma = 1/np.sqrt(W[W != 0])
    sigma_mask = W != 0
    return sigma, sigma_mask

def save_h5(filename, data):
    """
    data: dict[str, dict]
          outer key   -> group name
          inner dict  -> attrs + datasets
    """
    with h5py.File(filename, 'w') as f:
        for name, d in data.items():
            # overwrite group if it already exists
            if name in f:
                del f[name]

            g = f.create_group(name)

            for key, val in d.items():
                if isinstance(val, np.ndarray):
                    g.create_dataset(
                        key,
                        data=val,
                        compression="gzip"
                    )
                else:
                    g.attrs[key] = val

def query_maps(data, **criteria):
    """
    data: dict[str, dict]
    returns: list of matching item IDs
    """
    matches = []

    for item_id, d in data.items():
        if all(d.get(k) == v for k, v in criteria.items()):
            matches.append(item_id)

    return matches

def save_map_fits(savename, maps, results):
    fitsdict = dict()
    for i, result in enumerate(results):
        popt, perr, pcov, infodict, mesg, ier = result
        
        map2use = maps[i]
        meta_name = map2use.meta_name
       
        fitsdict[meta_name] = {
            "params":popt,
            "perr": perr,
            "pcov":pcov,
            "fvec": infodict["fvec"]
        }
    
    save_h5(savename, fitsdict)



def save_signal_fits(savename, results, keys, just_format=False):
    fitsdict = dict()
    for i, result in enumerate(results):
        popt, perr, pcov, infodict, mesg, ier = result
       
        fitsdict[keys[i]] = {
            "params":popt,
            "perr": perr,
            "pcov":pcov,
            "fvec": infodict["fvec"]
        }
    if just_format:
        return fitsdict
    
    save_h5(savename, fitsdict)

def format_results(results, keyname):
    new_entry = save_signal_fits("", results, keys=[keyname], just_format=True)
    return new_entry


def norm_center_map(image, amp = None, offset = None, center=None, target= None):
    if amp is not None and offset is None:
        normed = (image-offset)/amp 
    else:
        normed = image
    if center is None:
        raise Exception("Center not passed!")
    ny, nx = normed.shape
    cy, cx = center

    if target is None:
        target_cy = ny//2
        target_cx = nx//2
    else:
        target_cy, target_cx = target

    shift_y = target_cy - cy
    shift_x = target_cx - cx
    shift = (shift_y, shift_x)

    F = np.fft.fftn(normed)

    F_shifted = fourier_shift(F, shift)

    shifted_image = np.fft.ifftn(F_shifted).real
    return shifted_image

def coadd_maps(mapdata, weights = None):
    coadd = np.average(mapdata, weights=weights, axis=0)
    return coadd

def get_coords(arr):
    ny,nx = arr.shape
    y, x = np.mgrid[0:ny, 0:nx]
    coords = (y, x)
    return coords

def zernike_index(n, m):
    if abs(m) > n or (n - m) % 2 != 0:
        raise ValueError("Invalid (n,m) combination for Zernike ordering.")
    return n*(n + 1)//2 + (m + n)//2

def zernike_nm_from_index(idx):
    n = int((np.sqrt(8*idx + 1) - 1)//2)
    offset = n*(n+1)//2
    while offset + n < idx:  # handle boundary
        n += 1
        offset = n*(n+1)//2
    m = 2*(idx - offset) - n
    return int(n), int(m)

def scale_coeffs(coeffs, eps):
    nmax = int((-3 + np.sqrt(8 * len(coeffs) + 1)) / 2)
    if eps >= 1:
        raise Exception("scale factor eps must be less than 1!")

    def dais_formula(nm):
        n, m = nm
        sum_part = np.sum([
        coeffs[zernike_index(n + 2*i, m)] *
        np.sum([
            ((-1)**(i + j) * factorial(n + i + j)) /
            (factorial(n + j + 1) * factorial(i - j) * factorial(j)) *
            eps**(2 * j)
            for j in range(0, i + 1)
        ])
        for i in range(1, (nmax - n)//2 + 1)
        ])
    
        scaled_coeff = (eps**n)*((n+1)*sum_part + coeffs[zernike_index(n,m)])
        return scaled_coeff
    scaled = np.array(list(map(dais_formula, [(n,m) for n in range(nmax+1) for m in range(-n, n + 1, 2)])))
    return scaled

def rotate_coeffs(coeffs, phi):
    nmax = int((-3 + np.sqrt(8 * len(coeffs) + 1)) / 2)
    def remap_cfs(n,m,cf,phi):
        if m > 0:
            
            return cf[zernike_index(n,m)]*np.cos(m*phi)-cf[zernike_index(n,-m)]*np.sin(m*phi)
        elif m < 0:
            return cf[zernike_index(n,-m)]*np.sin(m*phi)+cf[zernike_index(n,m)]*np.cos(m*phi)
        else:
            return cf[zernike_index(n,m)]
        
    new_coeffs = np.array([remap_cfs(n,m,coeffs,phi) for n in range(nmax+1) for m in range(-n, n + 1, 2)])
    return new_coeffs

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
        center = 90, 90
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

def R_nm(n, m, rho):
    """Radial Zernike polynomial R_n^m(rho)"""
    R = np.zeros_like(rho)
    for k in range((n - abs(m))//2 + 1):
        coeff = ((-1)**k * factorial(n - k)) / \
                (factorial(k) * factorial((n + abs(m))//2 - k) * factorial((n - abs(m))//2 - k))
        R += coeff * rho**(n - 2*k)
    return R

def Z_nm(n, m, rho, phi):
    """Full Zernike polynomial Z_n^m(rho, phi)"""
    R = R_nm(n, m, rho)
    if m > 0:
        return R * np.cos(m * phi)
    elif m < 0:
        return R * np.sin(-m * phi)
    else:
        return R

def get_zernike_modes(nmax=30, mask_radius=30, coords=None, center=None):
    muy, mux = center
    y, x = coords
    r = np.sqrt((x - mux)**2 + (y - muy)**2)
    theta = np.arctan2(y - muy, x - mux)
    radius = mask_radius
    mask = r <= radius
    r_max = np.max(r[mask])
    r_norm = r / r_max 
    Z = np.array([Z_nm(n, m, r_norm, theta)*mask for n in range(nmax+1) for m in range(-n, n + 1, 2)])
    return Z

def radial_mask(coords, center, radius):
    y,x = coords
    muy, mux = center
    r = np.sqrt((x - mux)**2 + (y - muy)**2)
    theta = np.arctan2(y - muy, x - mux)
    mask = r <= radius
    return mask

def get_zernike_coeffs(residuals, center, radius, zernike_modes, get_var=False):
    coords = get_coords(residuals)
    y,x = coords
    muy, mux = center
    r = np.sqrt((x - mux)**2 + (y - muy)**2)
    theta = np.arctan2(y - muy, x - mux)
    mask = r <= radius
    r_max = np.max(r[mask])
    r_norm = r / r_max 
    G = residuals*mask
    Z_norm = np.sum(np.multiply(zernike_modes, zernike_modes), axis=(1,2))
    coeffs = np.sum(np.multiply(G, zernike_modes), axis=(1,2))/Z_norm
    if not get_var:
        return coeffs
    Gdelta = G - np.sum(coeffs[:, None, None]*zernike_modes, axis=0)
    sigma2 = np.sum(Gdelta**2) / (G.size - coeffs.size)
    cov_alpha = sigma2 / Z_norm
    return coeffs, cov_alpha

def zernike_abs_index(n, m):
    nmax = n+1
    nm_list = np.array([(n,m) for n in range(nmax+1) for m in range(n%2, n + 1, 2)])
    # Boolean mask where both n and m match
    mask = (nm_list[:,0] == n) & (nm_list[:,1] == m)
    return np.argmax(mask)

def filter_zernike_modes(coeffs, sig_lim=1, angular_orders = None, radial_orders = None, auto_modes = False, abs_indexes = False, ignore_thresh = False):
    # Implement auto modes later
    cf_abs = []
    nmax = int((-3 + (1 + 8*len(coeffs))**0.5) / 2)
    nm_list = np.array([(n,m) for n in range(nmax+1) for m in range(n%2, n + 1, 2)])
    for n, m in nm_list:
        cf_abs.append(np.sqrt(coeffs[zernike_index(n,m)]**2+coeffs[zernike_index(n,m)]**2))
    
    idx = []
    if not ignore_thresh:
        selection = np.where(cf_abs > sig_lim*np.std(cf_abs))[0]
    else:
        selection = np.arange(len(cf_abs))
        
    for n,m in nm_list[selection]:
    
        if angular_orders is not None:
            if abs(m) not in angular_orders:
                continue
        if radial_orders is not None:
            if n not in radial_orders:
                continue

        if not abs_indexes:
            idx.append(zernike_index(n,m))
            if m != 0:
                idx.append(zernike_index(n,-m))
        else:
            idx.append(zernike_abs_index(n,m))
    return idx



def get_abs_coeffs(coeffs):
    cf = []
    nmax = int((-3 + (1 + 8*len(coeffs))**0.5) / 2)
    nm_list = [(n,m) for n in range(nmax+1) for m in range(n%2, n + 1, 2)]
    for n, m in nm_list:
        cf.append(np.sqrt(coeffs[zernike_index(n,m)]**2+coeffs[zernike_index(n,m)]**2))
    return np.array(cf)

def get_m_angle(cf, m, nmax=31):
    weights, amps = [], []
    for n in range(nmax):
        try:
            zernike_index(n,m)
        except:
            continue
        w = np.sqrt(cf[zernike_index(n,m)]**2 + cf[zernike_index(n,-m)]**2)
        A = cf[zernike_index(n,m)]+ 1j*cf[zernike_index(n,-m)]
        weights.append(w)
        amps.append(A)
    Am = np.sum(np.array(weights)*np.array(amps))
    angle = (1/m)*np.arctan2(np.imag(Am), np.real(Am))
    return angle % (np.pi / 4)