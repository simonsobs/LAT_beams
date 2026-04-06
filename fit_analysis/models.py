import numpy as np
import h5py
from pixell import enmap
import os
from scipy.optimize import curve_fit
import traceback
import inspect


class BaseModel:
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

    def fit(self, mapdata, sigma=None, p0=None, bounds=None, mask=None, abs_sigma = False):
        """Fit model to data. Optional override in subclass"""
        ny, nx = mapdata.shape
        y, x = np.mgrid[0:ny, 0:nx]
        coords = (y, x)
        
        if p0 is None:
            p0 = self.guess(mapdata)
        if bounds is None:
            bounds = self.bounds(mapdata)

        if mask is not None:
            mask = mask.astype(bool)  
            data = mapdata[mask]
            coords = (y[mask], x[mask])
        
        coords = (coords[0].ravel(), coords[1].ravel())

        if sigma is not None:
            sigma = sigma.ravel()
            
        popt, pcov, infodict, mesg, ier  = curve_fit(
            self.func, coords, data.ravel(),
            p0=p0, bounds=bounds, sigma=sigma,
            absolute_sigma=abs_sigma, maxfev=20000, full_output=True
        )
        
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, pcov, infodict, mesg, ier 

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

    def fit(self, mapdata, sigma=None, p0=None, bounds=None, mask=None, abs_sigma = False):
        ny, nx = mapdata.shape
        y, x = np.mgrid[0:ny, 0:nx]
        coords = (y, x)
        
        if p0 is None:
            p0 = self.guess(mapdata)
        if bounds is None:
            bounds = self.bounds(mapdata)
    
        if mask is not None:
            mask = mask.astype(bool)  
            data = mapdata[mask]
            coords = (y[mask], x[mask])
        else:
            data = mapdata

        coords = (coords[0].ravel(), coords[1].ravel())

        if sigma is not None:
            sigma = sigma.ravel()
            
        popt, pcov, infodict, mesg, ier  = curve_fit(
            self.func, coords, data.ravel(),
            p0=p0, bounds=bounds, sigma=sigma,
            absolute_sigma=abs_sigma, maxfev=20000, full_output=True
        )
        
        amp, mux, muy, ellip, theta_fwhm, phi, offset = popt
        if ellip < 0:
                ellip = np.abs(ellip)
                phi += np.pi/2
        popt = amp, mux, muy, ellip, theta_fwhm, phi, offset
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, pcov, infodict, mesg, ier 

    def func(self, coords, amp, mux, muy, ellip, theta_fwhm, phi, offset):

        sigma_mean = theta_fwhm / np.sqrt(8.0 * np.log(2.0))
        r = (1.0 + ellip) / (1.0 - ellip)
        sigma_maj = sigma_mean * np.sqrt(r)
        sigma_min = sigma_mean / np.sqrt(r)

        y, x = coords
        dx = x - mux
        dy = y - muy

        xp =  dx * np.cos(phi) + dy * np.sin(phi)
        yp = -dx * np.sin(phi) + dy * np.cos(phi)
    
        g = amp * np.exp(-0.5 * ( (xp/sigma_maj)**2 + (yp/sigma_min)**2 )) + offset
        return g.ravel()

    def profile(self, popt, pcov):
        
        results = {
            "pcov": [float(x) for x in pcov.flatten()],
        }
        return results
    
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
        mux = 90
        muy = 90
        
        # central second moments
        dx = x - mux
        dy = y - muy
        var_xx = (data * dx * dx).sum() / total
        var_yy = (data * dy * dy).sum() / total
        var_xy = (data * dx * dy).sum() / total
        
        sigx = np.sqrt(var_xx)
        sigy = np.sqrt(var_yy)
        sigmean = np.sqrt(sigx*sigy) + 1e-12
        ellip = (sigx-sigy)/(sigx+sigy)
        theta = 0
        
        offset = np.min(mapdata)
        amp    = np.max(mapdata) - offset
        
        return [amp, mux, muy, ellip, sigmean, theta, offset]

    def bounds(self, mapdata):
        ny, nx = mapdata.shape
        bounds = (
            [0, 0, 0, -1, 1e-12, -np.pi/2, -np.inf], # we try to avoid fitting rho too close to bounds
            [np.inf, nx, ny, 1, 324000, np.pi/2, np.inf]
            )
        return bounds
    
    def chi2(self, pixels, infodict):
        chi2 = np.sum(infodict["fvec"]**2)
        nu = pixels - 7
        chi2_red = chi2/nu
        return chi2, chi2_red