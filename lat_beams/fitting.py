"""
Just fitting a single TOD for now, can genralize later (but maybe just do a WITCH interface for that)
Maybe should add priors
"""
import numpy as np
from . import noise as nn

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

def fit_aman(aman, init_pars, n_iters, chitol):
    chisq, grad, curve = objective(aman, init_pars)
    delta_chisq = np.inf
    lmd = 0.
    pars = np.array(init_pars)
    for i in range(n_iters):
        curve_use = curve.at[:].add(lmd * np.diag(np.diag(curve)))
        # Get the step
        step = np.dot(invscale(curve_use), grad)
        new_pars = pars + step
        # Get errs
        errs = np.sqrt(np.diag(invscale(curve_use)))
        # Now lets get an updated model
        new_chisq, new_grad, new_curve = objective(aman, init_pars)

        new_delta_chisq = chisq - new_chisq
        if new_delta_chisq > 0:
            if lmd < .2:
                lmd = 0.
            else:
                lmd /= np.sqrt(2)
            chisq = new_chisq
            grad = new_grad
            curve = new_curve
            pars = new_pars
            delta_chisq = new_delta_chisq
        else:
            lmd *= 2
            if lmd == 0:
                lmd = 1
            continue
        if delta_chisq<chitol:
            break
