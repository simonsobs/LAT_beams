import numpy as np


# TODO: ellipticity
def gauss_grad(x, y, x0, y0, amp, fwhm):
    dx = x - x0
    dy = y - y0
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    var = sigma**2
    gauss = amp * np.exp(-0.5 * (dx**2 + dy**2) / (var))

    dfdx = -(dx / var) * gauss
    dfdy = -(dy / var) * gauss
    dady = gauss / amp
    dsdy = ((dx**2 + dy**2) / sigma**3) * gauss
    grad = np.array([dfdx, dfdy, dady, dsdy])

    return gauss, grad


def invsafe(matrix, thresh: float = 1e-14):
    """
    Safe SVD based psuedo-inversion of the matrix.
    This zeros out modes that are too small when inverting.
    Use with caution in cases where you really care about what the inverse is.
    """
    u, s, v = np.linalg.svd(matrix, False)
    s_inv = np.array(np.where(np.abs(s) < thresh * np.max(s), 0, 1 / s))

    return np.dot(np.transpose(v), np.dot(np.diag(s_inv), np.transpose(u)))


def invscale(matrix, thresh: float = 1e-14):
    """
    Invert and rescale a matrix by the diagonal.
    This uses `invsafe` for the inversion.

    """
    diag = np.diag(matrix)
    vec = np.array(np.where(diag != 0, 1.0 / np.sqrt(np.abs(diag)), 1e-10))
    mm = np.outer(vec, vec)

    return mm * invsafe(mm * matrix, thresh)


def objective(aman, pars, filter_tod):
    npar = len(pars)
    chisq = np.array(0)
    grad = np.zeros(npar)
    curve = np.zeros((npar, npar))

    pred_dat, grad_dat = gauss_grad(aman.xi, aman.eta, *pars)

    aman.resid = aman.signal - pred_dat
    aman = filter_tod(aman, signal_name="resid")
    chisq = np.sum(aman.resid * aman.resid_filt)
    grad_filt = np.zeros_like(grad_dat)
    for i in range(npar):
        aman.grad_buf[0] = grad_dat[i]
        aman = filter_tod(aman, signal_name="grad_buf")
        grad_filt[i] = aman.grad_buf_filt.copy().ravel()
    grad_filt = np.reshape(grad_filt, (npar, -1))
    grad_dat = np.reshape(grad_dat, (npar, -1))
    resid = aman.resid.ravel()
    grad = np.dot(grad_filt, np.transpose(resid))
    curve = np.dot(grad_filt, np.transpose(grad_dat))

    return chisq, grad, curve


def prior_pars(pars, priors):
    prior_l, prior_u = priors
    at_edge_l = pars <= prior_l
    at_edge_u = pars >= prior_u
    pars = np.where(at_edge_l, prior_l, pars)
    pars = np.where(at_edge_u, prior_u, pars)

    return pars


def lm_fitter(aman, filter_tod, init_pars, bounds, max_iters=20, chitol=1e-5):
    priors = np.array(
        [[bound[i] for bound in bounds] for i in range(2)]
    )  # Convert from scipy opt bounds to flat priors
    pars = prior_pars(np.array(init_pars), priors)
    chisq, grad, curve = objective(aman, pars, filter_tod)
    errs = np.inf + np.zeros_like(pars)
    delta_chisq = np.inf
    lmd = 0
    i = 0

    for i in range(max_iters):
        if delta_chisq < chitol:
            break
        curve_use = curve + (lmd * np.diag(np.diag(curve)))
        # Get the step
        step = np.dot(invscale(curve_use), grad)
        new_pars = prior_pars(pars + step, priors)
        # Get errs
        errs = np.sqrt(np.diag(invscale(curve_use)))
        # Now lets get an updated model
        new_chisq, new_grad, new_curve = objective(aman, new_pars, filter_tod)
        new_delta_chisq = chisq - new_chisq

        if new_delta_chisq > 0:
            pars, chisq, grad, curve, delta_chisq = (
                new_pars,
                new_chisq,
                new_grad,
                new_curve,
                new_delta_chisq,
            )
            if lmd < 0.2:
                lmd = 0
            else:
                lmd /= np.sqrt(2)
        else:
            if lmd == 0:
                lmd = 1
            else:
                lmd *= 2

    return pars, errs, i, delta_chisq
