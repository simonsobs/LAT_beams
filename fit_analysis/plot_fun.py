import numpy as np
import matplotlib.pyplot as plt
from map_fun import radial_profile

def compare_plot(
    data1,
    data2,
    log = '00',
    figsize=(10,4),
    title1 = None,
    title2 = None,
    cmap1 = 'viridis',
    cmap2 = 'viridis'
):

    if log[0] == 0:
        img1 = data1
    else:
        img1 = np.log10(np.abs(data1))
    
    if log[1] == 0:
        img2 = data2
    else:
        img2 = np.log10(np.abs(data2))
        
    fig, ax = plt.subplots(1,2, figsize=figsize)
    im1 = ax[0].imshow(img1, cmap=cmap1)
    plt.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(img2, cmap=cmap2)
    plt.colorbar(im2, ax=ax[1])

    if title1 is not None:
        ax[0].set_title(title1)
    if title2 is not None:
        ax[1].set_title(title2)
    

    fig.tight_layout()
    plt.show()


def pcolormesh_plot(map_, mesh=None, log=True, title=None, mask=None, mask_radius=None, zoom=None, vmin=None, vmax=None):
    """
    Auxiliary function for plotting purposes with optional mask and zoom.

    Parameters
    ----------
    map_ : 2D array
        Data map to plot.
    mesh : tuple of 2D arrays, optional
        (x_vals, y_vals) meshgrid for coordinates.
    log : bool, optional
        Whether to plot log10 of data.
    title : str, optional
        Title for the plot.
    mask : 2D bool array, optional
        Boolean mask to show only a region (True=show, False=hide).
    zoom : tuple or list, optional
        (xmin, xmax, ymin, ymax) region to zoom into (in same units as mesh).
    """
    # Handle mesh
    if mesh is not None and mesh != []:
        x_vals, y_vals = mesh
    else:
        x_vals, y_vals = np.mgrid[0:180, 0:180]

    if mask_radius is not None:
        dist = np.sqrt((x_vals - 90)**2 + (y_vals - 90)**2)
        mask = dist <= mask_radius
        map_data = np.where(mask, map_, np.nan)

    # Apply mask
    if mask is not None:
        map_data = np.where(mask, map_, np.nan)
    else:
        map_data = map_

    # Apply log scaling
    if log:
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.log10(map_data)
    else:
        data = map_data

    # Plot
    if vmin is not None and vmax is not None:
        plt.pcolormesh(x_vals, y_vals, data, shading='auto', vmin=vmin, vmax=vmax)
    else:
        plt.pcolormesh(x_vals, y_vals, data, shading='auto')
    plt.colorbar(label='[pW]')
    if title is not None:
        plt.title(title)

    # Apply zoom if requested
    if zoom is not None:
        xmin, xmax, ymin, ymax = zoom
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
   
        
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def radial_prof_plot(data, figsize=(8,4)):
    r1x, r1y = radial_profile(data, nbins=60)
    fig, ax = plt.subplots(2,1, figsize=figsize)
    ax[0].plot(r1x, r1y,'-o', color='blue', label='Data')
    ax[1].plot(r1x, r1y,'-o', color='blue', label='Data')
    ax[1].set_xscale('log', base=10)
    ax[1].set_yscale('log', base=10)
    ax[0].set_title('Linear Radial Profile')
    ax[1].set_title('Log10 radial Profile')
    fig.tight_layout()
    plt.show()
    

def plot_fit_result(T, fit, residual=None, abs_log = True, figsize=(3*3, 2*3), figsize2=(8,4),
                   circle_radius=None, circle_center=None, savename=None, title=None):
    if residual is None:
        residual = T - fit

    fig, ax = plt.subplots(2,3, figsize=figsize)
    vmax = max(np.max(fit), np.max(T))
    vmin = min(np.min(fit), np.min(T))
    ax[0][0].imshow(T, vmin=vmin, vmax=vmax, origin='lower')
    ax[0][0].set_title("Data")
    ax[0][1].imshow(fit, vmin=vmin, vmax=vmax, origin='lower')
    ax[0][1].set_title("Fit")
    ax[0][2].imshow(residual, vmin=vmin, vmax=vmax, origin='lower')
    ax[0][2].set_title("Residual")
    if circle_radius is not None and circle_center is not None:
        circle = plt.Circle(circle_center, circle_radius, color='r', fill=False, linewidth=5)
        ax[0][0].add_patch(circle)
        circle = plt.Circle(circle_center, circle_radius, color='r', fill=False, linewidth=5)
        ax[0][1].add_patch(circle)
        circle = plt.Circle(circle_center, circle_radius, color='r', fill=False, linewidth=5)
        ax[0][2].add_patch(circle)
        


    log_T = np.log10(T-np.min(T) + 1e-6)
    log_fit = np.log10(fit-np.min(fit) + 1e-6)
    log_residual = np.log10(residual-np.min(residual) + 1e-6)
        
    vmax = max(np.max(log_fit), np.max(log_T))
    vmin = min(np.min(log_fit), np.min(log_T))
    ax[1][0].imshow(log_T, vmin=vmin, vmax=vmax, origin='lower', cmap='inferno')
    ax[1][0].set_title("Data (Log10 Abs)")
    ax[1][1].imshow(log_fit, vmin=vmin, vmax=vmax, origin='lower', cmap='inferno')
    ax[1][1].set_title("Fit (Log10 Abs)")
    ax[1][2].imshow(log_residual, vmin=vmin, vmax=vmax, origin='lower', cmap='inferno')
    ax[1][2].set_title("Residual (Log10 Abs)")

    if circle_radius is not None and circle_center is not None:
        circle = plt.Circle(circle_center, circle_radius, color='r', fill=False, linewidth=5)
        ax[1][0].add_patch(circle)
        circle = plt.Circle(circle_center, circle_radius, color='r', fill=False, linewidth=5)
        ax[1][1].add_patch(circle)
        circle = plt.Circle(circle_center, circle_radius, color='r', fill=False, linewidth=5)
        ax[1][2].add_patch(circle)
    
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if savename is not None:
        fig.savefig(f"plots/{savename}.png")
    
    plt.show()

    r2x, r2y = radial_profile(T, nbins=60)
    r1x, r1y = radial_profile(fit, nbins=60)
    
    fig, ax = plt.subplots(2,1, figsize=figsize2)
    ax[0].plot(r2x, r2y,'-o', color='blue', label='Data')
    ax[0].plot(r1x, r1y,'-o', color='red', label='Fit')
    ax[1].plot(r2x, r2y,'-o', color='blue', label='Data')
    ax[1].plot(r1x, r1y,'-o', color='red', label='Fit')
    ax[1].set_xscale('log', base=10)
    ax[1].set_yscale('log', base=10)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Linear Radial Profile')
    ax[1].set_title('Log10 radial Profile')

    if circle_radius is not None:
        ax[0].axvline(circle_radius, linestyle='--', color='black')
        ax[1].axvline(circle_radius, linestyle='--', color='black')
    if title is not None:
        fig.suptitle(title)

    
    fig.tight_layout()
    if savename is not None:
        fig.savefig(f"plots/{savename}-radial.png")
    plt.show()


######################## FILTER ###############
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def zernike_radial(n, m, r):
    """
    Radial part of Zernike polynomial
    n: radial order
    m: azimuthal frequency (|m| <= n, n-m even)
    r: radial coordinates (0 <= r <= 1)
    """
    R = np.zeros_like(r)
    m = abs(m)
    for k in range((n - m)//2 + 1):
        c = (-1)**k * factorial(n-k) / (factorial(k) * factorial((n+m)//2 - k) * factorial((n-m)//2 - k))
        R += c * r**(n - 2*k)
    return R

def zernike(n, m, rho, theta):
    """
    Full Zernike polynomial on polar coordinates
    n: radial order
    m: azimuthal frequency
    rho: radial coordinate (0 <= rho <= 1)
    theta: azimuthal coordinate
    """
    R = zernike_radial(n, m, rho)
    if m >= 0:
        return R * np.cos(m * theta)
    else:
        return R * np.sin(-m * theta)

def plot_zernike_mode(n, m, res=200):
    """
    Plot the Zernike mode (n, m) on a unit disk
    res: resolution of the grid
    """
    # Create a grid over [-1,1] x [-1,1]
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Mask outside the unit disk
    mask = rho <= 1
    Z = np.zeros_like(X)
    Z[mask] = zernike(n, m, rho[mask], theta[mask])

    # Plot
    plt.figure(figsize=(5,5))
    plt.imshow(Z, extent=[-1,1,-1,1], origin='lower', cmap='RdBu')
    plt.colorbar(label=f'Zernike n={n}, m={m}')
    plt.title(f'Zernike mode n={n}, m={m}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def noll_to_nm(j):
    # 1-based Noll index → (n,m,sign_cos_sin)
    # Returns (n, m). Assumes real Zernikes with cos/sin combined into ±m.
    n = 0
    j1 = j - 1
    while j1 >= n + 1:
        n += 1
        j1 -= n
    m = (-n) + 2*j1
    return n, m

def zernike_spectrum_2d(a_noll, power=False):
    """
    a_noll: dict or array-like mapping Noll index (1-based) -> coefficient
            e.g. {1: a1, 2: a2, ...} or list/np.array with a_noll[0] = a1
    power : if True plot |a|^2
    """
    # normalize input into dict
    if not isinstance(a_noll, dict):
        a_noll = {i+1: a_noll[i] for i in range(len(a_noll))}
    # map to (n,m)
    pairs = [noll_to_nm(j) + (a_noll[j],) for j in a_noll]
    n_max = max(n for n,_,_ in pairs)

    # build grid: rows n=0..n_max, cols m=-n_max..+n_max step 1 (we’ll mask invalid combos)
    m_vals = np.arange(-n_max, n_max+1)
    grid = np.full((n_max+1, len(m_vals)), np.nan, dtype=float)

    for n,m,val in pairs:
        if abs(m) <= n and ((n - abs(m)) % 2 == 0):
            j = np.where(m_vals == m)[0][0]
            grid[n, j] = (abs(val)**2 if power else val)

    return grid, m_vals, np.arange(0, n_max+1)

# ---- Example usage with dummy numbers up to j=28 ----

from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def fake_zernike_image(n, m, size=50):
    """Generate a fake Zernike-like pattern for demo."""
    y, x = np.linspace(-1, 1, size), np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    #img = np.cos(m * theta) * (R**n)
    #img[R > 1] = np.nan
    mask = R <= 1
    Z = np.zeros_like(X)*np.nan
    Z[mask] = zernike(n, m, R[mask], theta[mask])
    return Z
norm = Normalize(vmin=-1, vmax=1)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
