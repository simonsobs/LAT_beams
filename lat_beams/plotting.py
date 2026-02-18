import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from .beam_utils import radial_profile

plt.rcParams["image.cmap"] = "coolwarm"  # "RdGy_r"


def plot_map(
    data,
    posmap,
    pixsize,
    extent,
    cent,
    plot_dir,
    title,
    comp="T",
    log=False,
    log_thresh=1e-3,
    append="",
    units='"',
):
    plt.close()
    # Use posmap to setup coordinates
    # posmap, pixsize, extent, and cent must all be in the same units
    # We are assuming square pixels here
    plt_extent = (posmap[1].max(), posmap[1].min(), posmap[0].min(), posmap[0].max())

    # Get radial profile and plot map with appropriate norm
    _norm = None
    label = f"_{comp}"
    if append != "":
        label += f"_{append}"
    rsq = (posmap[0] - cent[1]) ** 2 + (posmap[1] - cent[0]) ** 2
    rprof = radial_profile(
        data,
        np.unravel_index( np.argmin(rsq), data.shape)[::-1],
    )[: int(0.5 * min(*data.shape))]
    if log:
        _norm = SymLogNorm(linthresh=log_thresh, clip=True, vmin=-1, vmax=1)
        label += f"_log10"
        with np.errstate(divide="ignore", invalid="ignore"):
            rprof = np.sign(rprof) * np.log10(np.abs(rprof))
        plt.imshow(data, origin="lower", extent=plt_extent, norm=_norm)
    else:
        vminmax = np.max(np.abs(data)[rsq < extent**2])
        plt.imshow(
            data, origin="lower", extent=plt_extent, vmin=-1 * vminmax, vmax=vminmax
        )
    plt.colorbar()
    plt.grid()
    plt.xlabel(f"Xi ({units})")
    plt.ylabel(f"Eta ({units})")
    plt.title(f"{title}{label.replace('_', ' ')}")

    plt.xlim((cent[0] - extent, cent[0] + extent))
    plt.ylim((cent[1] - extent, cent[1] + extent))
    plt.savefig(os.path.join(plot_dir, f"{title.replace(' ', '_')}_map{label}.png"))

    # Profile
    plt.close()
    x = np.linspace(0, pixsize * len(rprof), len(rprof))
    plt.plot(x, rprof)
    plt.xlabel(f"Radius ({units})")
    plt.title(f"{title}{label.replace('_', ' ')}")
    plt.xlim((0, extent))
    plt.savefig(
        os.path.join(plot_dir, f"{title.replace(' ', '_')}_prof{label}.png"),
        bbox_inches="tight",
    )


def plot_map_complete(
    data,
    posmap,
    pixsize,
    extent,
    cent,
    plot_dir,
    title,
    comps="TQU",
    log_thresh=1e-3,
    append="",
    units='"',
    lognorm=1,
    qrur=False,
):
    if len(data.shape) == 2:
        data = data[None, ...]
    for i, comp in enumerate(comps):
        for log in (False, True):
            plot_map(
                data[i] * (lognorm * log + (not log)),
                posmap,
                pixsize,
                extent,
                cent,
                plot_dir,
                title,
                comp=comp,
                log=log,
                log_thresh=log_thresh,
                append=append,
                units=units,
            )
    if not qrur:
        return
    if "Q" not in comps or "U" not in comps:
        raise ValueError("Cannot plot Qr and Ur without Q and U")
    Q = data[comps.find("Q")]
    U = data[comps.find("U")]
    eta, xi = posmap
    theta = np.arctan2(eta, xi)
    Q_r = Q*np.cos(2*theta) + U*np.sin(2*theta)
    U_r = U*np.cos(2*theta) - Q*np.sin(2*theta)
    for dat, comp in [(Q_r, "Qr"), (U_r, "Ur")]:
        for log in (False, True):
            plot_map(
                dat * (lognorm * log + (not log)),
                posmap,
                pixsize,
                extent,
                cent,
                plot_dir,
                title,
                comp=comp,
                log=log,
                log_thresh=log_thresh,
                append=append,
                units=units,
            )



def plot_tod(aman, sig_filt, tod_plot_dir, file_label):
    plt.close()

    plt.plot(np.array(aman.signal).T, alpha=0.3)
    plt.xlabel("Samples")
    plt.ylabel("Signal (pW)")
    plt.savefig(os.path.join(tod_plot_dir, f"{file_label}_tod.png"))
    plt.close()

    plt.plot(sig_filt.T, alpha=0.3)
    plt.xlabel("Samples")
    plt.ylabel("Filtered Signal (pW)")
    plt.savefig(os.path.join(tod_plot_dir, f"{file_label}_tod_filt.png"))
    plt.close()


def plot_focal_plane(focal_plane, fit_plot_dir, ufm):
    plt.close()

    plt.scatter(np.array(focal_plane.xi), np.array(focal_plane.eta), alpha=0.25)
    plt.xlabel("Xi (rad)")
    plt.ylabel("Eta (rad)")
    plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp.png"))
    plt.close()

    plt.scatter(np.array(focal_plane.az), np.array(focal_plane.el), alpha=0.25)
    plt.xlabel("Az (rad)")
    plt.ylabel("El (rad)")
    plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_enc.png"))
    plt.close()

    plt.hist(np.array(focal_plane.amp), bins=30, alpha=0.25)
    plt.xlabel("Amp (pW)")
    plt.ylabel("Dets (#)")
    plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp_amp.png"))
    plt.close()

    plt.hist(np.array(focal_plane.fwhm), bins=30, alpha=0.25)
    plt.xlabel("FWHM (rad)")
    plt.ylabel("Dets (#)")
    plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp_fwhm.png"))
    plt.close()

    plt.hist(np.array(focal_plane.hits), bins=30, alpha=0.25)
    plt.xlabel("Hits (#)")
    plt.ylabel("Dets (#)")
    plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp_hits.png"))
    plt.close()

    plt.hist(np.array(focal_plane.reduced_chisq), bins=30, alpha=0.25)
    plt.xlabel("Reduced Chi Squared")
    plt.ylabel("Dets (#)")
    plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp_red_chisq.png"))
    plt.close()

    plt.hist(np.array(focal_plane.R2), bins=30, alpha=0.25)
    plt.xlabel("R2")
    plt.ylabel("Dets (#)")
    plt.savefig(os.path.join(fit_plot_dir, f"{ufm}_fp_r2.png"))
    plt.close()
