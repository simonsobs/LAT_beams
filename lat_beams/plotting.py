import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from .beam_utils import radial_profile

plt.rcParams["image.cmap"] = "coolwarm"  # "RdGy_r"


def plot_map(
    data,
    pixsize,
    extent,
    plt_extent,
    cent,
    plt_cent,
    zoom,
    ufm_plot_dir,
    obs_id,
    ufm,
    band_name,
    comp="T",
    log=False,
    log_thresh=1e-3,
    append="",
):
    plt.close()
    _norm = None
    label = f"_{comp}"
    if append != "":
        label += f"_{append}"
    rprof = radial_profile(data, cent[::-1])[: int(0.5 * min(*data.shape))]
    if log:
        _norm = SymLogNorm(linthresh=log_thresh, clip=True, vmin=-1, vmax=1)
        label += f"_log10"
        with np.errstate(divide="ignore", invalid="ignore"):
            rprof = np.sign(rprof) * np.log10(np.abs(rprof))
        plt.imshow(data, origin="lower", extent=plt_extent, norm=_norm)
    else:
        vminmax = np.percentile(np.abs(data), 99)
        plt.imshow(
            data, origin="lower", extent=plt_extent, vmin=-1 * vminmax, vmax=vminmax
        )
    plt.colorbar()
    plt.grid()
    plt.xlabel('Xi (")')
    plt.ylabel('Eta (")')
    plt.title(f"{obs_id}_{ufm}_{band_name}{label.replace('_', ' ')}")
    plt.xlim((plt_cent[0] - extent, plt_cent[0] + extent))
    plt.ylim((plt_cent[1] - extent, plt_cent[1] + extent))
    plt.savefig(
        os.path.join(ufm_plot_dir, f"{obs_id}_{ufm}_{band_name}_map{label}.png")
    )
    if zoom != 1:
        plt.xlim((plt_cent[0] - extent / zoom, plt_cent[0] + extent / zoom))
        plt.ylim((plt_cent[1] - extent / zoom, plt_cent[1] + extent / zoom))
        plt.savefig(
            os.path.join(
                ufm_plot_dir, f"{obs_id}_{ufm}_{band_name}_map{label}_zoom.png"
            )
        )

    plt.close()
    x = np.linspace(0, pixsize * len(rprof), len(rprof))
    plt.plot(x, rprof)
    plt.xlabel('Radius (")')
    plt.title(f"{obs_id}_{ufm}_{band_name}{label.replace('_', ' ')}")
    plt.xlim((0, extent))
    plt.savefig(
        os.path.join(ufm_plot_dir, f"{obs_id}_{ufm}_{band_name}_prof{label}.png"),
        bbox_inches="tight",
    )
    if zoom != 1:
        plt.xlim((0, extent / zoom))
        lims = plt.gca().get_xlim()
        i = np.where((x >= lims[0]) & (x <= lims[1]))[0]
        plt.gca().set_ylim(np.nanmin(rprof[i]), np.nanmax(rprof[i]))
        plt.savefig(
            os.path.join(
                ufm_plot_dir, f"{obs_id}_{ufm}_{band_name}_prof{label}_zoom.png"
            ),
            bbox_inches="tight",
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
