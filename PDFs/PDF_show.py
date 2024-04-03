import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def plot_RDF3b(
    gr,
    dtheta,
    dr,
    theta_slices=None,
    r_slices=None,
    log=False,
    title=None,
    cbar=False,
    aspect=None,
    intensity_range = "ordered",
    vmin=None,
    vmax=None,
    **kwargs,
):
    def rFormatter(x, pos):
        return f"{x*dr:.3g}"

    def thetaFormatter(x, pos):
        return f"{x*dtheta:.3g}"

    # alias
    r_slices = kwargs.pop("r_slice", None) if r_slices is None else r_slices
    theta_slices = (
        kwargs.pop("theta_slice", None) if theta_slices is None else theta_slices
    )

    cmap = kwargs.pop("cmap", "magma")
    origin = kwargs.pop("origin", "lower")
    if aspect is None:
        aspect = "auto"
        aspect_num = 1
    else:
        aspect_num = aspect

    plot_thetas = False if theta_slices is None else True
    plot_rs = False if r_slices is None else True
    if plot_thetas == plot_rs:
        print(f"Theta slices: {theta_slices} -- {plot_thetas}")
        print(f"r slices: {r_slices} -- {plot_rs}")
        print("Currently plots theta slices or r slices, not both (or neither)")
        return

    val_slices = theta_slices if plot_thetas else r_slices
    dind = dtheta if plot_thetas else dr

    if isinstance(val_slices, str):
        if val_slices.lower() not in ["peak", "max"]:
            print(f"Wrong val slices given {val_slices}")
            print("Expected 'peak' or 'max'")
            return
        if plot_rs:
            val_slices = np.unravel_index(np.argmax(gr), gr.shape)[1] * dr
        elif plot_thetas:
            val_slices = np.unravel_index(np.argmax(gr), gr.shape)[0] * dr

    if isinstance(val_slices, (int, float)):
        int_slice = int(round(val_slices / dind))
        ind_slices = np.array([(int_slice, int_slice + 1)])
        val_slices = [val_slices]
    elif isinstance(val_slices, (tuple, list, np.ndarray)):
        if np.ndim(val_slices) == 1:
            int_slices = np.round(np.array(val_slices) / dind).astype("int")
            ind_slices = np.array([[sl, sl + 1] for sl in int_slices])
        elif np.ndim(val_slices) == 2:
            ind_slices = np.array(
                [np.round(np.array(sl) / dind) for sl in val_slices]
            ).astype("int")

    ncols = len(ind_slices)
    figsize = kwargs.pop("figsize", (ncols * 4 * aspect_num, 3))
    fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=figsize)
    if ncols == 1:
        axs = [axs]
    for i, (ax, slice) in enumerate(zip(axs, ind_slices)):
        if np.ndim(val_slices[i]) == 0:
            slice_title = str(round(val_slices[i], 3))
        elif np.ndim(val_slices[i]) == 1:
            islice = np.round(val_slices[i], 3)
            slice_title = f"{islice[0]} : {islice[1]}"
        if plot_thetas:
            gslice = gr[slice[0] : slice[1]].mean(axis=0)
            yformatter = rFormatter
            ax.set_title(f"$\\theta = {slice_title}^\\circ$")
            if i == 0:
                ax.set_ylabel(f"r2 (A)")
        elif plot_rs:
            gslice = gr[:, slice[0] : slice[1]].mean(axis=1)
            yformatter = thetaFormatter
            ax.set_title(f"r2 = {slice_title} A")
            if i == 0:
                ax.set_ylabel(f"$\\theta$ (deg)")
        if log:
            _mask = gslice > 0.0
            gslice[_mask] = np.log(gslice[_mask])
            gslice[~_mask] = 0

        if intensity_range == "ordered":
            vmin = 0.001 if vmin is None else vmin
            vmax = 0.999 if vmax is None else vmax
            vals = np.sort(gslice.ravel())
            ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
            ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
            ind_vmin = np.max([0, ind_vmin])
            ind_vmax = np.min([len(vals) - 1, ind_vmax])
            vmin_ax = vals[ind_vmin]
            vmax_ax = vals[ind_vmax]
        else:
            vmin_ax, vmax_ax = vmin, vmax

        im = ax.matshow(gslice, cmap=cmap, origin=origin, aspect=aspect, vmin=vmin_ax, vmax=vmax_ax, **kwargs)
        ax.set_xlabel(f"r1 (A)")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(yformatter))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(rFormatter))
        ax.xaxis.tick_bottom()
        ax.tick_params(direction="out")

        if cbar:
            cax = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=str(kwargs.get("cbar_title", "")))

    fig.subplots_adjust(wspace=0.35 if cbar else 0.15)

    if title is not None:
        fig.suptitle(title, y=1.02)

    return

def plot_r1r2_RDF3b(gr, dtheta, dr, log=False, title="", cbar=False, aspect=None, intensity_range="ordered", **kwargs):
    "theta_slice and r_slice given in deg and angstroms respectively"

    def rFormatter(x, pos):
        return f"{x*dr:.3g}"

    def thetaFormatter(x, pos):
        return f"{x*dtheta:.3g}"

    cmap = kwargs.pop("cmap", "magma")
    origin = kwargs.pop("origin", "lower")

    gr_r1r2 = np.array([np.diag(gr[i]) for i in range(gr.shape[0])])
    if log:
        _mask = gr_r1r2 > 0.0
        gr_r1r2[_mask] = np.log(gr_r1r2[_mask])
        gr_r1r2[~_mask] = 0

    if aspect is None:
        aspect = gr_r1r2.shape[0] / gr_r1r2.shape[1]
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (5, 5*aspect)))

    if intensity_range == "ordered":
        vmin = kwargs.pop("vmin", 0.001)
        vmax = kwargs.pop("vmax", 0.999)
        vals = np.sort(gr_r1r2.ravel())
        ind_vmin = np.round((vals.shape[0] - 1) * vmin).astype("int")
        ind_vmax = np.round((vals.shape[0] - 1) * vmax).astype("int")
        ind_vmin = np.max([0, ind_vmin])
        ind_vmax = np.min([len(vals) - 1, ind_vmax])
        vmin_ax = vals[ind_vmin]
        vmax_ax = vals[ind_vmax]
    else:
        vmin_ax = kwargs.pop("vmin", None)
        vmax_ax = kwargs.pop("vmax", None)

    im = ax.matshow(gr_r1r2, cmap=cmap, origin=origin, vmin=vmin_ax, vmax=vmax_ax, **kwargs)
    ax.set_ylabel(f"$\\theta$ (deg)")
    ax.set_xlabel(f"r1 = r2 (A)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(thetaFormatter))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(rFormatter))
    ax.set_yticks(np.arange(0, 190, 30) / dtheta)

    ax.xaxis.tick_bottom()
    ax.tick_params(direction="out")

    if cbar:
        ax_divider = make_axes_locatable(ax)
        c_axis = ax_divider.append_axes("right", size="4%", pad="2%")
        fig.colorbar(im, cax=c_axis, format="%g", label=str(kwargs.pop("cbar_title", "")))

    plt.show()
    return

