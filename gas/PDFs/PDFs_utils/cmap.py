import colorsys
from typing import Optional, Union

import matplotlib as mpl
import numpy as np
from matplotlib import colors
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt

try:
    import colorcet as cc
except ModuleNotFoundError:
    pass

def roll_cmap(
    cmap: Union[Colormap, str],
    frac: float,
    invert: bool = False,
) -> Colormap:
    """
    Shifts a matplotlib colormap by rolling.

    Args:
        cmap (mpl.colors.Colormap | str): The colormap to be shifted. Can be a colormap name or a Colormap object.
        frac (float): The fraction of the colorbar by which to shift (must be between 0 and 1).
        invert (bool, optional): Whether to invert the colormap. Defaults to False.

    Returns:
        mpl.colors.Colormap: The shifted colormap.
    """
    N = 256
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    n = cmap.name
    x = np.linspace(0, 1, N)
    out = np.roll(x, int(N * frac))
    if invert:
        out = 1 - out
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(f"{n}_s", cmap(out))
    return new_cmap

def shift_cmap_center(
    cmap: Union[Colormap, str],
    vmin: float = -1,
    vmax: float = 1,
    midpointval: float = 0,
    invert: bool = False,
) -> Colormap:
    """
    Shifts a matplotlib colormap such that the center is moved and the scaling is consistent.

    Args:
        cmap (mpl.colors.Colormap | str): The colormap to be shifted. Can be a colormap name or a Colormap object.
        vmin (float, optional): Minimum value for scaling. Defaults to -1.
        vmax (float, optional): Maximum value for scaling. Defaults to 1.
        midpointval (float, optional): Value at the center of the colormap. Defaults to 0.
        invert (bool, optional): Whether to invert the colormap. Defaults to False.

    Returns:
        mpl.colors.Colormap: The shifted colormap.
    """
    assert vmax > vmin

    midpoint_loc = (midpointval - vmin) / (vmax - vmin)
    used_ratio = (vmax - vmin) / (2 * max(abs(vmax - midpointval), abs(vmin - midpointval)))
    N = round(256 / used_ratio)

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    x = np.linspace(0, 1, N)
    roll_frac = used_ratio if midpoint_loc < 0.5 else used_ratio * -1

    roll_ind = round(N * roll_frac)
    out = np.roll(x, roll_ind)

    if midpoint_loc < 0.5:
        out = out[:256]
    elif midpoint_loc > 0.5:
        out = out[-256:]

    if invert:
        out = 1 - out
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(f"{cmap.name}_s", cmap(out))
    return new_cmap

def get_cmap(cmap: Optional[Union[str, None]] = None, **kwargs) -> Colormap:
    """
    Take a colormap or string input and return a Colormap object.

    Args:
        cmap (str | None, optional): String corresponding to a colorcet colormap name,
            a mpl.colors.LinearSegmentedColormap object, or a
            mpl.colors.ListedColormap object. Defaults to None -> CET_C7.

    Keyword Args:
        shift (float, optional): The amount to shift the colormap by in radians. Defaults to 0.
        invert (bool, optional): Whether to invert the colormap. Defaults to False.

    Raises:
        TypeError: If the input type is not recognized.

    Returns:
        mpl.colors.Colormap: Matplotlib.colors.Colormap object.
    """
    if cmap is None:
        cmap = "linear"
    elif isinstance(cmap, colors.LinearSegmentedColormap) or isinstance(cmap, colors.ListedColormap):
        return cmap
    elif isinstance(cmap, str):
        cmap = cmap.lower()
    else:
        raise TypeError(f"Unknown input type {type(cmap)}, please input a matplotlib colormap or valid string")

    shift = kwargs.get("shift", 0)
    invert = kwargs.get("invert", False)
    try:
        if cmap in ["linear", "lin", ""]:
            cmap = plt.get_cmap("gray")
        elif cmap in ["diverging", "div"]:
            cmap = plt.get_cmap("coolwarm")
        elif cmap in ["linear_cbl", "cbl", "lin_cbl"]:
            cmap = cc.cm.CET_CBL1
        elif cmap in ["diverging_cbl", "div_cbl"]:
            cmap = cc.cm.CET_CBD1
        elif cmap in ["cet_rainbow", "cet_r1", "r1"]:
            cmap = cc.cm.CET_R1
        elif cmap in ["legacy4fold", "cet_c2", "c2", "cet_2"]:
            cmap = cc.cm.CET_C2
            shift += -np.pi / 2  # matching directions of legacy 4-fold
        elif cmap in ["purehsv", "legacyhsv"]:
            cmap = plt.get_cmap("hsv")
            invert = not invert
            shift += np.pi / 2
        elif cmap in ["cet_c6", "c6", "cet_6", "6fold", "sixfold", "hsv", "cyclic"]:
            cmap = cc.cm.CET_C6
            invert = not invert
            shift += np.pi / 2
        elif cmap in ["cet_c7", "c7", "cet_7", "4fold", "fourfold", "4-fold"]:
            cmap = cc.cm.CET_C7
            invert = not invert
        elif cmap in ["cet_c8", "c8", "cet_8"]:
            cmap = cc.cm.CET_C8
        elif cmap in ["cet_c10", "c10", "cet_10", "isolum", "isoluminant", "iso"]:
            cmap = cc.cm.CET_C10
        elif cmap in ["cet_c11", "c11", "cet_11"]:
            cmap = cc.cm.CET_C11
        elif cmap in plt.colormaps():
            cmap = plt.get_cmap(cmap)
        else:
            print(f"Unknown colormap input '{cmap}'.")
            print("You can also pass a colormap object directly.")
            print("Proceeding with default cc.cm.CET_C7.")
            cmap = cc.cm.CET_C7
    except NameError:
        print("Colorcet not installed, proceeding with hsv from mpl")
        cmap = plt.get_cmap("hsv")
        invert = not invert
        shift -= np.pi / 2
    if shift != 0:  # given as radian convert to [0,1]
        shift = shift % (2 * np.pi) / (2 * np.pi)
    if shift != 0 or invert:
        cmap = roll_cmap(cmap, shift, invert)
    return cmap
