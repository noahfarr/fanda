import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import ticker
from matplotlib import rcParams
from matplotlib import rc
import seaborn as sns

from fanda.fanda import Fanda

sns.set_style("white")

rcParams["legend.loc"] = "best"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "serif"]
plt.rcParams["mathtext.fontset"] = "stix"

rc("text", usetex=False)


def decorate_axis(fanda, wrect=10, hrect=10, ticklabelsize="large", spines=None):
    """Helper function for decorating plots."""
    spines = spines or ["bottom", "left"]
    for spine in ["top", "right", "bottom", "left"]:
        if spine not in spines:
            fanda.ax.spines[spine].set_visible(False)
        else:
            fanda.ax.spines[spine].set_linewidth(2)

    # Deal with ticks and the blank space at the origin
    fanda.ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    if "top" not in spines and "right" not in spines:
        fanda.ax.spines["left"].set_position(("outward", hrect))
        fanda.ax.spines["bottom"].set_position(("outward", wrect))
    return fanda


def annotate_axis(
    fanda,
    labelsize="x-large",
    xticks=None,
    xticklabels=None,
    yticks=None,
    grid_alpha=0.2,
    xlabel="",
    ylabel="",
    title="",
):
    """Annotates and decorates the plot."""
    fanda.ax.set_xlabel(xlabel, fontsize=labelsize)
    fanda.ax.set_ylabel(ylabel, fontsize=labelsize)
    fanda.ax.set_title(title, fontsize=labelsize)
    if xticks is not None:
        fanda.ax.set_xticks(ticks=xticks)
        fanda.ax.set_xticklabels(xticklabels)
    if yticks is not None:
        fanda.ax.set_yticks(yticks)
    fanda.ax.grid(True, alpha=grid_alpha)
    return fanda


def lineplot(
    df,
    x: str,
    y: str,
    figsize=(7, 5),
    **kwargs,
):

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.lineplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        **kwargs,
    )

    return Fanda(fig=fig, ax=ax)


def pointplot(
    df,
    x: str,
    y: str,
    figwidth=3.4,
    row_height=0.37,
    **kwargs,
):
    figsize = (figwidth, row_height * len(df[y].unique()))
    fig, ax = plt.subplots(figsize=figsize)

    ax = sns.pointplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        **kwargs,
    )
    return Fanda(fig=fig, ax=ax)

def add_legend(fanda, labels, fontsize="x-large"):
    colors = sns.color_palette("colorblind", len(labels))
    colors = dict(zip(labels, colors))
    fake_patches = [mpatches.Patch(color=colors[label], alpha=0.75) for label in labels]
    fanda.ax.legend(
        fake_patches,
        labels,
        loc="upper center",
        fancybox=True,
        ncol=min(len(labels), 5),
        fontsize=fontsize,
        bbox_to_anchor=(0.5, 1.2),
    )
    return fanda

def save_fig(fanda, name):
    file_name = "{}.pdf".format(name)
    fanda.fig.savefig(file_name, format="pdf", bbox_inches="tight")
    return fanda

