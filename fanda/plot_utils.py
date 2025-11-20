import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _decorate_axis(ax, wrect=10, hrect=10, ticklabelsize="large"):
    """Helper function for decorating plots."""
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))
    return ax


def _annotate_and_decorate_axis(
    ax,
    labelsize="x-large",
    ticklabelsize="x-large",
    xticks=None,
    xticklabels=None,
    yticks=None,
    legend=False,
    grid_alpha=0.2,
    legendsize="x-large",
    xlabel="",
    ylabel="",
    wrect=10,
    hrect=10,
):
    """Annotates and decorates the plot."""
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    if xticks is not None:
        ax.set_xticks(ticks=xticks)
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, alpha=grid_alpha)
    ax = _decorate_axis(ax, wrect=wrect, hrect=hrect, ticklabelsize=ticklabelsize)
    if legend:
        ax.legend(fontsize=legendsize)
    return ax


def plot_sample_efficiency_curve(
    frames,
    point_estimates,
    interval_estimates,
    algorithms=None,
    colors=None,
    color_palette="colorblind",
    figsize=(7, 5),
    xlabel=r"Number of Frames (in millions)",
    ylabel="Aggregate Human Normalized Score",
    ax=None,
    labelsize="xx-large",
    ticklabelsize="xx-large",
    **kwargs,
):
    """Plots an aggregate metric with CIs as a function of environment frames.

    Args:
      frames: Array or list containing environment frames to mark on the x-axis.
      point_estimates: Dictionary mapping algorithm to a list or array of point
        estimates of the metric corresponding to the values in `frames`.
      interval_estimates: Dictionary mapping algorithms to interval estimates
        corresponding to the `point_estimates`. Typically, consists of stratified
        bootstrap CIs.
      algorithms: List of methods used for plotting. If None, defaults to all the
        keys in `point_estimates`.
      colors: Dictionary that maps each algorithm to a color. If None, then this
        mapping is created based on `color_palette`.
      color_palette: `seaborn.color_palette` object for mapping each method to a
        color.
      figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
        `ax` is None.
      xlabel: Label for the x-axis.
      ylabel: Label for the y-axis.
      ax: `matplotlib.axes` object.
      labelsize: Font size of the x-axis label.
      ticklabelsize: Font size of the ticks.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      `axes.Axes` object containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if algorithms is None:
        algorithms = list(point_estimates.keys())
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette))

    for algorithm in algorithms:
        metric_values = point_estimates[algorithm]
        lower, upper = interval_estimates[algorithm]
        ax.plot(
            frames,
            metric_values,
            color=colors[algorithm],
            marker=kwargs.get("marker", "o"),
            linewidth=kwargs.get("linewidth", 2),
            label=algorithm,
        )
        ax.fill_between(frames, y1=lower, y2=upper, color=colors[algorithm], alpha=0.2)
    kwargs.pop("marker", "0")
    kwargs.pop("linewidth", "2")

    return _annotate_and_decorate_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        labelsize=labelsize,
        ticklabelsize=ticklabelsize,
        **kwargs,
    )
