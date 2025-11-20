from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    style: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
):

    plt.figure(figsize=figsize)

    ax = sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        style=style,
        estimator="mean",
        errorbar="sd",
        palette="viridis",
        linewidth=2.0,
    )

    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    sns.despine()

    plt.tight_layout()

    return ax
