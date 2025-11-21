import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast

from fanda.wandb_client import fetch_history
from fanda.plot_utils import (
    plot_learning_curves,
    plot_interval_estimates,
    annotate_and_decorate_axis,
    add_legend,
    save_fig,
)


def filter_runs(df: pd.DataFrame) -> pd.DataFrame:
    latest_timestamps = (
        (
            df.groupby(
                [
                    "network",
                    "environment.env_id",
                    "seed",
                    "group",
                ]
            )["_timestamp"]
            .max()
            .reset_index()
        )
        .sort_values("_timestamp", ascending=False)
        .drop_duplicates(
            subset=[
                "network",
                "environment.env_id",
                "seed",
            ],
            keep="first",
        )
    )

    df = df[df["group"].isin(latest_timestamps["group"].unique())].copy()
    return df


def get_networks(row):
    torso = str(row["algorithm.actor.torso._target_"])
    cell = str(row["algorithm.actor.torso.cell._target_"])
    pattern = str(row["algorithm.actor.torso.cell.pattern"])

    torso_name = torso.split(".")[-1]

    if torso_name == "RNN":
        if cell == "None" or pd.isna(cell):
            return torso_name

        cell_name = cell.split(".")[-1]

        if cell_name == "xLSTMCell":
            pattern = ast.literal_eval(pattern)
            cell_name = "".join(pattern) + "LSTM"
        return cell_name.replace("Cell", "")

    return torso_name


def plot_popgym():
    df = pd.read_parquet("data/popgym_easy.parquet")

    metric = "evaluation/mmer"

    df["network"] = df.apply(get_networks, axis=1)

    df = filter_runs(df)

    df["seed"] = df.groupby(["network", "environment.env_id", "_step"]).cumcount()

    df["evaluation/mmer"] = df.groupby("environment.env_id")[metric].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    df = (
        df.groupby(["network", "seed", "_step"])[metric]
        .mean()
        .reset_index()
        .groupby(["network", "seed"])[metric]
        .max()
        .reset_index()
    )
    df.rename(columns={"evaluation/mmer": "MMER"}, inplace=True)

    df["network"] = df["network"].apply(lambda x: str(x).split(".")[-1])
    df = df.sort_values("MMER", ascending=False)

    color_palette = sns.color_palette("colorblind")
    xlabels = df["network"].unique().tolist()
    colors = dict(zip(xlabels, color_palette))

    fig, ax = plot_interval_estimates(
        df,
        x="MMER",
        y="network",
        hue="network",
        palette=colors,
        capsize=0.2,
        dodge=True,
    )
    ax = annotate_and_decorate_axis(
        ax,
        xlabel="Normalized MMER",
        ylabel="Memory Architecture",
        labelsize="xx-large",
        ticklabelsize="xx-large",
    )
    plt.show()
    save_fig(fig, "plots/popgym_easy")


def plot_bsuite():
    df = pd.read_parquet("data/bsuite_memory_chain.parquet")

    df["network"] = df.apply(get_networks, axis=1)

    df = filter_runs(df)

    color_palette = sns.color_palette("colorblind")
    xlabels = df["network"].unique().tolist()
    colors = dict(zip(xlabels, color_palette))

    fig, ax = plot_learning_curves(
        df,
        x="environment.env_params.memory_length",
        y="evaluation/mean_episode_returns",
        hue="network",
        palette=colors,
        marker="o",
        markeredgewidth=0,
        errorbar=("ci", 95),
        err_kws={"alpha": 0.2},
    )
    ax = annotate_and_decorate_axis(
        ax,
        xlabel="Memory Length",
        ylabel="Mean Episode Return",
        labelsize="xx-large",
        ticklabelsize="xx-large",
        legend=True,
    )
    color_palette = sns.color_palette("colorblind")
    xlabels = [xlabel.split(".")[-1] for xlabel in xlabels]
    colors = dict(zip(xlabels, color_palette))
    ax = add_legend(
        ax,
        labels=xlabels,
        colors=colors,
    )
    plt.show()
    save_fig(fig, "plots/bsuite_memory_chain")


if __name__ == "__main__":
    plot_popgym()
    # plot_bsuite()
