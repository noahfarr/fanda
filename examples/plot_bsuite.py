from functools import partial

import ast
import pandas as pd
from matplotlib import ticker
from scipy.stats import trim_mean

from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, lineplot, add_legend
from fanda.utils import save_fig, close_fig

ENV_IDS = ["MemoryChain-bsuite", "UmbrellaChain-bsuite"]

def get_networks(df):

    def func(row):
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

    df["network"] = df.apply(func, axis=1)
    return df

def make_log_axis(fanda, **kwargs):
    fanda.ax.set_xscale("log", **kwargs)
    return fanda

def plot_chain_length(df):

    def set_major_formatter(fanda):
        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(False)
        fanda.ax.xaxis.set_major_formatter(formatter)
        fanda.ax.xaxis.set_minor_formatter(formatter)
        return fanda

    labels = df["network"].unique()
    (
        lineplot(
            df=df[df["environment.env_params.max_steps_in_episode"] >= 4],
            x="environment.env_params.max_steps_in_episode", 
            y="evaluation/mean_episode_returns", 
            hue="network",
            palette="colorblind",
            marker="o",
            markeredgewidth=0,
            estimator=partial(trim_mean, proportiontocut=0.25),
            errorbar=("ci", 95),
            err_kws={"alpha": 0.2},
        )
        .pipe(
            annotate_axis, 
            xlabel="Chain Length",
            ylabel="IQM Episode Return",
            labelsize="xx-large",
        )
        .pipe(decorate_axis, ticklabelsize="xx-large")
        .pipe(make_log_axis, base=2)
        .pipe(set_major_formatter)
        .pipe(add_legend, labels=labels)
        .pipe(save_fig, name=f"plots/bsuite/{env_id}")
        .pipe(close_fig)
    )

def plot_steps(df, length):

    def set_major_formatter(fanda):
        formatter = ticker.FuncFormatter(lambda x, p: f"{x / 1e6:.0f}")
        fanda.ax.xaxis.set_major_formatter(formatter)
        fanda.ax.xaxis.set_minor_formatter(formatter)
        return fanda

    labels = df["network"].unique()
    (
        lineplot(
            df[df["environment.env_params.max_steps_in_episode"] == length],
            x="_step", 
            y="evaluation/mean_episode_returns", 
            hue="network",
            palette="colorblind",
            estimator=partial(trim_mean, proportiontocut=0.25),
            errorbar=("ci", 95),
            err_kws={"alpha": 0.2},
        )
        .pipe(
            annotate_axis, 
            xlabel="Number of Frames (in millions)",
            ylabel="IQM Episode Return",
            labelsize="xx-large",
        )
        .pipe(decorate_axis, ticklabelsize="xx-large")
        .pipe(set_major_formatter)
        .pipe(add_legend, labels=labels)
        .pipe(save_fig, name=f"plots/bsuite/{env_id}/steps/{length}")
        .pipe(close_fig)
    )

def plot_flops(df, length):
    labels = df["network"].unique()
    df = (
        df[df["environment.env_params.max_steps_in_episode"] == length]
        .pipe(transforms.align_column, column="FLOPS", groupby="network")
        .pipe(transforms.remove_outliers, column="evaluation/mean_episode_returns")
    )
    (
        lineplot(
                df=df,
                x="FLOPS", 
                y="evaluation/mean_episode_returns", 
                hue="network",
                palette="colorblind",
                estimator=partial(trim_mean, proportiontocut=0.25),
                errorbar=("ci", 95),
                err_kws={"alpha": 0.2},
        )
        .pipe(
            annotate_axis, 
            xlabel="Number of FLOPS",
            ylabel="IQM Episode Return",
            labelsize="xx-large",
        )
        .pipe(decorate_axis, ticklabelsize="xx-large")
        .pipe(make_log_axis)
        .pipe(add_legend, labels=labels)
        .pipe(save_fig, name=f"plots/bsuite/{env_id}/flops/{length}")
        .pipe(close_fig)
    )


def main(env_id):
    df = (
        fetch_wandb("noahfarr", "memorax", filters={
            "config.environment.env_id": env_id,
            "state": "finished",
        })
        .pipe(get_networks)
    )
    plot_chain_length(df.copy())
    for length in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        plot_steps(df.copy(), length)
        plot_flops(df.copy(), length)


if __name__ == "__main__":
    for env_id in ENV_IDS:
        main(env_id)

