import wandb
import numpy as np
import matplotlib.pyplot as plt

from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, lineplot, add_legend
from fanda.utils import show_fig, save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"

np.random.seed(42)


def generate_noisy_data(num_runs=5):
    groups = ["Baseline", "Proposed Method"]

    for group in groups:
        decay = 0.02 if group == "Baseline" else 0.035

        for i in range(num_runs):
            run = wandb.init(
                project="fanda",
                group=group,
                config={"lr": 0.01, "architecture": "CNN"},
                reinit=True,
            )

            steps = np.random.randint(100, 150)

            loss = 1.0
            for step in range(steps):
                noise = np.random.normal(0, 0.025)

                loss = loss * (1 - decay) + noise

                wandb.log({"step": step, "loss": max(0, loss)})

            run.finish()


# generate_noisy_data()

df = (
    fetch_wandb("noahfarr", "fanda")
    .pipe(transforms.truncate, column="step", groupby="group")
    .pipe(
        transforms.remove_outliers,
        column="loss",
        lower_quantile=0.05,
        upper_quantile=0.95,
    )
    .pipe(transforms.normalize, column="loss", groupby="group")
)

fanda = (
    lineplot(
        df=df,
        x="step",
        y="loss",
        hue="group",
        palette="deep",
        err_kws={"alpha": 0.15},
    )
    .pipe(
        annotate_axis,
        xlabel="Training Steps",
        ylabel="Normalized Loss (Smoothed)",
        labelsize="xx-large",
    )
    .pipe(
        decorate_axis,
        ticklabelsize="xx-large",
    )
    .pipe(
        add_legend,
        labels=df["group"].unique(),
        fontsize="xx-large",
    )
    .pipe(show_fig)
    .pipe(save_fig, name="images/noisy_training", format="png")
    .pipe(close_fig)
)
