import wandb
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["mathtext.default"] = "regular"

from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, lineplot, add_legend
from fanda.utils import show_fig, save_fig, close_fig

np.random.seed(0)


def generate_data(fn, n):

    for x in np.linspace(0, 2 * np.pi, n):
        y = fn(x)
        wandb.log({"x": x, "y": y})


def run():
    fns = {"sin(x)": np.sin, "cos(x)": np.cos}
    for label, fn in fns.items():
        run = wandb.init(
            project="fanda",
            group=label,
            config={
                "n": 100,
                "fn": label,
            },
        )

        generate_data(fn, n=wandb.config.n)
        run.finish()


# run()
df = fetch_wandb("noahfarr", "fanda")
fanda = (
    lineplot(
        df=df,
        x="x",
        y="y",
        hue="fn",
        palette="colorblind",
        err_kws={"alpha": 0.2},
    )
    .pipe(
        annotate_axis,
        xlabel="x",
        ylabel="y",
        labelsize="xx-large",
        xticks=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
        xticklabels=[
            "$0$",
            r"$\frac{\pi}{2}$",
            r"$\pi$",
            r"$\frac{3\pi}{2}$",
            r"$2\pi$",
        ],
    )
    .pipe(
        decorate_axis,
        ticklabelsize="xx-large",
        spines=["top", "right", "bottom", "left"],
    )
    .pipe(add_legend, labels=df["fn"].unique(), fontsize="xx-large")
    .pipe(show_fig)
    .pipe(save_fig, name="images/sin_cos", format="png")
    .pipe(close_fig)
)
