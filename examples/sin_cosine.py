import wandb
import numpy as np

from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, lineplot, add_legend
from fanda.utils import show_fig, save_fig, close_fig

def generate_data(n, seed):
    np.random.seed(seed)

    for x in np.linspace(0, 2 * np.pi, n):
        y = np.sin(x)
        wandb.log({"x": x, "y": y})


def run():
    for seed in range(1):
        run = wandb.init(
            project="fanda",
            config={
                "seed": seed,
                "n": 100,
                "fn": "sin"
            }
        )

        generate_data(n=wandb.config.n, seed=wandb.config.seed)
        run.finish()

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
    )
    .pipe(decorate_axis, ticklabelsize="xx-large")
    .pipe(add_legend, labels=["sin(x)", "cos(x)"], fontsize="xx-large")
    # .pipe(show_fig)
    .pipe(save_fig, name="cartpole")
    .pipe(close_fig)
)
