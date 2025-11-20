import wandb
import matplotlib.pyplot as plt

from wandp.wandb_client import fetch_history
from wandp.plotting import plot_metric


def main():
    api = wandb.Api()
    df = fetch_history(api, "noahfarr", "wandp")
    ax = plot_metric(
        df, "_step", "evaluation/mean_episode_returns", hue="algorithm.name"
    )
    plt.show()


if __name__ == "__main__":
    main()
