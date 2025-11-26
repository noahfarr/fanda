import wandb
import numpy as np
import matplotlib.pyplot as plt
from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import annotate_axis, decorate_axis, pointplot
from fanda.utils import show_fig, save_fig, close_fig

plt.rcParams["mathtext.default"] = "regular"

np.random.seed(42)


def generate_cartpole_data(project_name, n_seeds=5):
    algos = ["DQN", "PPO", "SAC", "PQN"]

    for algo in algos:
        for i in range(n_seeds):
            run = wandb.init(
                project=project_name,
                config={"algorithm": algo, "env_id": "CartPole-v1"},
                reinit=True,
            )

            if algo == "DQN":
                score = np.random.normal(450, 20)

            elif algo == "PPO":
                score = np.random.normal(480, 10)

            elif algo == "SAC":
                score = np.random.normal(490, 20)

            elif algo == "PQN":
                score = np.random.normal(485, 20)

            wandb.log({"reward": np.clip(score, 0, 500)})
            run.finish()


# generate_cartpole_data("fanda")

df = fetch_wandb("noahfarr", "fanda").pipe(
    transforms.remove_outliers,
    column="reward",
    groupby="algorithm",
    lower_quantile=0.15,
    upper_quantile=1.0,
)

fanda = (
    pointplot(
        df=df,
        x="reward",
        y="algorithm",
        hue="algorithm",
        palette="flare",
        capsize=0.2,
    )
    .pipe(
        annotate_axis,
        xlabel="Average Reward",
        title="CartPole-v1 Performance",
    )
    .pipe(
        decorate_axis,
        spines=["bottom"],
    )
    .pipe(show_fig)
    .pipe(save_fig, name="images/algorithm_comparison", format="png")
    .pipe(close_fig)
)
