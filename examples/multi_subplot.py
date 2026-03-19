import numpy as np
import pandas as pd

from fanda.visualizations import (
    subplots,
    add_lineplot,
    annotate_axis,
    decorate_axis,
)
from fanda.utils import save_fig, close_fig

np.random.seed(42)
steps = np.arange(0, 100)

# Simulate training loss (decreasing) and reward (increasing) for two methods.
records = []
for method in ["PPO", "DQN"]:
    offset = 0.0 if method == "PPO" else 0.3
    for step in steps:
        loss = 2.0 * np.exp(-0.03 * step) + offset + np.random.normal(0, 0.05)
        reward = 1.0 - np.exp(-0.04 * step) - offset + np.random.normal(0, 0.03)
        records.append({"step": step, "loss": loss, "reward": reward, "method": method})

df = pd.DataFrame(records)

f = (
    subplots(1, 2, figsize=(14, 5))
    .pipe(add_lineplot, df=df, x="step", y="loss", hue="method", palette="colorblind")
    .pipe(annotate_axis, xlabel="Step", ylabel="Loss", title="Training Loss")
    .pipe(decorate_axis)
    .select(1)
    .pipe(add_lineplot, df=df, x="step", y="reward", hue="method", palette="colorblind")
    .pipe(annotate_axis, xlabel="Step", ylabel="Reward", title="Reward")
    .pipe(decorate_axis)
    .pipe(save_fig, name="images/multi_subplot", format="png")
    .pipe(close_fig)
)
