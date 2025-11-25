# Figures and Axes

Figures and Axes (fanda) is a lightweight Python library designed to streamline the process of extracting experiment data from Weights & Biases (wandb) and generating publication-quality plots.The name is a play on "wandb" (Weights and Biases) $\rightarrow$ "fanda" (Figures and Axes).

## Features
- Seamless Extraction: Pull scalar histories, configuration configs, and summary metrics from W&B runs easily.
- Publication Ready: Generate clean, professional figures using matplotlib and seaborn defaults tailored for academic papers.
- Filtering: Easily select runs based on tags, groups, or specific configuration hyperparameters.
- DataFrame Integration: Convert W&B run histories directly into Pandas DataFrames for custom analysis.

## Installation
You can install fanda directly from the PyPi:
```bash
pip install fanda
```

## Configuration
Ensure you are logged into W&B before using the library:
```bash
wandb login
```

## Usage
Here is a simple example of how to pull data from a project and plot the training loss.
```python
from fanda.wandb_client import fetch_wandb
from fanda import transforms
from fanda.visualizations import lineplot, add_legend, save_fig

(
    fetch_wandb("entity", "project", filters={
        "state": "finished",
        "created_at": {"$gte": "2025-01-01"},
    })
    .pipe(transforms.exponential_moving_average, column="evaluation/mean_episode_returns", alpha=0.7)
    .pipe( 
        lineplot, 
        x="_step", 
        y="loss", 
        hue="network",
        xlabel="Number of Steps",
        ylabel="Loss",
    )
    .pipe(add_legend, column="network")
    .pipe(save_fig, name="plot")
)
```
## Contributing
Contributions are welcome! Please read the CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you find this tool useful for your research, please consider citing it:
```
@software{fanda2025github,
  author = {Noah Farr},
  title = {fanda: Figures and Axes for Weights and Biases},
  year = {2025},
  url = {https://github.com/noahfarr/fanda},
}
```


