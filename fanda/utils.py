import matplotlib.pyplot as plt
from pathlib import Path


def show_fig(fanda):
    plt.show()
    return fanda


def save_fig(fanda, name, format="pdf", ensure_dir=True):

    file_name = Path(f"{name}.{format}")

    if ensure_dir:
        file_name.parent.mkdir(parents=True, exist_ok=True)

    fanda.fig.savefig(file_name, format=format, bbox_inches="tight")
    return fanda


def close_fig(fanda):
    plt.close(fanda.fig)
    return fanda
