import matplotlib.pyplot as plt

def show_fig(fanda):
    plt.show()
    return fanda

def save_fig(fanda, name, format="pdf"):

    file_name = f"{name}.{format}"

    fanda.fig.savefig(file_name, format=format, bbox_inches="tight")
    return fanda

def close_fig(fanda):
    plt.close(fanda.fig)
    return fanda

