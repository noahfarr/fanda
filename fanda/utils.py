import matplotlib.pyplot as plt

def show_fig(fanda):
    plt.show()
    return fanda

def save_fig(fanda, name):

    file_name = "{}.pdf".format(name)

    fanda.fig.savefig(file_name, format="pdf", bbox_inches="tight")
    return fanda

def close_fig(fanda):
    plt.close(fanda.fig)
    return fanda

