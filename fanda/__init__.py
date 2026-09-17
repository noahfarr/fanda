import os

import matplotlib
import matplotlib.pyplot as plt

import fanda


def load_stylesheets():
    stylesheets = {}
    path = os.path.join(fanda.__path__[0], "styles")
    for folder, _, files in os.walk(path):
        for file in files:
            name, extension = os.path.splitext(file)
            if extension == ".mplstyle":
                stylesheets[name] = matplotlib.rc_params_from_file(
                    os.path.join(folder, file), use_default_template=False
                )
    return stylesheets


stylesheets = load_stylesheets()
plt.style.library.update(stylesheets)
plt.style.available[:] = sorted(plt.style.library.keys())
