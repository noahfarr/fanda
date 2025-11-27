import os

import matplotlib.pyplot as plt

import fanda

def load_stylesheets():
    stylesheets = {}
    path = os.path.join(fanda.__path__[0], "styles")
    for folder, _, _ in os.walk(path):
        new_stylesheets = plt.style.core.read_style_directory(folder)
        stylesheets.update(new_stylesheets)
    return stylesheets

stylesheets = load_stylesheets()
plt.style.core.update_nested_dict(plt.style.library, stylesheets)
plt.style.core.available[:] = sorted(plt.style.library.keys())
