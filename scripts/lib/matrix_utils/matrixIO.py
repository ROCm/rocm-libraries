import matplotlib.pyplot as plt
import numpy as np
import msgpack
import os


def loadMatrix(fname):
    with open(fname, "rb") as f:
        x = msgpack.load(f)
    # Since the array is written as column major but Numpy is row-major,
    # 1. The order of the sizes is backwards compared to what is expected here.
    # 2. In order to plot the data in a way that looks right, we then need to
    #    transpose the matrix.
    data = np.array(x["data"]).reshape(list(reversed(x["sizes"]))).T
    return data


def showMatrix(fname, cmap="plasma", aspect="auto", **kwargs):
    data = loadMatrix(fname)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    fig.suptitle(os.path.splitext(os.path.basename(fname))[0])
    colormap = ax.matshow(data, aspect=aspect, cmap=cmap, **kwargs)
    fig.colorbar(colormap)
    return fig
