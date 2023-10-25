import matplotlib.pyplot as plt
import numpy as np
import msgpack
import os
from PIL import Image


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


def writeNumpyToImage(data, m, n, image_file, empty=-1, **kwargs):
    imgData = np.zeros([m, n, 3], dtype=np.uint8)
    max = np.max(data)
    for i in range(m):
        for j in range(n):
            index = i * n + j
            if data[index] == empty:
                imgData[i, j] = [0, 0, 255]
            else:
                imgData[i, j] = [int((data[index] / max) * 255), 255, 0]
    with Image.fromarray(imgData) as img:
        img.save(image_file, **kwargs)
    print(f"Wrote {image_file}")
