################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

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
