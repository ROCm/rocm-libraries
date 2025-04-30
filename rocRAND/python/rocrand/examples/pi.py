# Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import rocrand
import numpy as np

interactive = True
if not interactive:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


figsize = None
if not interactive:
    figsize = (30, 20)

plt.figure(1, figsize=figsize)
plt.suptitle("Estimating Pi using Monte Carlo method")

n = 10000

cols = 3
for col in range(cols):
    if col == 0:
        title = "numpy.random"
        np.random.seed(0)
        xy = np.random.uniform(size=(2, n))
    elif col == 1:
        title = "rocrand.PRNG"
        gen = rocrand.PRNG()
        xy = np.empty(shape=(2, n))
        gen.uniform(xy)
    elif col == 2:
        title = "rocrand.QRNG"
        gen = rocrand.QRNG(ndim=2)
        xy = np.empty(shape=(2, n))
        gen.uniform(xy)

    inside = xy[0]**2 + xy[1]**2 <= 1.0

    in_xy  = xy[:,  inside]
    out_xy = xy[:, ~inside]

    trials = np.arange(1, n + 1)
    hits = np.cumsum(inside)
    pis = 4.0 * (hits.astype(np.float) / trials)

    plt.subplot(2, 3, col + 1, aspect="equal")
    plt.title(title)
    plt.scatter(*in_xy,  c="g", marker=".", s=0.5)
    plt.scatter(*out_xy, c="r", marker=".", s=0.5)

    plt.subplot(2, 3, cols + col + 1)
    plt.axhline(y=np.pi, color="b", linestyle="-")
    plt.plot(trials, pis, color="c", linewidth=0.5)
    plt.ylim(2.8, 3.4)

plt.show()

if not interactive:
    plt.savefig("pi.png")
