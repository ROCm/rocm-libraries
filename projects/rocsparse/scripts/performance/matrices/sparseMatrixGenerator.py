# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from numpy.random import default_rng
from scipy.io import mmwrite
import subprocess
import os
import pathlib

rng = default_rng(42)

def samePatternPerRow(m, n, col_per_row):
    row  = np.repeat(np.arange(0, m), len(col_per_row))
    col  = np.tile(col_per_row, m)

    data = np.ones((len(row),))

    return coo_matrix((data, (row, col)), shape=(m, n))

def consecutiveCols(m, n, nnz_per_row, first_col):
    if first_col + nnz_per_row >= n:
        col_per_row = np.arange(n-nnz_per_row, n)
    else:
        col_per_row = np.arange(first_col, first_col+nnz_per_row)

    return samePatternPerRow(m, n, col_per_row)

def sameRandomRow(m, n, nnz_per_row):
    col_per_row = np.sort(rng.choice(n, size=nnz_per_row, replace=False))

    return samePatternPerRow(m, n, col_per_row)

def sameNnzPerRow(m, n, nnz_per_row):
    row = np.repeat(np.arange(0, m), nnz_per_row)
    col = np.zeros((len(row),), dtype=int)
    for i in range(0, m):
        col[i*nnz_per_row:(i+1)*nnz_per_row]  = np.sort(rng.choice(n, size=nnz_per_row, replace=False))

    data = np.ones((len(row),))

    return coo_matrix((data, (row, col)), shape=(m, n))

def fullyRandom(m, n, nnz):
    oneDRand = np.sort(rng.choice(m*n, size=nnz, replace=False))

    row = oneDRand/n
    col = oneDRand%n

    data = np.ones((nnz,))

    return coo_matrix((data, (row, col)), shape=(m, n))

def writeToFile(filename, A):
    mmwrite(filename, A, comment='\n Matrix generated with sparseMatrixGenerator.\n', precision=6)

def convert(filename, A, cleanFile=True):
    current_work_dir = os.getcwd()
    os.chdir(pathlib.Path(__file__).parent.resolve())
    fileformat = ".mtx"
    writeToFile(filename + fileformat, A)
    subprocess.run("./convert "+ filename + fileformat + " " + filename + ".csr", shell=True, check=True)
    if cleanFile:
        os.remove(filename + fileformat)
    else:
        os.rename(str(pathlib.Path(__file__).parent.resolve()) + "/" + filename + fileformat, str(current_work_dir) + "/" + filename + fileformat)
    os.rename(str(pathlib.Path(__file__).parent.resolve()) + "/" + filename + ".csr", str(current_work_dir) + "/" + filename + ".csr")
    os.chdir(current_work_dir)
