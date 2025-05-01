"""Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# to use this script, you will need to download and install the 32-BIT VERSION of:
# - Python 2.7 x86 (32-bit) - https://www.python.org/download/releases/2.7.1
#
# you will also need the 32-BIT VERSIONS of the following packages as not all the packages are available in 64bit at the time of this writing
# The ActiveState python distribution is recommended for windows
# (make sure to get the python 2.7-compatible packages):
# - NumPy 1.5.1 (32-bit, 64-bit unofficial, supports Python 2.4 - 2.7 and 3.1 - 3.2.) - https://sourceforge.net/projects/numpy/files/NumPy/
# - matplotlib 1.0.1 (32-bit & 64-bit, supports Python 2.4 - 2.7) - https://sourceforge.net/projects/matplotlib/files/matplotlib/
#
# For ActiveState Python, all that one should need to type is 'pypm install matplotlib'

import datetime
import sys
import argparse
import subprocess
import itertools
import os
import matplotlib.pyplot as plt
import pylab
from matplotlib.backends.backend_pdf import PdfPages

os.system( "grep NT sgemm.txt > sgemm_NT.csv" )
input = open ('sgemm_NT.csv', 'r')
x = []
y = []
shape = ''
for line in input:
    line = line.replace("(", ",")
    line = line.replace(")", ",")
    value = line.split(',')
    x.append(value[1])
    y.append(value[7])
    shape = value[0]
    #print value


f = plt.figure()
plt.rcParams.update({'font.size':20})
plt.xlabel('M=N=K')
plt.ylabel("Gflop/s")
plt.title('rocBLAS SGEMM '  + shape)
plt.yticks()
plt.grid(True)
plt.legend( loc = 2)
plot1 = plt.plot(x, y)
f.savefig("sgemm.pdf", bbox_inches='tight')
input.close()
