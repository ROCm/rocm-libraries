################################################################################
#
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import os
import sys
import argparse
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

def RunPlot():

    userArgs = sys.argv[1:]

    argParser = argparse.ArgumentParser()
    argParser.add_argument("current_file", help="path where the current results are located")
    argParser.add_argument("plot_file", help="path of plot")

    args = argParser.parse_args(userArgs)

    currentFileName = args.current_file
    plotFileName = args.plot_file


    current_data = pd.read_csv(currentFileName)

    n_series = current_data['N']
    m_series = current_data['M']
    p_series = current_data['eff']

    fig1, ax1 = plt.subplots()
    ax1.plot(n_series,p_series,'+')
    ax1.set_xlabel("n")
    ax1.set_ylabel("eff")
    plot1Name = plotFileName + "_effn.pdf"
    fig1.savefig(plot1Name, dpi=300, facecolor="#f1f1f1")

    fig2, ax2 = plt.subplots()
    ax2.plot(m_series,p_series,'+')
    ax2.set_xlabel("m")
    ax2.set_ylabel("eff")
    plot2Name = plotFileName + "_effm.pdf"
    fig2.savefig(plot2Name, dpi=300, facecolor="#f1f1f1")



if __name__ == "__main__":
    RunPlot()
