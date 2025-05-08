#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

import argparse
import subprocess
import os
import re # regexp package
import sys
import tempfile
import json
import xml.etree.ElementTree as ET

#
#
# GNUPLOT COMMANDS FOR HISTOGRAM
#
#
def simple_histogram(out,ofilename,title,indices,ifilename,x_range,y_label,col_index, titles):
    nplots = len(indices)
    out.write("reset\n")
    out.write("set grid\n")
    out.write("set style fill solid 0.2\n")
    out.write("set style data histograms\n")
    out.write("set term pdfcairo enhanced color font 'Helvetica,9'\n")
    out.write("set output \"" + ofilename + "\"\n")
    out.write("set termoption noenhanced\n")
    out.write("set tmargin 0\n")
    out.write("set bmargin 0\n")
    out.write("set ylabel \"" + y_label + "\"\n")
    out.write("set xrange [" + str(x_range[0]) + ":" + str(x_range[1]) + "]\n")
    out.write("set offsets 0.25, 0.25, 0, 0\n")
    out.write("set xtics rotate by -45\n")
    out.write("set boxwidth 0.5\n")
    out.write("set size ratio 0.35\n")
    out.write("set style fill noborder\n")
    out.write("set title '" + title +"'\n")
    out.write("plot '"+ifilename+"' index "+str(indices[0])+" using "+str(col_index)+ ":xtic(1) title '"+ titles[0] +"' with histogram")
    for i in range(1,nplots):
        out.write(",\\\n '' index "+str(indices[i])+" using "+str(col_index)+":xtic(1) title '"+titles[i]+"' with histogram")
    out.write("\n")



def histogram(out,ofilename,title,indices,ifilename,x_range,y_label,col_index,col_index_low,col_index_high, titles, linear):
    nplots = len(indices)
    out.write("reset\n")
    out.write("set grid\n")
    out.write("set style fill solid 0.2\n")
    out.write("set style data histograms\n")
    out.write("set term pdfcairo enhanced color font 'Helvetica,9'\n")
    out.write("set output \"" + ofilename + "\"\n")
    out.write("set termoption noenhanced\n")
    out.write("set tmargin 0\n")
    out.write("set bmargin 0\n")

    # ERROR BARS
    out.write("set style histogram errorbars linewidth 1\n")
    out.write("set errorbars linecolor black\n")
    out.write("set bars front\n")

    out.write("set ylabel \"" + y_label + "\"\n")
    out.write("set xrange [" + str(x_range[0]) + ":" + str(x_range[1]) + "]\n")
    if not linear:
        out.write("set logscale y\n")
    out.write("set offsets 0.25, 0.25, 0, 0\n")
    out.write("set xtics rotate by -45\n")
    out.write("set boxwidth 0.5\n")
    out.write("set size ratio 0.35\n")
    out.write("set style fill noborder\n")
    out.write("set title '" + title +"'\n")
    out.write("plot '"+ifilename+"' index "+str(indices[0])+" using "+str(col_index)+":"+str(col_index_low)+":"+str(col_index_high)+":xtic(1) title '"+ titles[0] +"' with histogram")
    for i in range(1,nplots):
        out.write(",\\\n '' index "+str(indices[i])+" using "+str(col_index)+":"+str(col_index_low)+":"+str(col_index_high)+":xtic(1) title '"+titles[i]+"' with histogram")
    out.write("\n")


def curve(out,ofilename,title,indices,ifilename,x_range,y_label,col_index, titles):
    nplots = len(indices)
    out.write("reset\n")
    out.write("set grid\n")
    out.write("set term pdfcairo enhanced color font 'Helvetica,9'\n")
    out.write("set output \"" + ofilename + "\"\n")
    out.write("set termoption noenhanced\n")
    out.write("set tmargin 0\n")
    out.write("set bmargin 0\n")

    out.write("set ylabel \"" + y_label + "\"\n")
    out.write("set xrange [" + str(x_range[0]) + ":" + str(x_range[1]) + "]\n")
#    out.write("set yrange [1e-3:*]\n")
    out.write("set logscale y\n")
    out.write("set offsets 0.25, 0.25, 0, 0\n")
    out.write("set xtics rotate by -45\n")
    out.write("set size ratio 0.35\n")
    out.write("set style fill noborder\n")
    out.write("set title '" + title +"'\n")
    out.write("plot '"+ifilename+"' index "+str(indices[0])+" using "+str(col_index)+" with linespoints title '"+ titles[0] +"'")
    for i in range(1,nplots):
        out.write(",\\\n '' index "+str(indices[i])+" using "+str(col_index) + " with linespoints title '"+titles[i]+"'")
    out.write("\n")


def call(ifilename):
    cmdgnuplot = ["gnuplot", ifilename]
    proc = subprocess.Popen(cmdgnuplot)
    proc.wait()
    rc = proc.returncode
    if rc != 0:
        print('//rocsparse_bench_gnuplot_helper::call failed (err='+str(rc)+')')
        print('//rocsparse_bench_gnuplot_helper::note: check files \''+ifilename+'\'')
        exit(1)
