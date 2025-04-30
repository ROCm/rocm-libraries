#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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


def defcurves(out,ofilename,title,ifilename,x_label, y_label, titles):
    nplots = len(titles)
    out.write("reset\n")
    out.write("set grid\n")
    out.write("set term pdfcairo enhanced color font 'Helvetica,9'\n")
    out.write("set output \"" + ofilename + "\"\n")
    out.write("set termoption noenhanced\n")
    out.write("set tmargin 0\n")
    out.write("set bmargin 0\n")
    out.write("set xlabel \"" + x_label + "\"\n")
    out.write("set ylabel \"" + y_label + "\"\n")
#    out.write("set logscale x\n")
    out.write("set logscale y\n")
    out.write("set offsets 0.25, 0.25, 0, 0\n")
    out.write("set xtics rotate by -45\n")
    out.write("set size ratio 0.35\n")
    out.write("set style fill noborder\n")
    out.write("set title '" + title +"'\n")
    out.write("plot '"+ifilename+"' using 1:2 with lines title '"+ titles[0] +"'")
    out.write(", '' "+" using 1:3 with lines title '"+titles[1]+"'")
    out.write(", '' "+" using 1:4 with lines title '"+titles[2]+"'")
#    out.write("plot '"+ifilename+"' using 1:"+str(2)+" with lines title '"+ titles[0] +"'")
#    out.write(", '' "+" using 1:" + str(3) + " with lines title '"+titles[1]+"'")
#    for i in range(len(titles)):
#
#            if (i>0):
#            out.write(",\\\n '' "+" using 1:" + str(i+1) + " with linespoints title '"+titles[i]+"'")
#        else:
#            out.write("plot '"+ifilename+" using 1:"+str(2)+" with linespoints title '"+ titles[0] +"'")
#        out.write("\n")



import rocsparse_bench_gnuplot_helper

#
# EXPORT TO PDF WITH GNUPLOT
#
##
def export_gnuplot(obasename, legend, results, verbose = False,debug = False):

    datafile = open(obasename + ".dat", "w+")
    for i in range(len(results)):
        tg = results[i]
        datafile.write(str(float(tg["time"])/float(results[len(results)-1]["time"])) + " " +
                       str(float(tg["nbytes_host"])/1024/1024) + " " + str(float(tg["nbytes_device"])/1024/1024) +  " " + str(float(tg["nbytes_managed"])/1024/1024) + "\n")
    datafile.close();

    if verbose:
        print('//rocsparse-memstat-plot  -  write gnuplot file : \'' + obasename + '.gnuplot\'')

    cmdfile = open(obasename + ".gnuplot", "w+")
    # for each plot
    num_curves=3
    filetype="pdf"
    filename_extension= "." + filetype

    defcurves(cmdfile,
              obasename + filename_extension,
              'Memory',
              obasename + ".dat",
              "operations",
              "Mbytes",
              ["host","device","managed"])
    cmdfile.close();

    rocsparse_bench_gnuplot_helper.call(obasename + ".gnuplot")
    if verbose:
        print('//rocsparse-memstat-plot CLEANING')

    if not debug:
        os.remove(obasename + '.dat')
        os.remove(obasename + '.gnuplot')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workingdir',     required=False, default = './')
    parser.add_argument('-o', '--obasename',    required=False, default = 'a')
    parser.add_argument('-v', '--verbose',         required=False, default = False, action = "store_true")
    parser.add_argument('-d', '--debug',         required=False, default = False, action = "store_true")
    user_args, unknown_args = parser.parse_known_args()
    verbose=user_args.verbose
    debug=user_args.debug
    obasename = user_args.obasename
    if len(unknown_args) > 1:
        print('expecting only one input file.')
    with open(unknown_args[0],"r") as f:
        case=json.load(f)

    results = case['results']
    legend =  case['legend']
    if verbose:
        print('//rocsparse-memstat-plot')
        print('//rocsparse-memstat-plot  - file : \'' + unknown_args[0] + '\'')

    export_gnuplot( obasename, legend, results, verbose,debug)

if __name__ == "__main__":
    main()

