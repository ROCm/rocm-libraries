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

#
#
#
def export_ptsdata(ibasename, obasename,xargs, yargs, results,verbose = False):

    len_xargs = len(xargs)
    for iplot in range(len(yargs)):
        one_only = True
        yarg=yargs[iplot]
        if yarg != "":
            yarg="_" + yarg
        yarg = yarg.replace("=","")
        yarg = yarg.replace(",","_")
        filename=obasename + yarg + ".csv"
        print("//rocsparse-bench-ptsdata  - writing into file '" + filename + "'")
        datafile = open(filename, "w+")
        for ixarg  in range(len_xargs):
            isample = iplot * len_xargs + ixarg
            tg = results[isample]["timing"]
            tg_raw_legend = ','.join(tg["raw_legend"].split())
            tg_raw = ','.join(tg["raw_data"].split())
        
            if len(tg_raw_legend) != 0:
                if verbose:
                    print('//rocsparse-bench-ptsdata  -  write pts data file : \'' + obasename + '.csv\'')
                if one_only:
                    one_only = False
                    datafile.write("test_name , " + tg_raw_legend + ",time, time_low, time_high, flops, flops_low, flops_high, bandwidth, bandwidth_low, bandwidth_high\n")
                datafile.write(ibasename + ", " + tg_raw + ", " +
                               tg["time"][0] + ", " +
                               tg["time"][1] + ", " +
                               tg["time"][2] + ", " +
                               tg["flops"][0] + ", " +
                               tg["flops"][1] + ", " +
                               tg["flops"][2] + ", " +
                               tg["bandwidth"][0] + ", " +
                               tg["bandwidth"][1] + ", "+
                               tg["bandwidth"][2] + "\n")
    datafile.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="Convert a rocsparse benchmark .json file to a csv file.")
    parser.add_argument('-o', '--obasename',    required=False, default = 'a')
    parser.add_argument('-v', '--verbose',         required=False, default = False, action = "store_true")
    user_args, unknown_args = parser.parse_known_args()
    verbose=user_args.verbose
    obasename = user_args.obasename
    if len(unknown_args) > 1:
        print('expecting only one input file.')
    ibasename = os.path.basename(os.path.splitext(unknown_args[0])[0])
    with open(unknown_args[0],"r") as f:
        case=json.load(f)

    cmd = case['cmdline']
    xargs = case['xargs']
    yargs = case['yargs']
    results = case['results']
    num_samples = len(results)
    len_xargs = len(xargs)

    if verbose:
        print('//rocsparse-bench-ptsdata')
        print('//rocsparse-bench-ptsdata  - file : \'' + unknown_args[0] + '\'')

    export_ptsdata( ibasename, obasename, xargs,yargs, results, verbose)

if __name__ == "__main__":
    main()

