#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
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




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose',         required=False, default = False, action = "store_true")
    parser.add_argument('-t', '--tol',         required=True, default = 2.0,type=float)
    user_args, unknown_args = parser.parse_known_args()

    verbose=user_args.verbose
    percentage_tol = user_args.tol
    data = []
    num_files = len(unknown_args)

    titles = []
    for file_index in range(num_files):
        titles.append(os.path.basename(os.path.splitext(unknown_args[file_index])[0]))

    for file_index in range(num_files):
        with open(unknown_args[file_index],"r") as f:
            data.append(json.load(f))

    cmd = [d['cmdline'] for d in data]
    xargs = [d['xargs'] for d in data]
    yargs = [d['yargs'] for d in data]
    samples = [d['results'] for d in data]
    num_samples = len(samples[0])
    len_xargs = len(xargs[0])

    if verbose:
        print('//rocsparse-bench-regression CONFIG')
        for i in range(num_files):
            print('//rocsparse-bench-regression file'+str(i) +'      : \'' + unknown_args[i] + '\'')
        print('//rocsparse-bench-regression COMPARISON')

####
    for i in range(1,num_files):
        if xargs[0] != xargs[i]:
            print('xargs\'s must be equal, xargs from file \''+unknown_args[i]+'\' is not equal to xargs from file \''+unknown_args[0]+'\'')
            exit(1)

    if verbose:
        print('//rocsparse-bench-regression  -  xargs checked.')
####
    for i in range(1,num_files):
        if yargs[0] != yargs[i]:
            print('yargs\'s must be equal, yargs from file \''+unknown_args[i]+'\' is not equal to yargs from file \''+unknown_args[0]+'\'')
            exit(1)
    if verbose:
        print('//rocsparse-bench-regression  -  yargs checked.')
####
    for i in range(1,num_files):
        if num_samples != len(samples[i]):
            print('num_samples\'s must be equal, num_samples from file \''+unknown_args[i]+'\' is not equal to num_samples from file \''+unknown_args[0]+'\'')
            exit(1)
    if verbose:
        print('//rocsparse-bench-regression  -  num samples checked.')
####
    if verbose:
        print('//rocsparse-bench-regression percentage_tol: ' + str(percentage_tol) + '%')

    global_regression=False
    for file_index in range(1,num_files):
        print("//rocsparse-bench-regression - '"+ unknown_args[file_index])
        for iplot in range(len(yargs[0])):
            print('//rocsparse-bench-regression plot index ' + str(iplot) + ': \'' + yargs[0][iplot] + '\'',end='')
            mx_rel_flops=0
            mx_rel_time=0
            mx_rel_bandwidth=0
            mn_rel_flops=0
            mn_rel_time=0
            mn_rel_bandwidth=0
            regression=False
            for ixarg  in range(len_xargs):
                isample = iplot * len_xargs + ixarg
                tg = samples[file_index][isample]["timing"]
                tg0=samples[0][isample]["timing"]
                rel_flops = 100*(float(tg["flops"][0])-float(tg0["flops"][0]))/float(tg0["flops"][0])

                rel_time = 100*(float(tg["time"][0])-float(tg0["time"][0]))/float(tg0["time"][0])
                rel_bandwidth = 100*(float(tg["bandwidth"][0])-float(tg0["bandwidth"][0]))/float(tg0["bandwidth"][0])

                if ixarg > 0:
                    mx_rel_flops=max(mx_rel_flops,rel_flops)
                    mn_rel_flops=min(mn_rel_flops,rel_flops)

                    mx_rel_time=max(mx_rel_time,rel_time)
                    mn_rel_time=min(mn_rel_time,rel_time)

                    mx_rel_bandwidth=max(mx_rel_bandwidth,rel_bandwidth)
                    mn_rel_bandwidth=min(mn_rel_bandwidth,rel_bandwidth)
                else:
                    mx_rel_flops=rel_flops
                    mn_rel_flops=rel_flops

                    mx_rel_time=rel_time
                    mn_rel_time=rel_time

                    mx_rel_bandwidth=rel_bandwidth
                    mn_rel_bandwidth=rel_bandwidth

                if (rel_flops < -percentage_tol) or (rel_time < -percentage_tol) or (rel_bandwidth < -percentage_tol):
                    print("")
                if (rel_flops < -percentage_tol):
                    regression=True
                    print("//rocsparse-bench-regression   FAIL flops exceeds tolerance of  "  +  str(percentage_tol) + "%, [" +"{:.2f}".format(mn_rel_flops) + "," + "{:.2f}".format(mx_rel_flops) + "] from '" + xargs[file_index][ixarg] + "'")
                if (rel_time < -percentage_tol):
                    regression=True
                    print("//rocsparse-bench-regression   FAIL time exceeds tolerance of  "  +  str(percentage_tol) + "%, [" +"{:.2f}".format(mn_rel_time) + "," + "{:.2f}".format(mx_rel_time) + "] from '" + xargs[file_index][ixarg] + "'")
                if (rel_bandwidth < -percentage_tol):
                    regression=True
                    print("//rocsparse-bench-regression   FAIL bandwidth exceeds tolerance of  "  +  str(percentage_tol) + "%, [" +"{:.2f}".format(mn_rel_bandwidth) + "," + "{:.2f}".format(mx_rel_bandwidth) + "] from '" + xargs[file_index][ixarg] + "'")

            if regression:
                global_regression=True
            if regression:
                print('//rocsparse-bench-regression plot index ' + str(iplot) + ': \'' + yargs[0][iplot] + '\' FAILED')
            else:
                print("   PASSED")
        if verbose:
            print("//rocsparse-bench-regression    flops [" +"{:.2f}".format(mn_rel_flops) + "," + "{:.2f}".format(mx_rel_flops) + "], " + "time [" +"{:.2f}".format(mn_rel_time) + "," + "{:.2f}".format(mx_rel_time) + "], " + "bandwidth [" +"{:.2f}".format(mn_rel_bandwidth) + "," + "{:.2f}".format(mx_rel_bandwidth) + "]")
    if global_regression:
        exit(1)
    else:
        exit(0)
if __name__ == "__main__":
    main()

