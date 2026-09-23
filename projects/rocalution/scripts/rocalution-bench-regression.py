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
        print('//rocalution-bench-regression CONFIG')
        for i in range(num_files):
            print('//rocalution-bench-regression file'+str(i) +'      : \'' + unknown_args[i] + '\'')
        print('//rocalution-bench-regression COMPARISON')

####
    for i in range(1,num_files):
        if xargs[0] != xargs[i]:
            print('xargs\'s must be equal, xargs from file \''+unknown_args[i]+'\' is not equal to xargs from file \''+unknown_args[0]+'\'')
            exit(1)

    if verbose:
        print('//rocalution-bench-regression  -  xargs checked.')
####
    for i in range(1,num_files):
        if yargs[0] != yargs[i]:
            print('yargs\'s must be equal, yargs from file \''+unknown_args[i]+'\' is not equal to yargs from file \''+unknown_args[0]+'\'')
            exit(1)
    if verbose:
        print('//rocalution-bench-regression  -  yargs checked.')
####
    for i in range(1,num_files):
        if num_samples != len(samples[i]):
            print('num_samples\'s must be equal, num_samples from file \''+unknown_args[i]+'\' is not equal to num_samples from file \''+unknown_args[0]+'\'')
            exit(1)
    if verbose:
        print('//rocalution-bench-regression  -  num samples checked.')
####
    if verbose:
        print('//rocalution-bench-regression percentage_tol: ' + str(percentage_tol) + '%')

    global_regression=False
    global_improvement=False
    for file_index in range(1,num_files):
        print("//rocalution-bench-regression - '"+ unknown_args[file_index])
        for iplot in range(len(yargs[0])):
            print('//rocalution-bench-regression plot index ' + str(iplot) + ': \'' + yargs[0][iplot] + '\'',end='')
            mx_rel_time_analyze=0
            mx_rel_time_solve=0
            mx_rel_iter=0
            mx_rel_norm_residual=0
            mn_rel_time_analyze=0
            mn_rel_time_solve=0
            mn_rel_iter=0
            mn_rel_norm_residual=0
            regression=False
            improvement=False
            regression_time_solve=0
            regression_time_analyze=0
            regression_iter=0
            regression_norm_residual=0
            for ixarg  in range(len_xargs):
                isample = iplot * len_xargs + ixarg
                tg = samples[file_index][isample]["timing"]
                tg0=samples[0][isample]["timing"]
                rel_time_analyze = 100*(float(tg["median"]["time_analyze"])-float(tg0["median"]["time_analyze"]))/float(tg0["median"]["time_analyze"])
                rel_time_solve = 100*(float(tg["median"]["time_solve"])-float(tg0["median"]["time_solve"]))/float(tg0["median"]["time_solve"])
                rel_iter = 100*(float(tg["median"]["iter"])-float(tg0["median"]["iter"]))/float(tg0["median"]["iter"])
                rel_norm_residual = 100*(float(tg["median"]["norm_residual"])-float(tg0["median"]["norm_residual"]))/float(tg0["median"]["norm_residual"])

                if ixarg > 0:
                    mx_rel_time_solve=max(mx_rel_time_solve,rel_time_solve)
                    mn_rel_time_solve=min(mn_rel_time_solve,rel_time_solve)

                    mx_rel_time_analyze=max(mx_rel_time_analyze,rel_time_analyze)
                    mn_rel_time_analyze=min(mn_rel_time_analyze,rel_time_analyze)

                    mx_rel_iter=max(mx_rel_iter,rel_iter)
                    mn_rel_iter=min(mn_rel_iter,rel_iter)

                    mx_rel_norm_residual=max(mx_rel_norm_residual,rel_norm_residual)
                    mn_rel_norm_residual=min(mn_rel_norm_residual,rel_norm_residual)
                else:

                    mx_rel_norm_residual=rel_norm_residual
                    mn_rel_norm_residual=rel_norm_residual

                    mx_rel_iter=rel_iter
                    mn_rel_iter=rel_iter

                    mx_rel_time_analyze=rel_time_analyze
                    mn_rel_time_analyze=rel_time_analyze

                    mx_rel_time_solve=rel_time_solve
                    mn_rel_time_solve=rel_time_solve

                if (rel_time_solve > percentage_tol):
                    regression_time_solve = -1
                elif (rel_time_solve < -percentage_tol):
                    regression_time_solve = 1

                if (rel_time_analyze > percentage_tol):
                    regression_time_analyze = -1
                elif (rel_time_analyze < -percentage_tol):
                    regression_time_analyze = 1

                if (rel_iter > percentage_tol):
                    regression_iter = -1
                elif (rel_iter < -percentage_tol):
                    regression_iter = 1

                if (rel_norm_residual > percentage_tol):
                    regression_norm_residual = -1
                elif (rel_norm_residual < -percentage_tol):
                    regression_norm_residual = 1

                if (regression_time_solve == -1) or (regression_time_analyze == -1) or (regression_iter == -1) or (regression_norm_residual == -1):
                    print("")
                if (regression_time_solve == 1):
                    improvement=True
                if (regression_time_analyze == 1):
                    improvement=True
                if (regression_iter == 1):
                    improvement=True
                if (regression_norm_residual == 1):
                    improvement=True

                if (regression_time_solve == -1):
                    improvement=False
                    regression=True
                    print("//rocalution-bench-regression   FAIL time_solve exceeds tolerance of  "  +  str(percentage_tol) + "%, [" +"{:.2f}".format(mn_rel_time_solve) + "," + "{:.2f}".format(mx_rel_time_solve) + "] from '" + xargs[file_index][ixarg] + "'")

                if (regression_time_analyze == -1):
                    improvement=False
                    regression=True
                    print("//rocalution-bench-regression   FAIL time_analyze exceeds tolerance of  "  +  str(percentage_tol) + "%, [" +"{:.2f}".format(mn_rel_time_analyze) + "," + "{:.2f}".format(mx_rel_time_analyze) + "] from '" + xargs[file_index][ixarg] + "'")

                if (regression_iter == -1):
                    improvement=False
                    regression=True
                    print("//rocalution-bench-regression   FAIL iter exceeds tolerance of  "  +  str(percentage_tol) + "%, [" +"{:.2f}".format(mn_rel_iter) + "," + "{:.2f}".format(mx_rel_iter) + "] from '" + xargs[file_index][ixarg] + "'")

                if (regression_norm_residual == -1):
                    improvement=False
                    regression=True
                    print("//rocalution-bench-regression   FAIL norm_residual exceeds tolerance of  "  +  str(percentage_tol) + "%, [" +"{:.2f}".format(mn_rel_norm_residual) + "," + "{:.2f}".format(mx_rel_norm_residual) + "] from '" + xargs[file_index][ixarg] + "'")

            if not improvement:
                global_improvement=False

            if regression:
                global_regression=True
            if regression:
                print("   FAILED")
                print('//rocalution-bench-regression plot index ' + str(iplot) + ': \'' + yargs[0][iplot] + '\' FAILED')
            else:
                if improvement:
                    print("   IMPROVED")
                else:
                    print("   PASSED")
        if verbose:
            print("//rocalution-bench-regression    time_solve [" +"{:.2f}".format(mn_rel_time_solve) + "," + "{:.2f}".format(mx_rel_time_solve) + "], " + "time_analyze [" +"{:.2f}".format(mn_rel_time_analyze) + "," + "{:.2f}".format(mx_rel_time_analyze) + "], " + "iter [" +"{:.2f}".format(mn_rel_iter) + "," + "{:.2f}".format(mx_rel_iter) + "], " + "norm_residual [" +"{:.2f}".format(mn_rel_norm_residual) + "," + "{:.2f}".format(mx_rel_norm_residual) )
    if global_regression:
        exit(1)
    else:
        exit(0)
if __name__ == "__main__":
    main()

