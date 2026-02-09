#!/usr/bin/env python3

# ########################################################################
# Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
import json
import glob
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="Execute a set of rocsparse-bench commands from a .json file", epilog="The .json file must contain an array 'cmdlines' of strings, where each string contains the list of options to pass to rocsparse-bench to execute a test.\nExample:\n {\n   \"cmdlines\": [\"-f csrmv\",\n                \"-f csrmm\"]\n }\n")
    parser.add_argument('-w', '--workingdir',     required=False, default = './')
    parser.add_argument('-v', '--verbose',         required=False, default = False, action = "store_true")
    parser.add_argument('-d', '--device',         required=False, default = 0)

    user_args, unknown_args = parser.parse_known_args()
    verbose=user_args.verbose
    workingdir = user_args.workingdir
    dev=user_args.device
    datadir=os.getenv('ROCSPARSE_BENCH_DATA_DIR')
    if datadir == None:
        print('//rocsparse-bench-execute:error You must define environment variable ROCSPARSE_BENCH_DATA_DIR as the directory of sparse matrices.')
        print('//rocsparse-bench-execute:error   export ROCSPARSE_BENCH_DATA_DIR=<where-to-find-sparse-matrices>')
        exit(1)
    if verbose:
        print('//rocsparse-bench-execute:ROCSPARSE_BENCH_DATA_DIR ' + datadir)

    if len(unknown_args) > 1:
        print('expecting only one input file.')
    with open(unknown_args[0],"r") as f:
        case=json.load(f)
    cmdlines = case['cmdlines']
    num_cmdlines = len(cmdlines)
    progname = "rocsparse-bench"
    prog = os.path.join(workingdir, progname)
    if not os.path.isfile(prog):
        print("**** Error: unable to find " + prog)
        sys.exit(1)

    for i in range(num_cmdlines):
        # execute the cmdline
        full_cmd = prog + " " + cmdlines[i];
        full_cmd=full_cmd.replace('$dev',str(dev))
        full_cmd=full_cmd.replace('${dev}',str(dev))
        print('//rocsparse-bench-execute:verbose:execute command "' + full_cmd+ '"')
        full_cmd=full_cmd.split(' ')
        subprocess_arg=[]
        for w in full_cmd:
            w=w.replace('${ROCSPARSE_BENCH_DATA_DIR}',datadir)
            gw=glob.glob(w)
            if (len(gw)!=0):
                for g in gw:
                    subprocess_arg.append(g)
            else:
                subprocess_arg.append(w)

        if verbose:
            print('//rocsparse-bench-execute:verbose:execute command with glob "' + ' '.join(subprocess_arg) + '"')
        proc = subprocess.Popen(subprocess_arg)
        proc.wait()
        rc = proc.returncode
        if rc != 0:
            print('//rocsparse-bench-execute:failure (err='+str(rc)+')')
            exit(1)

if __name__ == "__main__":
    main()
