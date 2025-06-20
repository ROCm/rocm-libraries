#!/usr/bin/env python3

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import subprocess
import sys
import argparse
import tempfile


def main():
    print("main")

    parser = argparse.ArgumentParser(prog='rocfft_mpi_test')

    parser.add_argument('--worker',
                        type=str,
                        default=None,
                        help='mpi worker path')
    parser.add_argument('--rocffttest',
                        type=str,
                        default=None,
                        help='path to rocfft-test')
    parser.add_argument('--launcher', type=str, default=None, help='FIXME')
    parser.add_argument('--gpuidvar', type=str, default=None, help='FIXME')
    parser.add_argument('--gpusperrank',
                        type=int,
                        help='Gpus per rank',
                        default=1)
    parser.add_argument('--timeout',
                        type=int,
                        help='timeout in seconds',
                        default=5 * 60)
    parser.add_argument('--nranks',
                        type=int,
                        help='number of ranks',
                        required=True)

    args = parser.parse_args()

    print("nranks", args.nranks)
    print("gpus per rank", args.gpusperrank)

    print("Running token generation:")

    tokencmd = [
        args.rocffttest, "--gtest_filter=*multi_gpu*-*adhoc*", "--mp_lib",
        "mpi", "--mp_ranks",
        str(args.nranks), "--mp_launch", "foo", "--printtokens", "--ngpus",
        str(1)
    ]

    print(tokencmd)

    fout = tempfile.TemporaryFile(mode="w+")
    ptoken = subprocess.Popen(tokencmd, stdout=fout, stderr=subprocess.STDOUT)
    ptoken.wait()

    fout.seek(0)
    cout = fout.read()

    if ptoken.returncode != 0:
        print("token generation failed")
        print(cout)
        sys.exit(1)

    tokens = []
    foundtokenstart = False
    for line in cout.splitlines():
        if foundtokenstart:
            tokens.append(line)
        if line.startswith("Tokens:"):
            foundtokenstart = True

    print("Generated", len(tokens), "tokens")

    failedtokens = []
    for idx, token in enumerate(tokens):
        print("testing token", idx)
        workercmd = [args.worker] + ["--token", token]
        if args.gpuidvar != None:
            bashprecmd = "export ROCR_VISIBLE_DEVICES="
            for idx in range(args.gpusperrank):
                if (idx != 0):
                    bashprecmd += ","
                bashprecmd += "$(( " + str(
                    args.gpusperrank) + " * ${" + args.gpuidvar + "} + " + str(
                        idx) + " ))"
            bashprecmd += "; "
            cmd = ["bash", "-c", bashprecmd + " " + " ".join(workercmd)]
        else:
            cmd = workercmd

        allcmd = [args.launcher] + cmd
        print(allcmd)

        fout = tempfile.TemporaryFile(mode="w+")
        p = subprocess.Popen(allcmd, stdout=fout, stderr=subprocess.STDOUT)
        p.wait(args.timeout)

        fout.seek(0)
        cout = fout.read()
        print(cout)

        if p.returncode == 0:
            print("PASSED")
        else:
            print("FAILED")
            failedtokens.append(token)

    print("Failed", len(failedtokens), "tokens:")
    for token in failedtokens:
        print(token)

    print("Ran", len(tokens), "tokens, with", len(failedtokens), "failures.")

    sys.exit(len(failedtokens))


if __name__ == '__main__':
    main()
