#!/usr/bin/python3

# ########################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

"""Copyright 2025 Advanced Micro Devices, Inc. All rights Reserved.
Run tests on build"""

import os
import sys
import argparse
import pathlib
import subprocess


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""
    Checks build arguments
    """)

    # Mutually exclusive group for --test and --emulation
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(      '--emulation', type=str, choices=['smoke', 'regression', 'extended'],
                        help='Enable specific emulation test mode, e.g. smoke test')
    parser.add_argument(      '--install_dir', type=str, required=False, default="",
                        help='Installation directory where build or release folders are (optional, default: $PWD)')

    args = parser.parse_args()

    return args


def run_cmd(args, filter):

    test_binary = ""
    if args.install_dir :
        test_binary = os.path.join(args.install_dir, "hipsparselt-test")
    else:
        test_binary = os.path.join(pathlib.os.curdir, "hipsparselt-test")

    if not os.path.isfile(test_binary):
        return

    sub_env = os.environ.copy()
    sub_env["PATH"] = os.getcwd() + os.pathsep + sub_env["PATH"]

    cmd = [test_binary, filter]
    test_proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, env=sub_env)

    return test_proc.returncode


def run_test(args):

    cwd = os.curdir

    if args.emulation == "smoke":
        run_cmd(args, "--gtest_filter=*smoke*")
    elif args.emulation == "regression":
        run_cmd(args, "--gtest_filter=*quick*")
    elif args.emulation == "extended":
         print("There is no test for extended test cases.")

    if (os.curdir != cwd):
        os.chdir( cwd )

    return 0

def main():

    args = parse_args()

    status = run_test(args)

    sys.exit


if __name__ == '__main__':
    main()
