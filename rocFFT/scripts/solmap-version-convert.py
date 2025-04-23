#!/usr/bin/env python3

# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

import argparse
import logging
import subprocess
import sys
import os
import pathlib

from os import listdir
from os.path import isfile, join
from pathlib import Path

console = logging.StreamHandler()


# #
# # Commands
# #
def version_check(arguments):
    """Check versions of solution map and do convertion"""

    input_folder = None

    if arguments.infolder:
        input_folder = Path(arguments.infolder)

    if (input_folder is None) or (not input_folder.exists()):
        print(
            "input folder is not set or not existing. use -infolder=/inputfolder/of/solutionmaps"
        )
        return

    if arguments.outfolder:
        output_folder = Path(arguments.outfolder)
        output_folder.mkdir(parents=True, exist_ok=True)
    else:
        print(
            "output folder is not set. use -outfolder=/subfolder/of/outputfolder"
        )
        return

    # pick files that are solution maps
    map_filenames = [
        f for f in listdir(input_folder)
        if isfile(join(input_folder, f)) and '_rocfft_solution_map.dat' in f
    ]

    # we'll do this in the cpp
    # os.environ['ROCFFT_USE_EMPTY_SOL_MAP'] = '1'

    for filename in map_filenames:
        print("checking file format version: " + str(filename))
        cmd = [pathlib.Path(arguments.converter).resolve()]
        cmd += ['--input_file', str(input_folder / filename)]
        cmd += ['--output_file', str(output_folder / filename)]
        cmd = [str(x) for x in cmd]
        logging.info('checking file format version: ' + ' '.join(cmd))

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        try:
            proc.wait(timeout=None)
        except subprocess.TimeoutExpired:
            logging.info("killed")
            proc.kill()
        if proc.returncode != 0:
            print('Failed on checking version: ' + ' '.join(cmd))

        for line in proc.stdout:
            line = line.decode('utf-8').rstrip('\n')
            print(line + '\n')


#
# Main
#
def main():
    parser = argparse.ArgumentParser(prog='solmap-version-convert')

    parser.add_argument(
        '--converter',
        type=str,
        help='converter executable path',
        default='./build/release/library/src/rocfft_solmap_convert')

    parser.add_argument(
        '--infolder',
        type=str,
        help=
        'folder of the original solution map data, default is [repo_folder]/solution_map/',
        default='./solution_map')

    parser.add_argument(
        '--outfolder',
        type=str,
        help=
        'folder of the version-coverted solution map data, default is [repo_folder]/solution_map/converted/',
        default='./solution_map/converted')

    arguments = parser.parse_args()

    version_check(arguments)

    sys.exit(0)


if __name__ == '__main__':
    logging.basicConfig(filename='solmap-version-convert.log',
                        format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.DEBUG)

    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter('%(levelname)-8s: %(message)s'))
    logging.getLogger('').addHandler(console)

    main()
