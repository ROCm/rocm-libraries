################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import argparse
import csv
import pathlib
import sys

from collections import defaultdict

def gather_data(indexFile):

    indexData = defaultdict(set)
    with open(indexFile, "rt") as csvfile:
        indexreader = csv.DictReader(csvfile, delimiter=",")
        for row in indexreader: # read a row as {column1: value1, column2: value2,...}
            for key, value in row.items():
                indexData[key].add(value)
    return indexData

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("library_path", help="Tensile library path")
    args = argParser.parse_args()
    libraryPath = args.library_path
    indexFileName = "TensileMasterSolutionIndex.csv"

    # Check that path exists
    if not pathlib.Path(libraryPath).is_dir():
        print(f"ERROR: {libraryPath} does not exists.")
        sys.exit(1)

    # Check that TensileMasterSolutionIndex.csv exists
    csvpath = pathlib.Path(libraryPath) / indexFileName
    if not csvpath.is_file():
        print(f"ERROR: {csvpath} does not exists.")
        sys.exit(1)

    data = gather_data(csvpath)

    # List files in library path
    datFiles = [f.stem for f in pathlib.Path(libraryPath).glob("*.dat")]
    coFiles = [f.stem for f in pathlib.Path(libraryPath).glob("*.co")]
    lazyArchFiles = [f for f in datFiles if "_lazy_" in f]
    metaDataFiles = [f for f in datFiles if not "_lazy_" in f]
    nonfallback = set([f for f in data['libraryName'] if not "fallback" in f])

    print(f"MetaData files should match library names in index file: {set(metaDataFiles) == data['libraryName']}")
    print(f"Asm files should match non-fallback library names in index file: {set(coFiles) == nonfallback}")
    print(f"Lazy library files should match number of architectures in index file: {len(set(lazyArchFiles)) == len(data['architectureName'])}")
