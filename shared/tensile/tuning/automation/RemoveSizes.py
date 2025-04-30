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

# Removes tuned sizes from a library logic file.
# Usage:
# $ python3 RemoveSizes.py [-v] <input lib logic> <output lib logic> <csv file with sizes>

from yaml import SafeDumper as yamlDumper
from yaml import SafeLoader as yamlLoader

import argparse
import csv
import yaml

def parseArgs():
    argParser = argparse.ArgumentParser()

    h = {"inLogic"  : "Input library logic file",
         "outLogic" : "Output library logic file",
         "sizeList" : "CSV file containing list of sizes to remove",
         "verbose"  : "Verbose output"
    }

    argParser.add_argument("inLogic", type=str, help=h["inLogic"])
    argParser.add_argument("outLogic", type=str, help=h["outLogic"])
    argParser.add_argument("sizeList", type=str, help=h["sizeList"])
    argParser.add_argument("--verbose", "-v", action="store_true", help=h["verbose"])

    return argParser.parse_args()

def main():
    args = parseArgs()
    if args.verbose:
        print("Removing tuned sizes")
        print("Input Logic : " + args.inLogic)
        print("Output Logic: " + args.outLogic)
        print("Sizes File  : " + args.sizeList)

    with open(args.inLogic) as inFile:
        logicData = yaml.load(inFile, yamlLoader)

    mapping = logicData[7]
    if args.verbose:
        print("Initial size count = {}".format(len(mapping)))

    with open(args.sizeList) as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            found = False
            m = int(row[0])
            n = int(row[1])
            b = int(row[2])
            k = int(row[3])
            for tune in mapping:
                size = tune[0]
                if size[0] == m and size[1] == n and size[2] == b and size[3] == k:
                    mapping.remove(tune)
                    found = True
                    break
            if args.verbose:
                print("{} {}".format([m, n, b, k], "removed" if found else "not found"))

    if args.verbose:
        print("Final size count = {}".format(len(mapping)))

    with open(args.outLogic, "w") as outFile:
        yaml.dump(logicData, outFile, yamlDumper, default_flow_style=None, sort_keys=False, width=5000)

    if args.verbose:
        print("Done writing new logic file")

if __name__ == "__main__":
    main()
