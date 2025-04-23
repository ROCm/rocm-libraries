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

# Generates a CSV file listing sizes that have improved performance between rocblas-bench runs.
# To use, run rocblas-bench and save the output to a file.
# Run rocblas-bench again with the same problem sizes and a new library to be compared.
# Use this script to compare the two sets of results and generate a CSV containing a list of sizes that have improved.
# Usage:
# $ python3 ListSizes.py [-v] <first benchmark results> <second benchmark results> <output csv file>

import argparse
import csv

from decimal import Decimal

def parseArgs():
    argParser = argparse.ArgumentParser()

    h = {"baseBench" : "Results of baseline benchmark",
         "newBench"  : "Results of new benchmark",
         "sizeList"  : "Output CSV file listing sizes faster in new benchmark",
         "verbose"   : "Verbose output"
    }

    argParser.add_argument("baseBench", type=str, help=h["baseBench"])
    argParser.add_argument("newBench", type=str, help=h["newBench"])
    argParser.add_argument("sizeList", type=str, help=h["sizeList"])
    argParser.add_argument("--verbose", "-v", action="store_true", help=h["verbose"])

    return argParser.parse_args()

def main():
    args = parseArgs()
    if args.verbose:
        print("List winning sizes")
        print("Base benchmark : " + args.baseBench)
        print("New benchmark  : " + args.newBench)
        print("Sizes file     : " + args.sizeList)

    baseData = []
    with open(args.baseBench) as baseFile:
        for line in baseFile:
            if line.startswith("transA,"):
                labels = line.split(",")
                mIdx = labels.index("M")
                nIdx = labels.index("N")
                kIdx = labels.index("K")
                bIdx = labels.index("batch_count")
                gflopsIdx = labels.index("rocblas-Gflops")
                dataLine = next(baseFile)
                data = dataLine.split(",")
                data = [d.strip() for d in data]
                baseData.append([int(data[mIdx]), int(data[nIdx]), int(data[bIdx]), int(data[kIdx]), Decimal(data[gflopsIdx])])
                
    newData = []
    with open(args.newBench) as newFile:
        for line in newFile:
            if line.startswith("transA,"):
                labels = line.split(",")
                mIdx = labels.index("M")
                nIdx = labels.index("N")
                kIdx = labels.index("K")
                bIdx = labels.index("batch_count")
                gflopsIdx = labels.index("rocblas-Gflops")
                dataLine = next(newFile)
                data = dataLine.split(",")
                data = [d.strip() for d in data]
                newData.append([int(data[mIdx]), int(data[nIdx]), int(data[bIdx]), int(data[kIdx]), Decimal(data[gflopsIdx])])

    sizeData = []
    for n in newData:
        for b in baseData:
            if n[0] == b[0] and n[1] == b[1] and n[2] == b[2] and n[3] == b[3]:
                if n[4] >= b[4]:
                    sizeData.append([n[0], n[1], n[2], n[3]])
                    if args.verbose:
                        print("Adding: {}".format([n[0], n[1], n[2], n[3]]))
                break

    with open(args.sizeList, "w") as sizeFile:
        csvWriter = csv.writer(sizeFile)
        csvWriter.writerows(sizeData)

    if args.verbose:
        print("Done writing size list")

if __name__ == "__main__":
    main()
