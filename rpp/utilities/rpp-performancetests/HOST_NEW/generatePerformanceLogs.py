"""
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--case_start", type=str, default="0", help="Testing range starting case # - (0:86)")
parser.add_argument("--case_end", type=str, default="86", help="Testing range ending case # - (0:86)")
args = parser.parse_args()

caseStart = args.case_start
caseEnd = args.case_end

if caseEnd < caseStart:
    print("Ending case# must be greater than starting case#. Aborting!")
    exit(0)

if caseStart < "0" or caseStart > "86":
    print("Starting case# must be in the 0:86 range. Aborting!")
    exit(0)

if caseEnd < "0" or caseEnd > "86":
    print("Ending case# must be in the 0:86 range. Aborting!")
    exit(0)

subprocess.call(["./rawLogsGenScript.sh", caseStart, caseEnd])

log_file_list = [
    "../OUTPUT_PERFORMANCE_LOGS_HOST_NEW/BatchPD_host_pkd3_host_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_HOST_NEW/BatchPD_host_pln3_host_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_HOST_NEW/BatchPD_host_pln1_host_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_HOST_NEW/Tensor_host_pkd3_host_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_HOST_NEW/Tensor_host_pln3_host_raw_performance_log.txt",
    "../OUTPUT_PERFORMANCE_LOGS_HOST_NEW/Tensor_host_pln1_host_raw_performance_log.txt"
    ]

functionality_group_list = [
    "image_augmentations",
    "statistical_functions",
    "geometry_transforms",
    "advanced_augmentations",
    "fused_functions",
    "morphological_transforms",
    "color_model_conversions",
    "filter_operations",
    "arithmetic_operations",
    "logical_operations",
    "computer_vision"
]

for log_file in log_file_list:

    # Opening log file
    try:
        f = open(log_file,"r")
        print("\n\n\nOpened log file -> ", log_file)
    except IOError:
        print("Skipping file -> ", log_file)
        continue

    stats = []
    maxVals = []
    minVals = []
    avgVals = []
    functions = []
    frames = []
    prevLine = ""
    funcCount = 0

    # Loop over each line
    for line in f:
        for functionality_group in functionality_group_list:
            if functionality_group in line:
                functions.extend([" ", functionality_group, " "])
                frames.extend([" ", " ", " "])
                maxVals.extend([" ", " ", " "])
                minVals.extend([" ", " ", " "])
                avgVals.extend([" ", " ", " "])

        if "max,min,avg" in line:
            split_word_start = "Running "
            split_word_end = " 100"
            prevLine = prevLine.partition(split_word_start)[2].partition(split_word_end)[0]
            if prevLine not in functions:
                functions.append(prevLine)
                frames.append("100")
                split_word_start = "max,min,avg = "
                split_word_end = "\n"
                stats = line.partition(split_word_start)[2].partition(split_word_end)[0].split(",")
                maxVals.append(stats[0])
                minVals.append(stats[1])
                avgVals.append(stats[2])
                funcCount += 1

        if line != "\n":
            prevLine = line

    # Print log lengths
    print("Functionalities - ", funcCount)

    # Print summary of log
    print("\n\nFunctionality\t\t\t\t\t\tFrames Count\tmax(s)\t\tmin(s)\t\tavg(s)\n")
    if len(functions) != 0:
        maxCharLength = len(max(functions, key=len))
        functions = [x + (' ' * (maxCharLength - len(x))) for x in functions]
        for i, func in enumerate(functions):
            print(func, "\t", frames[i], "\t\t", maxVals[i], "\t", minVals[i], "\t", avgVals[i])
    else:
        print("No variants under this category")

    # Closing log file
    f.close()
