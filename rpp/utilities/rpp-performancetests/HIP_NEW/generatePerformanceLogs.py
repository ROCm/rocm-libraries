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
parser.add_argument('--profiling', type=str, default='NO', help='Run with profiler? - (YES/NO)')
parser.add_argument("--case_start", type=str, default="0", help="Testing range starting case # - (0:86)")
parser.add_argument("--case_end", type=str, default="86", help="Testing range ending case # - (0:86)")
args = parser.parse_args()

profilingOption = args.profiling
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

if profilingOption == "NO":

    subprocess.call(["./rawLogsGenScript.sh", "0", caseStart, caseEnd])

    log_file_list = [
        "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/BatchPD_hip_pkd3_hip_raw_performance_log.txt",
        "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/BatchPD_hip_pln3_hip_raw_performance_log.txt",
        "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/BatchPD_hip_pln1_hip_raw_performance_log.txt",
        "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/Tensor_hip_pkd3_hip_raw_performance_log.txt",
        "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/Tensor_hip_pln3_hip_raw_performance_log.txt",
        "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/Tensor_hip_pln1_hip_raw_performance_log.txt"
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

        # Open log file
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

        # Close log file
        f.close()

elif profilingOption == "YES":

    NEW_FUNC_GROUP_LIST = [0, 15, 20, 29, 36, 40, 42, 49, 56, 65, 69]

    # Functionality group finder
    def func_group_finder(case_number):
        if case_number == 0:
            return "image_augmentations"
        elif case_number == 15:
            return "statistical_functions"
        elif case_number == 20:
            return "geometry_transforms"
        elif case_number == 29:
            return "advanced_augmentations"
        elif case_number == 36:
            return "fused_functions"
        elif case_number == 40:
            return "morphological_transforms"
        elif case_number == 42:
            return "color_model_conversions"
        elif case_number == 49:
            return "filter_operations"
        elif case_number == 56:
            return "arithmetic_operations"
        elif case_number == 65:
            return "logical_operations"
        elif case_number == 69:
            return "computer_vision"

    subprocess.call(["./rawLogsGenScript.sh", "1", caseStart, caseEnd])

    RESULTS_DIR = "../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE_BATCHPD_PKD3 = RESULTS_DIR + "/consolidated_results_BatchPD_PKD3.stats.csv"
    CONSOLIDATED_FILE_BATCHPD_PLN1 = RESULTS_DIR + "/consolidated_results_BatchPD_PLN1.stats.csv"
    CONSOLIDATED_FILE_BATCHPD_PLN3 = RESULTS_DIR + "/consolidated_results_BatchPD_PLN3.stats.csv"
    CONSOLIDATED_FILE_TENSOR_PKD3 = RESULTS_DIR + "/consolidated_results_Tensor_PKD3.stats.csv"
    CONSOLIDATED_FILE_TENSOR_PLN1 = RESULTS_DIR + "/consolidated_results_Tensor_PLN1.stats.csv"
    CONSOLIDATED_FILE_TENSOR_PLN3 = RESULTS_DIR + "/consolidated_results_Tensor_PLN3.stats.csv"

    TYPE_LIST = ["BatchPD_PKD3", "BatchPD_PLN1", "BatchPD_PLN3", "Tensor_PKD3", "Tensor_PLN1", "Tensor_PLN3"]
    BATCHPD_TYPE_LIST = ["BatchPD_PKD3", "BatchPD_PLN1", "BatchPD_PLN3"]
    TENSOR_TYPE_LIST = ["Tensor_PKD3", "Tensor_PLN1", "Tensor_PLN3"]
    CASE_NUM_LIST = range(int(caseStart), int(caseEnd) + 1, 1)
    BIT_DEPTH_LIST = range(0, 7, 1)
    OFT_LIST = range(0, 2, 1)
    d_counter = {"BatchPD_PKD3":0, "BatchPD_PLN1":0, "BatchPD_PLN3":0, "Tensor_PKD3":0, "Tensor_PLN1":0, "Tensor_PLN3":0}

    for TYPE in TYPE_LIST:

        # Open csv file
        new_file = open(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv",'w')
        new_file.write('"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

        prev=""

        # Loop through cases
        for CASE_NUM in CASE_NUM_LIST:

            # Add functionality group header
            if CASE_NUM in NEW_FUNC_GROUP_LIST:
                FUNC_GROUP = func_group_finder(CASE_NUM)
                new_file.write("0,0,0,0,0\n")
                new_file.write(FUNC_GROUP + ",0,0,0,0\n")
                new_file.write("0,0,0,0,0\n")

            # Set results directory
            CASE_RESULTS_DIR = RESULTS_DIR + "/" + TYPE + "/case_" + str(CASE_NUM)
            print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)

            # Loop through bit depths
            for BIT_DEPTH in BIT_DEPTH_LIST:

                # Loop through output format toggle cases
                for OFT in OFT_LIST:

                    if (CASE_NUM == 40 or CASE_NUM == 41 or CASE_NUM == 49) and TYPE.startswith("Tensor"):
                        KSIZE_LIST = [3, 5, 7, 9]
                        # Loop through extra param kSize
                        for KSIZE in KSIZE_LIST:
                            # Write into csv file
                            CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + "_kSize" + str(KSIZE) + ".stats.csv"
                            print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                            try:
                                case_file = open(CASE_FILE_PATH,'r')
                                for line in case_file:
                                    print(line)
                                    if not(line.startswith('"Name"')):
                                        if TYPE in TENSOR_TYPE_LIST:
                                            new_file.write(line)
                                            d_counter[TYPE] = d_counter[TYPE] + 1
                                        elif TYPE in BATCHPD_TYPE_LIST:
                                            if prev != line.split(",")[0]:
                                                new_file.write(line)
                                                prev = line.split(",")[0]
                                                d_counter[TYPE] = d_counter[TYPE] + 1
                                case_file.close()
                            except IOError:
                                print("Unable to open case results")
                                continue
                    elif (CASE_NUM == 24 or CASE_NUM == 21) and TYPE.startswith("Tensor"):
                        INTERPOLATIONTYPE_LIST = [0, 1, 2, 3, 4, 5]
                        # Loop through extra param interpolationType
                        for INTERPOLATIONTYPE in INTERPOLATIONTYPE_LIST:
                            # Write into csv file
                            CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + "_interpolationType" + str(INTERPOLATIONTYPE) + ".stats.csv"
                            print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                            try:
                                case_file = open(CASE_FILE_PATH,'r')
                                for line in case_file:
                                    print(line)
                                    if not(line.startswith('"Name"')):
                                        if TYPE in TENSOR_TYPE_LIST:
                                            new_file.write(line)
                                            d_counter[TYPE] = d_counter[TYPE] + 1
                                        elif TYPE in BATCHPD_TYPE_LIST:
                                            if prev != line.split(",")[0]:
                                                new_file.write(line)
                                                prev = line.split(",")[0]
                                                d_counter[TYPE] = d_counter[TYPE] + 1
                                case_file.close()
                            except IOError:
                                print("Unable to open case results")
                                continue
                    elif (CASE_NUM == 8) and TYPE.startswith("Tensor"):
                        NOISETYPE_LIST = [0, 1, 2]
                        # Loop through extra param noiseType
                        for NOISETYPE in NOISETYPE_LIST:
                            # Write into csv file
                            CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + "_noiseType" + str(NOISETYPE) + ".stats.csv"
                            print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                            try:
                                case_file = open(CASE_FILE_PATH,'r')
                                for line in case_file:
                                    print(line)
                                    if not(line.startswith('"Name"')):
                                        if TYPE in TENSOR_TYPE_LIST:
                                            new_file.write(line)
                                            d_counter[TYPE] = d_counter[TYPE] + 1
                                        elif TYPE in BATCHPD_TYPE_LIST:
                                            if prev != line.split(",")[0]:
                                                new_file.write(line)
                                                prev = line.split(",")[0]
                                                d_counter[TYPE] = d_counter[TYPE] + 1
                                case_file.close()
                            except IOError:
                                print("Unable to open case results")
                                continue
                    else:
                        # Write into csv file
                        CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + "_oft" + str(OFT) + ".stats.csv"
                        print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                        try:
                            case_file = open(CASE_FILE_PATH,'r')
                            for line in case_file:
                                print(line)
                                if not(line.startswith('"Name"')):
                                    if TYPE in TENSOR_TYPE_LIST:
                                        new_file.write(line)
                                        d_counter[TYPE] = d_counter[TYPE] + 1
                                    elif TYPE in BATCHPD_TYPE_LIST:
                                        if prev != line.split(",")[0]:
                                            new_file.write(line)
                                            prev = line.split(",")[0]
                                            d_counter[TYPE] = d_counter[TYPE] + 1
                            case_file.close()
                        except IOError:
                            print("Unable to open case results")
                            continue

        new_file.close()
        os.system('chown $USER:$USER ' + RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")

    try:
        import pandas as pd
        pd.options.display.max_rows = None

        # Generate performance report
        for TYPE in TYPE_LIST:
            print("\n\n\nKernels tested - ", d_counter[TYPE], "\n\n")
            df = pd.read_csv(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")
            df["AverageMs"] = df["AverageNs"] / 1000000
            dfPrint = df.drop(['Percentage'], axis=1)
            dfPrint["HIP Kernel Name"] = dfPrint.iloc[:,0].str.lstrip("Hip_")
            dfPrint_noIndices = dfPrint.astype(str)
            dfPrint_noIndices.replace(['0', '0.0'], '', inplace=True)
            dfPrint_noIndices = dfPrint_noIndices.to_string(index=False)
            print(dfPrint_noIndices)

    except ImportError:
        print("\nPandas not available! Results of GPU profiling experiment are available in the following files:\n" + \
            CONSOLIDATED_FILE_BATCHPD_PKD3 + "\n" + \
                CONSOLIDATED_FILE_BATCHPD_PLN1 + "\n" + \
                    CONSOLIDATED_FILE_BATCHPD_PLN3 + "\n" + \
                        CONSOLIDATED_FILE_TENSOR_PKD3 + "\n" + \
                            CONSOLIDATED_FILE_TENSOR_PLN1 + "\n" + \
                                CONSOLIDATED_FILE_TENSOR_PLN3 + "\n")

    except IOError:
        print("Unable to open results in " + RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")
