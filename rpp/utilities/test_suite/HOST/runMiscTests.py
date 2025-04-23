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
import subprocess  # nosec
import argparse
import datetime
import shutil
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.join(os.path.dirname( __file__ ), '..' ))
from common import *

# Set the timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
scriptPath = os.path.dirname(os.path.realpath(__file__))
outFolderPath = os.getcwd()
buildFolderPath = os.getcwd()
caseMin = 0
caseMax = 4
errorLog = [{"notExecutedFunctionality" : 0}]

# Get a list of log files based on a flag for preserving output
def get_log_file_list():
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_MISC_LOGS_HOST_" + timestamp + "/Tensor_misc_host_raw_performance_log.txt",
    ]

def run_unit_test_cmd(numDims, case, numRuns, testType, toggle, batchSize, outFilePath, additionalArg):
    bitDepths = range(7)
    if testType == 0:
        bitDepths = [2]
    for bitDepth in bitDepths:
        print("\n./Tensor_misc_host " + str(case) + " " + str(testType) + " " + str(toggle) + " " + str(numDims) + " " + str(batchSize) + " " + str(numRuns) + " " + str(additionalArg))
        result = subprocess.Popen([buildFolderPath + "/build/Tensor_misc_host", str(case), str(testType), str(toggle), str(numDims), str(batchSize), str(numRuns), str(bitDepth), str(additionalArg), outFilePath, scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
        log_detected(result, errorLog, miscAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_misc_func_name(int(case), numDims, additionalArg))
        print("------------------------------------------------------------------------------------------")

def run_performance_test_cmd(loggingFolder, numDims, case, numRuns, testType, toggle, batchSize, bitDepth, outFilePath, additionalArg):
    with open(loggingFolder + "/Tensor_misc_host_raw_performance_log.txt", "a") as logFile:
        logFile.write("./Tensor_misc_host " + str(case) + " " + str(testType) + " " + str(toggle) + " " + str(numDims) + " " + str(batchSize) + " " + str(numRuns) + " " + str(additionalArg) + "\n")
        process = subprocess.Popen([buildFolderPath + "/build/Tensor_misc_host", str(case), str(testType), str(toggle), str(numDims), str(batchSize), str(numRuns), str(bitDepth), str(additionalArg), outFilePath, scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
        read_from_subprocess_and_write_to_log(process, logFile)
        log_detected(process, errorLog, miscAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_misc_func_name(int(case), numDims, additionalArg))

def run_test(loggingFolder, numDims, case, numRuns, testType, toggle, batchSize, outFilePath, additionalArg = ""):
    if testType == 0:
        run_unit_test_cmd(numDims, case, numRuns, testType, toggle, batchSize, outFilePath, additionalArg)
    elif testType == 1:
        print("\n")
        bitDepths = range(7)
        for bitDepth in bitDepths:
            run_performance_test_cmd(loggingFolder, numDims, case, numRuns, testType, toggle, batchSize, bitDepth, outFilePath, additionalArg)

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_start", type = int, default = caseMin, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument("--case_end", type = int, default = caseMax, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = QA tests / 1 = Performance tests)")
    parser.add_argument('--toggle', type = int, default = 0, help = "Toggle outputs")
    parser.add_argument('--case_list', nargs = "+", help = "A list of specific case numbers to run separated by spaces", required = False)
    parser.add_argument("--num_dims", type = int, default = 2, help = "Number of dimensions for input")
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Outputs from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )" )
    print_case_list(miscAugmentationMap, "HOST", parser)
    args = parser.parse_args()

    # validate the parameters passed by user
    if ((args.case_start < caseMin or args.case_start > caseMax) or (args.case_end < caseMin or args.case_end > caseMax)):
        print("Starting case# and Ending case# must be in the 0:1 range. Aborting!")
        exit(0)
    elif args.case_end < args.case_start:
        print("Ending case# must be greater than starting case#. Aborting!")
        exit(0)
    elif args.test_type < 0 or args.test_type > 1:
        print("Test Type# must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.qa_mode < 0 or args.qa_mode > 1:
        print("QA mode must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.case_list is not None and args.case_start > caseMin and args.case_end < caseMax:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
        exit(0)
    elif args.num_runs <= 0:
        print("Number of Runs must be greater than 0. Aborting!")
        exit(0)
    elif args.batch_size <= 0:
        print("Batch size must be greater than 0. Aborting!")
        exit(0)

    case_list = []
    if args.case_list:
        for case in args.case_list:
            try:
                case_number = get_case_number(miscAugmentationMap, case)
                case_list.append(case_number)
            except ValueError as e:
                print(e)

    args.case_list = case_list
    if args.case_list is None or len(args.case_list) == 0:
        args.case_list = range(args.case_start, args.case_end + 1)
        args.case_list = [str(x) for x in args.case_list]
    else:
        for case in args.case_list:
            if int(case) < caseMin or int(case) > caseMax:
                print("The case# must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
                exit(0)
    return args

args = rpp_test_suite_parser_and_validator()
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
toggle = args.toggle
caseList = args.case_list
numDims = args.num_dims
numRuns = args.num_runs
batchSize = args.batch_size
qaMode = args.qa_mode
if qaMode:
    testType = 0
preserveOutput = args.preserve_output
outFilePath = " "

if testType == 0 and batchSize != 3:
    print("QA mode can only run with a batch size of 3.")
    exit(0)
if preserveOutput == 0:
    validate_and_remove_folders(outFolderPath, "QA_RESULTS_MISC_HOST")
    validate_and_remove_folders(outFolderPath, "OUTPUT_PERFORMANCE_MISC_LOGS_HOST")

if(testType == 0):
    outFilePath = outFolderPath + '/QA_RESULTS_MISC_HOST_' + timestamp
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 100   #default numRuns for running performance tests
    outFilePath = outFolderPath + '/OUTPUT_PERFORMANCE_MISC_LOGS_HOST_' + timestamp
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = QA tests / 1 = Performance tests)")
    exit(0)

os.mkdir(outFilePath)
loggingFolder = outFilePath
dstPath = outFilePath

# Enable extglob
if os.path.exists(buildFolderPath + "/build"):
    shutil.rmtree(buildFolderPath + "/build")
os.makedirs(buildFolderPath + "/build")
os.chdir(buildFolderPath + "/build")

# Run cmake and make commands
subprocess.call(["cmake", scriptPath], cwd=".")   # nosec
subprocess.call(["make", "-j16"], cwd=".")    # nosec

noCaseSupported = all(int(case) not in miscAugmentationMap for case in caseList)
if noCaseSupported:
    print("\ncase numbers %s are not supported" % caseList)
    exit(0)
for case in caseList:
    if int(case) not in miscAugmentationMap:
        continue
    if case == "0":
        for transposeOrder in range(1, numDims):
            run_test(loggingFolder, numDims, case, numRuns, testType, toggle, batchSize, outFilePath, transposeOrder)
    elif case == "1":
        for axisMask in range(1, pow(2, numDims)):
            run_test(loggingFolder, numDims, case, numRuns, testType, toggle, batchSize, outFilePath, axisMask)
    elif case == "3":
        for axisMask in range(0, numDims):
            run_test(loggingFolder, numDims, case, numRuns, testType, toggle, batchSize, outFilePath, axisMask)
    else:
        run_test(loggingFolder, numDims, case, numRuns, testType, toggle, batchSize, outFilePath)

# print the results of qa tests
nonQACaseList = []
supportedCaseList = [key for key, values in miscAugmentationMap.items() if "HOST" in values]
supportedCases = 0
for num in caseList:
    if int(num) in miscAugmentationMap:
        supportedCases += 1
caseInfo = "Tests are run for " + str(supportedCases) + " supported cases out of the " + str(len(caseList)) + " cases requested"
if testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        print("---------------------------------- Results of QA Test - Tensor_misc_host ----------------------------------\n")
        print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList, "Tensor_misc_host")

# Performance tests
if (testType == 1):
    logFileList = get_log_file_list()
    functionalityGroupList = ["statiscal_operations"]

    for logFile in logFileList:
        print_performance_tests_summary(logFile, functionalityGroupList, numRuns)

if len(errorLog) > 1 or errorLog[0]["notExecutedFunctionality"] != 0:
    print("\n---------------------------------- Log of function variants requested but not run - Tensor_misc_host ----------------------------------\n")
    for i in range(1,len(errorLog)):
        print(errorLog[i])
    if(errorLog[0]["notExecutedFunctionality"] != 0):
        print(str(errorLog[0]["notExecutedFunctionality"]) + " functionality variants requested by test_suite_misc_host were not executed since these sub-variants are not currently supported in RPP.\n")
    print("-----------------------------------------------------------------------------------------------")