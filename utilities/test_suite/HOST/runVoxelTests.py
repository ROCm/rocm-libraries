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
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.join(os.path.dirname( __file__ ), '..' ))
from common import *

# Set the timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

scriptPath = os.path.dirname(os.path.realpath(__file__))
headerFilePath = scriptPath + "/../TEST_QA_IMAGES_VOXEL"
dataFilePath = scriptPath + "/../TEST_QA_IMAGES_VOXEL"
qaInputFile = scriptPath + "/../TEST_QA_IMAGES_VOXEL"
outFolderPath = os.getcwd()
buildFolderPath = os.getcwd()
caseMin = min(voxelAugmentationMap.keys())
caseMax = max(voxelAugmentationMap.keys())
errorLog = [{"notExecutedFunctionality" : 0}]

# Get a list of log files based on a flag for preserving output
def get_log_file_list():
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_VOXEL_" + timestamp + "/Tensor_voxel_host_pkd3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_VOXEL_" + timestamp + "/Tensor_voxel_host_pln3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_VOXEL_" + timestamp + "/Tensor_voxel_host_pln1_raw_performance_log.txt"
    ]

def run_unit_test_cmd(headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize):
    print("\n./Tensor_voxel_host " + headerPath + " " + dataPath + " " + dstPathTemp + " " + str(layout) + " " + str(case) + " " + str(numRuns) + " " + str(testType) + " " + str(qaMode) + " " + str(batchSize) + " " + str(bitDepth.value))
    result = subprocess.Popen([buildFolderPath + "/build/Tensor_voxel_host", headerPath, dataPath, dstPathTemp, str(layout), str(case), str(numRuns), str(testType), str(qaMode), str(batchSize), str(bitDepth.value), scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # nosec
    log_detected(result, errorLog, voxelAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth.value)), get_voxel_layout_type(layout, "HOST"))
    print("\n------------------------------------------------------------------------------------------")

def run_performance_test_cmd(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize):
    with open(loggingFolder + "/Tensor_voxel_host_" + logFileLayout + "_raw_performance_log.txt", "a") as logFile:
        logFile.write("./Tensor_voxel_host " + headerPath + " " + dataPath + " " + dstPathTemp + " " + str(layout) + " " + str(case) + " " + str(numRuns) + " " + str(testType) + " " + str(qaMode) + " " + str(batchSize) + " " + str(bitDepth.value) + "\n")
        process = subprocess.Popen([buildFolderPath + "/build/Tensor_voxel_host", headerPath, dataPath, dstPathTemp, str(layout), str(case), str(numRuns), str(testType), str(qaMode), str(batchSize), str(bitDepth.value), scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # nosec
        while True:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            output = output.decode('utf-8')
            if output:
                print(output)
            if "Running" in output or "max,min,avg wall times" in output:
                cleanedOutput = ''.join(char for char in output if 32 <= ord(char) <= 126)  # Remove control characters
                cleanedOutput = cleanedOutput.strip()  # Remove leading/trailing whitespace
                logFile.write(cleanedOutput + '\n')
                if "max,min,avg wall times" in output:
                    logFile.write("\n")
            else:
                logFile.write(output)


        log_detected(process, errorLog, voxelAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth.value)), get_voxel_layout_type(layout, "HOST"))
        print("\n------------------------------------------------------------------------------------------")

def run_test(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize):
    if testType == TestType.UNIT_TEST.value:
        run_unit_test_cmd(headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize)
    elif testType == TestType.PERFORMANCE_TEST.value:
        run_performance_test_cmd(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize)

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header_path", type = str, default = headerFilePath, help = "Path to the nii header")
    parser.add_argument("--data_path", type = str, default = dataFilePath, help = "Path to the nii data file")
    parser.add_argument("--case_start", type = int, default = caseMin, help = "Testing range starting case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument("--case_end", type = int, default = caseMax, help = "Testing range ending case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = Unit tests / 1 = Performance tests)")
    parser.add_argument('--case_list', nargs = "+", help = "A list of specific case numbers to run separated by spaces", required = False)
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Output images from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )" )
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    print_case_list(voxelAugmentationMap, "HOST", parser)
    args = parser.parse_args()

    # check if the folder exists
    validate_path(args.header_path)
    validate_path(args.data_path)
    validate_path(qaInputFile)

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
    elif args.preserve_output < 0 or args.preserve_output > 1:
        print("Preserve Output must be in the 0/1 (0 = override / 1 = preserve). Aborting")
        exit(0)

    case_list = []
    if args.case_list:
        for case in args.case_list:
            try:
                case_number = get_case_number(voxelAugmentationMap, case)
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

    # if QA mode is enabled overwrite the input folders with the folders used for generating golden outputs
    if args.qa_mode:
        args.header_path = headerFilePath
        args.data_path = dataFilePath

    return args

args = rpp_test_suite_parser_and_validator()
headerPath = args.header_path
dataPath = args.data_path
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
caseList = args.case_list
qaMode = args.qa_mode
numRuns = args.num_runs
preserveOutput = args.preserve_output
batchSize = args.batch_size

if qaMode and os.path.abspath(qaInputFile) != os.path.abspath(headerPath):
    print("QA mode should only run with the given Input path: ", qaInputFile)
    exit(0)

if qaMode and batchSize != 3:
    print("QA mode can only run with a batch size of 3.")
    exit(0)

# set the output folders and number of runs based on type of test (unit test / performance test)
if(testType == TestType.UNIT_TEST.value):
    if qaMode:
        outFilePath = os.path.join(outFolderPath + "/QA_RESULTS_HOST_VOXEL_" + timestamp)
    else:
        outFilePath = os.path.join(outFolderPath + "/OUTPUT_VOXEL_HOST_" + timestamp)
    numRuns = 1
elif(testType == TestType.PERFORMANCE_TEST.value):
    if "--num_runs" not in sys.argv:
        numRuns = 100 #default numRuns for running performance tests
    outFilePath = os.path.join(outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_VOXEL_" + timestamp)
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)")
    exit()

if preserveOutput == 0:
    validate_and_remove_folders(outFolderPath, "OUTPUT_VOXEL_HOST")
    validate_and_remove_folders(outFolderPath, "QA_RESULTS_HOST_VOXEL")
    validate_and_remove_folders(outFolderPath, "OUTPUT_PERFORMANCE_LOGS_HOST_VOXEL")

os.mkdir(outFilePath)
loggingFolder = outFilePath
dstPath = outFilePath

# Validate DST_FOLDER
validate_and_remove_files(dstPath)

# Enable extglob
if os.path.exists(buildFolderPath + "/build"):
    shutil.rmtree(buildFolderPath + "/build")
os.makedirs(buildFolderPath + "/build")
os.chdir(buildFolderPath + "/build")

# Run cmake and make commands
run_cmake_build(scriptPath)

bitDepths = [BitDepthTestMode.U8_TO_U8, BitDepthTestMode.F32_TO_F32]
noCaseSupported = all(int(case) not in voxelAugmentationMap for case in caseList)
if noCaseSupported:
    print("\ncase numbers %s are not supported" % caseList)
    exit(0)
for case in caseList:
    if int(case) not in voxelAugmentationMap:
        continue
    for layout in list(Layout):
        dstPathTemp, logFileLayout = process_layout(layout, qaMode, case, dstPath, "host", voxelAugmentationGroupMap, func_group_finder, voxelAugmentationMap)
        if testType == TestType.UNIT_TEST.value and not qaMode:
            if not os.path.isdir(dstPathTemp):
                os.mkdir(dstPathTemp)

        bitDepths = [BitDepthTestMode.U8_TO_U8, BitDepthTestMode.F32_TO_F32]
        if testType == TestType.UNIT_TEST.value and qaMode:
            bitDepths = [BitDepthTestMode.F32_TO_F32]
        for bitDepth in bitDepths:
            run_test(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout.value, case, numRuns, testType, qaMode, batchSize)

# print the results of qa tests
nonQACaseList = ['6'] # Add cases present in supportedCaseList, but without QA support
supportedCaseList = [key for key, values in voxelAugmentationMap.items() if "HOST" in values]

if qaMode and testType == TestType.UNIT_TEST.value:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList, "Tensor_voxel_host")

if (testType == TestType.UNIT_TEST.value and not qaMode):   # Unit tests
    create_layout_directories(dstPath)
elif (testType == TestType.PERFORMANCE_TEST.value):   # Performance tests
    logFileList = get_log_file_list()
    functionalityGroupList = ["arithmetic_operations", "geometric_augmentations", "effects_augmentations"]

    for logFile in logFileList:
        print_performance_tests_summary(logFile, functionalityGroupList, numRuns)

if len(errorLog) > 1 or errorLog[0]["notExecutedFunctionality"] != 0:
    print("\n---------------------------------- Log of function variants requested but not run - Tensor_voxel_host ----------------------------------\n")
    for i in range(1,len(errorLog)):
        print(errorLog[i])
    if(errorLog[0]["notExecutedFunctionality"] != 0):
        print(str(errorLog[0]["notExecutedFunctionality"]) + " functionality variants requested by test_suite_voxel_host were not executed since these sub-variants are not currently supported in RPP.\n")
    print("-----------------------------------------------------------------------------------------------")