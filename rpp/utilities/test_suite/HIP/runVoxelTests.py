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
caseMin = 0
caseMax = 6
errorLog = [{"notExecutedFunctionality" : 0}]

def get_log_file_list(preserveOutput):
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL_" + timestamp + "/Tensor_voxel_hip_pkd3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL_" + timestamp + "/Tensor_voxel_hip_pln3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL_" + timestamp + "/Tensor_voxel_hip_pln1_raw_performance_log.txt"
    ]

# Functionality group finder
def func_group_finder(case_number):
    if case_number == 0:
        return "arithmetic_operations"
    elif case_number == 1:
        return "geometric_augmentations"
    else:
        return "miscellaneous"

def run_unit_test_cmd(headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize):
    print("\n./Tensor_voxel_hip " + headerPath + " " + dataPath + " " + dstPathTemp + " " + str(layout) + " " + str(case) + " " + str(numRuns) + " " + str(testType) + " " + str(qaMode) + " " + str(batchSize) + " " + str(bitDepth))
    result = subprocess.Popen([buildFolderPath + "/build/Tensor_voxel_hip", headerPath, dataPath, dstPathTemp, str(layout), str(case), str(numRuns), str(testType), str(qaMode), str(batchSize), str(bitDepth), scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # nosec
    log_detected(result, errorLog, voxelAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_voxel_layout_type(layout, "HIP"))
    print("\n------------------------------------------------------------------------------------------")

def run_performance_test_cmd(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize):
   with open(loggingFolder + "/Tensor_voxel_hip_" + logFileLayout + "_raw_performance_log.txt", "a") as logFile:
        logFile.write("./Tensor_voxel_hip " + headerPath + " " + dataPath + " " + dstPathTemp + " " + str(layout) + " " + str(case) + " " + str(numRuns) + " " + str(testType) + " " + str(qaMode) + " " + str(batchSize) + " " + str(bitDepth) + "\n")
        process = subprocess.Popen([buildFolderPath + "/build/Tensor_voxel_hip", headerPath, dataPath, dstPathTemp, str(layout), str(case), str(numRuns), str(testType), str(qaMode), str(batchSize), str(bitDepth), scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE) # nosec
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

        log_detected(process, errorLog, voxelAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_voxel_layout_type(layout, "HIP"))
        print("\n------------------------------------------------------------------------------------------")

def run_performance_test_with_profiler_cmd(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize):
    layoutName = get_layout_name(layout)
    directory_path = os.path.join(loggingFolder, "Tensor_" + layoutName, "case_" + str(case))
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    bitDepths = [0, 2]
    for bitDepth in bitDepths:
        with open(loggingFolder + "/Tensor_voxel_hip_" + logFileLayout + "_raw_performance_log.txt", "a") as logFile:
            logFile.write("\nrocprof --basenames on --timestamp on --stats -o " + dstPathTemp + "/Tensor_" + layoutName + "/case_" + str(case) + "/output_case" + str(case) + ".csv ./Tensor_voxel_hip " + headerPath + " " + dataPath + " " + dstPathTemp + " " + str(layout) + " " + str(case) + " " + str(numRuns) + " " + str(testType) + " " + str(qaMode) + " " + str(batchSize) + " " + str(bitDepth) + "\n")
            process = subprocess.Popen(['rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', dstPath + "/Tensor_" + layoutName + "/case_" + str(case) + "/output_case" + str(case) + ".csv", buildFolderPath + "/build/Tensor_voxel_hip", headerPath, dataPath, dstPathTemp, str(layout), str(case), str(numRuns), str(testType), str(qaMode), str(batchSize), str(bitDepth), scriptPath], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # nosec
            while True:
                output = process.stdout.readline()
                if not output and process.poll() is not None:
                    break
                print(output.strip())
                logFile.write(output.decode('utf-8'))
        log_detected(process, errorLog, voxelAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_voxel_layout_type(layout, "HIP"))
    print("------------------------------------------------------------------------------------------")

def run_test(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize, profilingOption = 'NO'):
    if testType == 0:
        run_unit_test_cmd(headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize)
    elif testType == 1 and profilingOption == "NO":
        run_performance_test_cmd(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize)
    elif testType == 1 and profilingOption == "YES":
        run_performance_test_with_profiler_cmd(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize)

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header_path", type = str, default = headerFilePath, help = "Path to the nii header")
    parser.add_argument("--data_path", type = str, default = dataFilePath, help = "Path to the nii data file")
    parser.add_argument("--case_start", type = int, default = caseMin, help = "Testing range starting case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument("--case_end", type = int, default = caseMax, help = "Testing range ending case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = Unit tests / 1 = Performance tests)")
    parser.add_argument('--case_list', nargs = "+", help = "A list of specific case numbers to run separated by spaces", required = False)
    parser.add_argument('--profiling', type = str , default = 'NO', help = 'Run with profiler? - (YES/NO)', required = False)
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Output images from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )" )
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    print_case_list(voxelAugmentationMap, "HIP", parser)
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
    elif args.profiling != 'YES' and args.profiling != 'NO':
        print("Profiling option value must be either 'YES' or 'NO'.")
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
profilingOption = args.profiling
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
if(testType == 0):
    if qaMode:
        outFilePath = outFolderPath + "/QA_RESULTS_HIP_VOXEL_" + timestamp
    else:
        outFilePath = outFolderPath + "/OUTPUT_VOXEL_HIP_" + timestamp
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 100 #default numRuns for running performance tests
    outFilePath = outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL_" + timestamp
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)")
    exit()

if preserveOutput == 0:
    validate_and_remove_folders(outFolderPath, "OUTPUT_VOXEL_HIP")
    validate_and_remove_folders(outFolderPath, "QA_RESULTS_HIP_VOXEL")
    validate_and_remove_folders(outFolderPath, "OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL")

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
subprocess.call(["cmake", scriptPath], cwd=".")   # nosec
subprocess.call(["make", "-j16"], cwd=".")  # nosec

# Create folders based on testType and profilingOption
if testType == 1 and profilingOption == "YES":
    os.makedirs(dstPath + "/Tensor_PKD3")
    os.makedirs(dstPath + "/Tensor_PLN1")
    os.makedirs(dstPath + "/Tensor_PLN3")

bitDepths = [0, 2]
if (testType == 0 or (testType == 1 and profilingOption == "NO")):
    noCaseSupported = all(int(case) not in voxelAugmentationMap for case in caseList)
    if noCaseSupported:
        print("\ncase numbers %s are not supported" % caseList)
        exit(0)
    for case in caseList:
        if int(case) not in voxelAugmentationMap:
            continue
        for layout in range(3):
            dstPathTemp, logFileLayout = process_layout(layout, qaMode, case, dstPath, "hip", func_group_finder)
            if testType == 0 and qaMode == 0:
                if not os.path.isdir(dstPathTemp):
                    os.mkdir(dstPathTemp)

            bitDepths = [0, 2]
            if testType == 0 and qaMode:
                bitDepths = [2]
            for bitDepth in bitDepths:
                run_test(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize)
elif (testType == 1 and profilingOption == "YES"):
    NEW_FUNC_GROUP_LIST = [0, 1]
    noCaseSupported = all(int(case) not in voxelAugmentationMap for case in caseList)
    if noCaseSupported:
        print("\ncase numbers %s are not supported" % caseList)
        exit(0)
    for case in caseList:
        if int(case) not in voxelAugmentationMap:
            continue
        for layout in range(3):
            dstPathTemp, logFileLayout = process_layout(layout, qaMode, case, dstPath, "hip", func_group_finder)
            run_test(loggingFolder, logFileLayout, headerPath, dataPath, dstPathTemp, layout, case, numRuns, testType, qaMode, batchSize, profilingOption)

        RESULTS_DIR = ""
        RESULTS_DIR = outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HIP_VOXEL_" + timestamp
        print("RESULTS_DIR = " + RESULTS_DIR)
        CONSOLIDATED_FILE_TENSOR_PKD3 = RESULTS_DIR + "/consolidated_results_Tensor_PKD3.stats.csv"
        CONSOLIDATED_FILE_TENSOR_PLN1 = RESULTS_DIR + "/consolidated_results_Tensor_PLN1.stats.csv"
        CONSOLIDATED_FILE_TENSOR_PLN3 = RESULTS_DIR + "/consolidated_results_Tensor_PLN3.stats.csv"

        TYPE_LIST = ["Tensor_PKD3", "Tensor_PLN1", "Tensor_PLN3"]
        TENSOR_TYPE_LIST = ["Tensor_PKD3", "Tensor_PLN1", "Tensor_PLN3"]
        CASE_NUM_LIST = caseList
        BIT_DEPTH_LIST = [0, 2]
        OFT_LIST = [0]
        d_counter = {"Tensor_PKD3":0, "Tensor_PLN1":0, "Tensor_PLN3":0}

        for TYPE in TYPE_LIST:
            # Open csv file
            new_file = open(RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv",'w')
            new_file.write('"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

            prev = ""

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
                        # Write into csv file
                        CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + ".stats.csv"
                        print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                        fileCheck = case_file_check(CASE_FILE_PATH, TYPE, TENSOR_TYPE_LIST, new_file, d_counter)
                        if fileCheck == False:
                            continue

            new_file.close()
            subprocess.call(['chown', str(os.getuid()) + ':' + str(os.getgid()), RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv"])  # nosec
        try:
            generate_performance_reports(d_counter, TYPE_LIST, RESULTS_DIR)

        except ImportError:
            print("\nPandas not available! Results of GPU profiling experiment are available in the following files:\n" + \
                    CONSOLIDATED_FILE_TENSOR_PKD3 + "\n" + \
                        CONSOLIDATED_FILE_TENSOR_PLN1 + "\n" + \
                            CONSOLIDATED_FILE_TENSOR_PLN3 + "\n")

        except IOError:
            print("Unable to open results in " + RESULTS_DIR + "/consolidated_results_" + TYPE + ".stats.csv")

# print the results of qa tests
nonQACaseList = ['6'] # Add cases present in supportedCaseList, but without QA support
supportedCaseList = [key for key, values in imageAugmentationMap.items() if "HIP" in values]

if qaMode and testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        print("---------------------------------- Results of QA Test - Tensor_voxel_hip ----------------------------------\n")
        print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList, "Tensor_voxel_hip")

layoutDict = {0:"PKD3", 1:"PLN3", 2:"PLN1"}
if (testType == 0 and qaMode == 0): # Unit tests
    create_layout_directories(dstPath, layoutDict)
elif (testType == 1 and profilingOption == "NO"): # Performance tests
    logFileList = get_log_file_list(preserveOutput)
    functionalityGroupList = ["arithmetic_operations", "geometric_augmentations", "effects_augmentations"]

    for logFile in logFileList:
        print_performance_tests_summary(logFile, functionalityGroupList, numRuns)

if len(errorLog) > 1 or errorLog[0]["notExecutedFunctionality"] != 0:
    print("\n---------------------------------- Log of function variants requested but not run - Tensor_voxel_hip  ----------------------------------\n")
    for i in range(1,len(errorLog)):
        print(errorLog[i])
    if(errorLog[0]["notExecutedFunctionality"] != 0):
        print(str(errorLog[0]["notExecutedFunctionality"]) + " functionality variants requested by test_suite_voxel_hip were not executed since these sub-variants are not currently supported in RPP.\n")
    print("-----------------------------------------------------------------------------------------------")