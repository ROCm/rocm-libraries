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
inFilePath = scriptPath + "/../TEST_AUDIO_FILES/three_samples_single_channel_src1"
outFolderPath = os.getcwd()
buildFolderPath = os.getcwd()
caseMin = 0
caseMax = 7
errorLog = [{"notExecutedFunctionality" : 0}]

# Get a list of log files based on a flag for preserving output
def get_log_file_list():
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_AUDIO_LOGS_HIP_" + timestamp + "/Tensor_audio_hip_raw_performance_log.txt",
    ]

def case_file_check(CASE_FILE_PATH, new_file):
    try:
        case_file = open(CASE_FILE_PATH,'r')
        for line in case_file:
            print(line)
            if not(line.startswith('"Name"')):
                new_file.write(line)
        case_file.close()
        return True
    except IOError:
        print("Unable to open case results")
        return False

# Generate performance reports based on counters and a list of types
def generate_performance_reports(RESULTS_DIR):
    import pandas as pd
    pd.options.display.max_rows = None
    # Generate performance report
    df = pd.read_csv(RESULTS_DIR + "/consolidated_results.stats.csv")
    df["AverageMs"] = df["AverageNs"] / 1000000
    dfPrint = df.drop(['Percentage'], axis = 1)
    dfPrint["HIP Kernel Name"] = dfPrint.iloc[:,0].str.lstrip("Hip_")
    dfPrint_noIndices = dfPrint.astype(str)
    dfPrint_noIndices.replace(['0', '0.0'], '', inplace = True)
    dfPrint_noIndices = dfPrint_noIndices.to_string(index = False)
    print(dfPrint_noIndices)

def run_unit_test_cmd(srcPath, case, numRuns, testType, batchSize, outFilePath):
    print("\n./Tensor_audio_hip " + srcPath + " " + str(case) + " " + str(numRuns) + " " + str(testType) + " " + str(numRuns) + " " + str(batchSize))
    result = subprocess.Popen([buildFolderPath + "/build/Tensor_audio_hip", srcPath, str(case), str(testType), str(numRuns), str(batchSize), outFilePath, scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
    log_detected(result, errorLog, audioAugmentationMap[int(case)][0], get_bit_depth(int(2)), "HIP")
    print("------------------------------------------------------------------------------------------")

def run_performance_test_cmd(loggingFolder, srcPath, case, numRuns, testType, batchSize, outFilePath):
    with open(loggingFolder + "/Tensor_audio_hip_raw_performance_log.txt", "a") as logFile:
        print("./Tensor_audio_hip " + srcPath + " " + str(case) + " " + str(numRuns) + " " + str(testType) + " " + str(numRuns) + " " + str(batchSize))
        process = subprocess.Popen([buildFolderPath + "/build/Tensor_audio_hip", srcPath, str(case), str(testType), str(numRuns), str(batchSize), outFilePath, scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
        read_from_subprocess_and_write_to_log(process, logFile)
        log_detected(process, errorLog, audioAugmentationMap[int(case)][0], get_bit_depth(int(2)), "HIP")
        print("------------------------------------------------------------------------------------------")

def run_performance_test_with_profiler_cmd(loggingFolder, srcPath, case, numRuns, testType, batchSize, outFilePath):
    if not os.path.isdir(outFilePath + "/case_" + case):
        os.mkdir(outFilePath + "/case_" + case)
    with open(loggingFolder + "/Tensor_audio_hip_raw_performance_log.txt", "a") as logFile:
        print("\nrocprof --basenames on --timestamp on --stats -o " + outFilePath + "/case_" + str(case) + "/output_case" + str(case) + ".csv ./Tensor_audio_hip " + srcPath + " " + str(case) + " " + str(numRuns) + " " + str(testType) + " " + str(numRuns) + " " + str(batchSize))
        process = subprocess.Popen([ 'rocprof', '--basenames', 'on', '--timestamp', 'on', '--stats', '-o', outFilePath + "/case_" + case + "/output_case" + case + ".csv", "./Tensor_audio_hip", srcPath, str(case), str(testType), str(numRuns), str(batchSize), outFilePath, scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # nosec
        while True:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            print(output.strip())
            output_str = output.decode('utf-8')
            logFile.write(output_str)

        log_detected(process, errorLog, audioAugmentationMap[int(case)][0], get_bit_depth(int(2)), "HIP")
        print("------------------------------------------------------------------------------------------")

def run_test(loggingFolder, srcPath, case, numRuns, testType, batchSize, outFilePath, profilingOption = "NO"):
    if testType == 0:
        run_unit_test_cmd(srcPath, case, numRuns, testType, batchSize, outFilePath)
    elif testType == 1 and profilingOption == "NO":
        print("\n")
        run_performance_test_cmd(loggingFolder, srcPath, case, numRuns, testType, batchSize, outFilePath)
    else:
        run_performance_test_with_profiler_cmd(loggingFolder, srcPath, case, numRuns, testType, batchSize, outFilePath)

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type = str, default = inFilePath, help = "Path to the input folder")
    parser.add_argument("--case_start", type = int, default = caseMin, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument("--case_end", type = int, default = caseMax, help = "Testing end case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = QA tests / 1 = Performance tests)")
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Output audio data from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--case_list', nargs = "+", help = "A list of specific case numbers to run separated by spaces", required = False)
    parser.add_argument('--profiling', type = str , default = 'NO', help = 'Run with profiler? - (YES/NO)', required = False)
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )")
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    print_case_list(audioAugmentationMap, "HIP", parser)
    args = parser.parse_args()

    # check if the folder exists
    validate_path(args.input_path)

    # validate the parameters passed by user
    if ((args.case_start < caseMin or args.case_start > caseMax) or (args.case_end < caseMin or args.case_end > caseMax)):
        print("Starting case# and Ending case# must be in the " + str(caseMin) + ":" + str(caseMax) + " range. Aborting!")
        exit(0)
    elif args.case_end < args.case_start:
        print("Ending case# must be greater than starting case#. Aborting!")
        exit(0)
    elif args.test_type < 0 or args.test_type > 1:
        print("Test Type# must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.case_list is not None and args.case_start != caseMin and args.case_end != caseMax:
        print("Invalid input! Please provide only 1 option between case_list, case_start and case_end")
        exit(0)
    elif args.qa_mode < 0 or args.qa_mode > 1:
        print("QA mode must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.num_runs <= 0:
        print("Number of Runs must be greater than 0. Aborting!")
        exit(0)
    elif args.batch_size <= 0:
        print("Batch size must be greater than 0. Aborting!")
        exit(0)
    elif args.profiling != 'YES' and args.profiling != 'NO':
        print("Profiling option value must be either 'YES' or 'NO'.")
        exit(0)
    elif args.preserve_output < 0 or args.preserve_output > 1:
        print("Preserve Output must be in the 0/1 (0 = override / 1 = preserve). Aborting")
        exit(0)
    elif args.test_type == 0 and args.input_path != inFilePath:
        print("Invalid input path! QA mode can run only with path:", inFilePath)
        exit(0)

    case_list = []
    if args.case_list:
        for case in args.case_list:
            try:
                case_number = get_case_number(audioAugmentationMap, case)
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
                print("Invalid case number " + str(case) + "! Case number must be in the " + str(caseMin) + ":" + str(caseMax) + " range. Aborting!")
                exit(0)
    return args

args = rpp_test_suite_parser_and_validator()
srcPath = args.input_path
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
caseList = args.case_list
qaMode = args.qa_mode
profilingOption = args.profiling
numRuns = args.num_runs
preserveOutput = args.preserve_output
batchSize = args.batch_size
outFilePath = " "

# Override testType to 0 if testType is 1 and qaMode is 1
if testType == 1 and qaMode == 1:
    print("WARNING: QA Mode cannot be run with testType = 1 (performance tests). Resetting testType to 0")
    testType = 0

# set the output folders and number of runs based on type of test (unit test / performance test)
if(testType == 0):
    outFilePath = outFolderPath + "/QA_RESULTS_AUDIO_HIP_" + timestamp
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 100   #default numRuns for running performance tests
    outFilePath = outFolderPath + "/OUTPUT_PERFORMANCE_AUDIO_LOGS_HIP_" + timestamp
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = QA tests / 1 = Performance tests)")
    exit(0)

if preserveOutput == 0:
    validate_and_remove_folders(outFolderPath, "QA_RESULTS_AUDIO_HIP")
    validate_and_remove_folders(outFolderPath, "OUTPUT_PERFORMANCE_AUDIO_LOGS_HIP")

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
subprocess.call(["make", "-j16"], cwd=".")    # nosec

if qaMode and batchSize != 3:
    print("QA tests can only run with a batch size of 3.")
    exit(0)

noCaseSupported = all(int(case) not in audioAugmentationMap for case in caseList)
if noCaseSupported:
    print("\ncase numbers %s are not supported" % caseList)
    exit(0)

for case in caseList:
    if "--input_path" not in sys.argv:
        if case == "3":
            srcPath = scriptPath + "/../TEST_AUDIO_FILES/three_sample_multi_channel_src1"
        else:
            srcPath = inFilePath

    if int(case) not in audioAugmentationMap:
        continue
    run_test(loggingFolder, srcPath, case, numRuns, testType, batchSize, outFilePath, profilingOption)

# print the results of qa tests
nonQACaseList = [] # Add cases present in supportedCaseList, but without QA support
supportedCaseList = [key for key, values in imageAugmentationMap.items() if "HIP" in values]

if testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        print("---------------------------------- Results of QA Test - Tensor_audio_hip -----------------------------------\n")
        print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList, "Tensor_audio_hip")

# Performance tests
if testType == 1 and profilingOption == "NO":
    logFileList = get_log_file_list()
    for logFile in logFileList:
        print_performance_tests_summary(logFile, "", numRuns)
elif testType == 1 and profilingOption == "YES":
    RESULTS_DIR = outFolderPath + "/OUTPUT_PERFORMANCE_AUDIO_LOGS_HIP_" + timestamp
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE = RESULTS_DIR + "/consolidated_results.stats.csv"

    CASE_NUM_LIST = caseList
    BIT_DEPTH_LIST = [2]
    OFT_LIST = [0]

    # Open csv file
    new_file = open(CONSOLIDATED_FILE, 'w')
    new_file.write('"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

    # Loop through cases
    for CASE_NUM in CASE_NUM_LIST:
        # Set results directory
        CASE_RESULTS_DIR = RESULTS_DIR + "/case_" + str(CASE_NUM)
        print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)

        # Loop through bit depths
        for BIT_DEPTH in BIT_DEPTH_LIST:
            # Loop through output format toggle cases
            for OFT in OFT_LIST:
                # Write into csv file
                CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + ".stats.csv"
                print("CASE_FILE_PATH = " + CASE_FILE_PATH)
                fileCheck = case_file_check(CASE_FILE_PATH, new_file)
                if fileCheck == False:
                    continue

    new_file.close()
    subprocess.call(['chown', str(os.getuid()) + ':' + str(os.getgid()), CONSOLIDATED_FILE])  # nosec
    try:
        generate_performance_reports(RESULTS_DIR)
    except ImportError:
        print("\nPandas not available! Results of GPU profiling experiment are available in the following files:\n" + \
                CONSOLIDATED_FILE + "\n")
    except IOError:
        print("Unable to open results in " + CONSOLIDATED_FILE)

if len(errorLog) > 1 or errorLog[0]["notExecutedFunctionality"] != 0:
    print("\n---------------------------------- Log of function variants requested but not run - Tensor_audio_hip  ----------------------------------\n")
    for i in range(1,len(errorLog)):
        print(errorLog[i])
    if(errorLog[0]["notExecutedFunctionality"] != 0):
        print(str(errorLog[0]["notExecutedFunctionality"]) + " functionality variants requested by test_suite_audio_hip were not executed since these sub-variants are not currently supported in RPP.\n")
    print("-----------------------------------------------------------------------------------------------")
