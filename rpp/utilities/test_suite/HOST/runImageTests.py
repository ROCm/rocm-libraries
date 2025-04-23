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
inFilePath1 = scriptPath + "/../TEST_IMAGES/three_images_mixed_src1"
inFilePath2 = scriptPath + "/../TEST_IMAGES/three_images_mixed_src2"
ricapInFilePath = scriptPath + "/../TEST_IMAGES/three_images_150x150_src1"
lensCorrectionInFilePath = scriptPath + "/../TEST_IMAGES/lens_distortion"
qaInputFile = scriptPath + "/../TEST_IMAGES/three_images_mixed_src1"
perfQaInputFile = scriptPath + "/../TEST_IMAGES/eight_images_mixed_src1"
outFolderPath = os.getcwd()
buildFolderPath = os.getcwd()
caseMin = 0
caseMax = 92
errorLog = [{"notExecutedFunctionality" : 0}]

# Get a list of log files based on a flag for preserving output
def get_log_file_list(preserveOutput):
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/Tensor_image_host_pkd3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/Tensor_image_host_pln3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/Tensor_image_host_pln1_raw_performance_log.txt"
    ]

def run_unit_test(srcPath1, srcPath2, dstPathTemp, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList):
    bitDepths = range(7)
    outputFormatToggles = [0, 1]
    if qaMode:
        bitDepths = [0, 2]
    for bitDepth in bitDepths:
        for outputFormatToggle in outputFormatToggles:
            # There is no layout toggle for PLN1 case, so skip this case
            if layout == 2 and outputFormatToggle == 1:
                continue

            if case == "49" or case == "54":
                for kernelSize in range(3, 10, 2):
                    print("./Tensor_image_host " + srcPath1 + " " + srcPath2 + " " + dstPathTemp + " " + str(bitDepth) + " " + str(outputFormatToggle) + " " + str(case) + " " + str(kernelSize) + " 0")
                    result = subprocess.Popen([buildFolderPath + "/build/Tensor_image_host", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), str(kernelSize), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
                    log_detected(result, errorLog, imageAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_image_layout_type(layout, outputFormatToggle, "HOST"))
            elif case == "8":
                # Run all variants of noise type functions with additional argument of noiseType = gausssianNoise / shotNoise / saltandpepperNoise
                for noiseType in range(3):
                    print("./Tensor_image_host " + srcPath1 + " " + srcPath2 + " " + dstPathTemp + " " + str(bitDepth) + " " + str(outputFormatToggle) + " " + str(case) + " " + str(noiseType) + " 0")
                    result = subprocess.Popen([buildFolderPath + "/build/Tensor_image_host", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), str(noiseType), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
                    log_detected(result, errorLog, imageAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_image_layout_type(layout, outputFormatToggle, "HOST"))
            elif case == "21" or case == "23" or case == "24" or case == "79" or case == "28":
                # Run all variants of interpolation functions with additional argument of interpolationType = bicubic / bilinear / gaussian / nearestneigbor / lanczos / triangular
                interpolationRange = 6
                if case =='79' or case == "28":
                    interpolationRange = 2
                for interpolationType in range(interpolationRange):
                    print("./Tensor_image_host " + srcPath1 + " " + srcPath2 + " " + dstPathTemp + " " + str(bitDepth) + " " + str(outputFormatToggle) + " " + str(case) + " " + str(interpolationType) + " 0")
                    result = subprocess.Popen([buildFolderPath + "/build/Tensor_image_host", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), str(interpolationType), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
                    log_detected(result, errorLog, imageAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_image_layout_type(layout, outputFormatToggle, "HOST"))
            else:
                print("./Tensor_image_host " + srcPath1 + " " + srcPath2 + " " + dstPathTemp + " " + str(bitDepth) + " " + str(outputFormatToggle) + " " + str(case) + " 0 " + str(numRuns) + " " + str(testType) + " " + str(layout) + " 0")
                result = subprocess.Popen([buildFolderPath + "/build/Tensor_image_host", srcPath1, srcPath2, dstPathTemp, str(bitDepth), str(outputFormatToggle), str(case), "0", str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
                log_detected(result, errorLog, imageAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_image_layout_type(layout, outputFormatToggle, "HOST"))

            print("------------------------------------------------------------------------------------------")

def run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, additionalParam, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList):
    if qaMode == 1:
        with open(loggingFolder + "/BatchPD_host_" + logFileLayout + "_raw_performance_log.txt", "a") as logFile:
            process = subprocess.Popen([buildFolderPath + "/build/BatchPD_host_" + logFileLayout, srcPath1, srcPath2, str(bitDepth), str(outputFormatToggle), str(case), str(additionalParam), "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
            read_from_subprocess_and_write_to_log(process, logFile)
            log_detected(process, errorLog, imageAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_image_layout_type(layout, outputFormatToggle, "HOST"))
    with open(loggingFolder + "/Tensor_image_host_" + logFileLayout + "_raw_performance_log.txt", "a") as logFile:
        logFile.write("./Tensor_image_host " + srcPath1 + " " + srcPath2 + " " + dstPath + " " + str(bitDepth) + " " + str(outputFormatToggle) + " " + str(case) + " " + str(additionalParam) + " 0\n")
        process = subprocess.Popen([buildFolderPath + "/build/Tensor_image_host", srcPath1, srcPath2, dstPath, str(bitDepth), str(outputFormatToggle), str(case), str(additionalParam), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
        read_from_subprocess_and_write_to_log(process, logFile)
        log_detected(process, errorLog, imageAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth)), get_image_layout_type(layout, outputFormatToggle, "HOST"))

def run_performance_test(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList):
    print("\n")
    bitDepths = range(7)
    if qaMode:
        bitDepths = [0]
    for bitDepth in bitDepths:
        for outputFormatToggle in range(2):
            # There is no layout toggle for PLN1 case, so skip this case
            if layout == 2 and outputFormatToggle == 1:
                continue
            if case == "49" or case == "54":
                for kernelSize in range(3, 10, 2):
                    run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, kernelSize, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
            elif case == "8":
                # Run all variants of noise type functions with additional argument of noiseType = gausssianNoise / shotNoise / saltandpepperNoise
                for noiseType in range(3):
                    run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, noiseType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
                    print("")
            elif case == "21" or case == "23" or case == "24" or case == "28" or case == "79":
                # Run all variants of interpolation functions with additional argument of interpolationType = bicubic / bilinear / gaussian / nearestneigbor / lanczos / triangular
                for interpolationType in range(6):
                    run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, interpolationType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
                    print("")
            else:
                run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, "0", numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
            print("------------------------------------------------------------------------------------------\n")

# Parse and validate command-line arguments for the RPP test suite
def rpp_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path1", type = str, default = inFilePath1, help = "Path to the input folder 1")
    parser.add_argument("--input_path2", type = str, default = inFilePath2, help = "Path to the input folder 2")
    parser.add_argument("--case_start", type = int, default = caseMin, help = "Testing start case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument("--case_end", type = int, default = caseMax, help = "Testing end case # - Range must be in [" + str(caseMin) + ":" + str(caseMax) + "]")
    parser.add_argument('--test_type', type = int, default = 0, help = "Type of Test - (0 = Unit tests / 1 = Performance tests)")
    parser.add_argument('--case_list', nargs = "+", help = "A list of specific case numbers to run separated by spaces", required = False)
    parser.add_argument('--qa_mode', type = int, default = 0, help = "Run with qa_mode? Output images from tests will be compared with golden outputs - (0 / 1)", required = False)
    parser.add_argument('--decoder_type', type = int, default = 0, help = "Type of Decoder to decode the input data - (0 = TurboJPEG / 1 = OpenCV)")
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )" )
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    parser.add_argument('--roi', nargs = 4, help = "specifies the roi values", required = False)
    print_case_list(imageAugmentationMap, "HOST", parser)
    args = parser.parse_args()

    # check if the folder exists
    validate_path(args.input_path1)
    validate_path(args.input_path2)
    validate_path(qaInputFile)
    validate_path(perfQaInputFile)

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
    elif args.qa_mode < 0 or args.qa_mode > 1:
        print("QA mode must be in the 0 / 1. Aborting!")
        exit(0)
    elif args.decoder_type < 0 or args.decoder_type > 1:
        print("Decoder Type must be in the 0/1 (0 = OpenCV / 1 = TurboJPEG). Aborting")
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
    elif args.roi is not None and any(int(val) < 0 for val in args.roi[:2]):
        print(" Invalid ROI. Aborting")
        exit(0)
    elif args.roi is not None and any(int(val) <= 0 for val in args.roi[2:]):
        print(" Invalid ROI. Aborting")
        exit(0)

    case_list = []
    if args.case_list:
        for case in args.case_list:
            try:
                case_number = get_case_number(imageAugmentationMap, case)
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
srcPath1 = args.input_path1
srcPath2 = args.input_path2
caseStart = args.case_start
caseEnd = args.case_end
testType = args.test_type
caseList = args.case_list
qaMode = args.qa_mode
decoderType = args.decoder_type
numRuns = args.num_runs
preserveOutput = args.preserve_output
batchSize = args.batch_size
roiList = ['0', '0', '0', '0'] if args.roi is None else args.roi

if qaMode and testType == 0 and batchSize != 3:
    print("QA mode can only run with a batch size of 3.")
    exit(0)

if qaMode and testType == 1 and batchSize != 8:
    print("Performance QA mode can only run with a batch size of 8.")
    exit(0)

# set the output folders and number of runs based on type of test (unit test / performance test)
if(testType == 0):
    if qaMode:
        outFilePath = outFolderPath + "/QA_RESULTS_HOST_" + timestamp
    else:
        outFilePath = outFolderPath + "/OUTPUT_IMAGES_HOST_" + timestamp
    numRuns = 1
elif(testType == 1):
    if "--num_runs" not in sys.argv:
        numRuns = 100 #default numRuns for running performance tests
    outFilePath = outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp
else:
    print("Invalid TEST_TYPE specified. TEST_TYPE should be 0/1 (0 = Unittests / 1 = Performancetests)")
    exit()

if preserveOutput == 0:
    validate_and_remove_folders(outFolderPath, "OUTPUT_IMAGES_HOST")
    validate_and_remove_folders(outFolderPath, "QA_RESULTS_HOST")
    validate_and_remove_folders(outFolderPath, "OUTPUT_PERFORMANCE_LOGS_HOST")

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

supportedCaseList = [key for key, values in imageAugmentationMap.items() if "HOST" in values]
if testType == 0:
    noCaseSupported = all(int(case) not in supportedCaseList for case in caseList)
    if noCaseSupported:
        print("\ncase numbers %s are not supported" % caseList)
        exit(0)
    for case in caseList:
        if int(case) not in imageAugmentationMap:
            continue
        if case == "82" and (("--input_path1" not in sys.argv and "--input_path2" not in sys.argv) or qaMode == 1):
            srcPath1 = ricapInFilePath
            srcPath2 = ricapInFilePath
        if case == "26" and (("--input_path1" not in sys.argv and "--input_path2" not in sys.argv) or qaMode == 1):
            srcPath1 = lensCorrectionInFilePath
            srcPath2 = lensCorrectionInFilePath
        # if QA mode is enabled overwrite the input folders with the folders used for generating golden outputs
        if qaMode == 1 and (case != "82" and case != "26"):
            srcPath1 = inFilePath1
            srcPath2 = inFilePath2
        for layout in range(3):
            dstPathTemp, logFileLayout = process_layout(layout, qaMode, case, dstPath, "host", func_group_finder)

            if qaMode == 0:
                if not os.path.isdir(dstPathTemp):
                    os.mkdir(dstPathTemp)

            run_unit_test(srcPath1, srcPath2, dstPathTemp, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)
    layoutDict = {0:"PKD3", 1:"PLN3", 2:"PLN1"}
    if qaMode == 0:
        create_layout_directories(dstPath, layoutDict)
else:
    noCaseSupported = all(int(case) not in imageAugmentationMap for case in caseList)
    if noCaseSupported:
        print("case numbers %s are not supported" % caseList)
        exit(0)
    for case in caseList:
        if int(case) not in imageAugmentationMap:
            continue
        # if QA mode is enabled overwrite the input folders with the folders used for generating golden outputs
        if qaMode == 1 and case != "82":
            srcPath1 = inFilePath1
            srcPath2 = inFilePath2
        if case == "82" and "--input_path1" not in sys.argv and "--input_path2" not in sys.argv:
            srcPath1 = ricapInFilePath
            srcPath2 = ricapInFilePath
        if case == "26" and "--input_path1" not in sys.argv and "--input_path2" not in sys.argv:
            srcPath1 = lensCorrectionInFilePath
            srcPath2 = lensCorrectionInFilePath
        for layout in range(3):
            dstPathTemp, logFileLayout = process_layout(layout, qaMode, case, dstPath, "host", func_group_finder)
            run_performance_test(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList)

# print the results of qa tests
nonQACaseList = ['6', '8', '10', '11', '24', '28', '54', '84'] # Add cases present in supportedCaseList, but without QA support

if qaMode and testType == 0:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        print("---------------------------------- Results of QA Test - Tensor_image_host ----------------------------------\n")
        print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList, "Tensor_image_host")

layoutDict = {0:"PKD3", 1:"PLN3", 2:"PLN1"}
# unit tests and QA mode disabled
if testType == 0 and qaMode == 0:
    create_layout_directories(dstPath, layoutDict)
# Performance tests
elif (testType == 1 and qaMode == 1):
    columns = ['BatchPD_Augmentation_Type', 'Tensor_Augmentation_Type', 'Performance Speedup (%)', 'Test_Result']
    tensorAugVariations = []
    batchPDAugVariations = []
    achievedPerf = []
    status = []
    df = pd.DataFrame(columns=columns)
    tensorLogFileList = get_log_file_list(preserveOutput)
    batchpdLogFileList = [sub.replace("Tensor_image_host", "BatchPD_host") for sub in tensorLogFileList] # will be needed only in qa mode

    stats = []
    tensorVal = []
    batchpdVal = []
    functions = []
    functionsBatchPD = []
    funcCount = 0
    performanceNoise = 10
    perfQASupportCaseList = ["resize", "color_twist", "phase"]
    for i in range(3):
        tensorLogFile = tensorLogFileList[i]
        batchpdLogFile = batchpdLogFileList[i]
        # Opening log file
        try:
            tensorFile = open(tensorLogFile,"r")
        except IOError:
            print("Skipping file -> "+ tensorLogFile)
            continue

        # Opening log file
        try:
            batchpdFile = open(batchpdLogFile,"r")
        except IOError:
            print("Skipping file -> "+ batchpdLogFile)
            continue

        prevLine = ""
        # Loop over each line
        for line in tensorFile:
            if "max,min,avg wall times in ms/batch" in line and "u8_Tensor" in prevLine:
                layoutCheck = "PKD3_toPKD3" in prevLine or "PLN3_toPLN3" in prevLine or "PLN1_toPLN1" in prevLine
                interpolationCheck = "interpolationType" not in prevLine or "interpolationTypeBilinear" in prevLine
                if layoutCheck and interpolationCheck:
                    splitWordStart = "Running "
                    splitWordEnd = " " + str(numRuns)
                    prevLine = prevLine.partition(splitWordStart)[2].partition(splitWordEnd)[0]
                    splitWordStart = "max,min,avg wall times in ms/batch = "
                    splitWordEnd = "\n"
                    if prevLine not in functions:
                        functions.append(prevLine)
                        stats = line.partition(splitWordStart)[2].partition(splitWordEnd)[0].split(",")
                        tensorVal.append(float(stats[2]))
                        funcCount += 1

            if line != "\n":
                prevLine = line

        # Closing log file
        tensorFile.close()

        stats = []
        prevLine = ""
        for line in batchpdFile:
            if "max,min,avg" in line and "u8_BatchPD" in prevLine:
                if "PKD3_toPKD3" in prevLine or "PLN3_toPLN3" in prevLine or "PLN1_toPLN1" in prevLine:
                    splitWordStart = "Running "
                    splitWordEnd = " " + str(numRuns)
                    prevLine = prevLine.partition(splitWordStart)[2].partition(splitWordEnd)[0]
                    splitWordStart = "max,min,avg"
                    splitWordEnd = "\n"
                    if prevLine not in functionsBatchPD:
                        functionsBatchPD.append(prevLine)
                        stats = line.partition(splitWordStart)[2].partition(splitWordEnd)[0].split(",")
                        batchpdVal.append(float(stats[2]) * float(1000.0))

            if line != "\n":
                prevLine = line

        # Closing log file
        batchpdFile.close()

    print("---------------------------------- Results of QA Test - Tensor_image_host ----------------------------------\n")
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    excelFilePath = os.path.join(outFilePath, "performance_qa_results.xlsx")
    f = open(qaFilePath, 'w')
    numLines = 0
    numPassed = 0
    removalList = ["_HOST", "_toPKD3", "_toPLN3", "_toPLN1"]
    for i in range(len(functions)):
        perfImprovement = int(((batchpdVal[i] - tensorVal[i]) / batchpdVal[i]) * 100)
        numLines += 1
        funcName = functions[i]
        caseName = funcName.split("_u8_")[0]
        for string in removalList:
            funcName = funcName.replace(string, "")
        if caseName not in perfQASupportCaseList:
            print("Error! QA mode is not yet available for variant: " + funcName)
            continue
        achievedPerf.append(perfImprovement)
        tensorAugVariations.append(funcName)
        if perfImprovement > -performanceNoise:
            numPassed += 1
            status.append("PASSED")
            print(funcName + ": PASSED")
        else:
            status.append("FAILED")
            print(funcName + ": FAILED")

    resultsInfo = "\n\nFinal Results of Tests:"
    resultsInfo += "\n    - Total test cases including all subvariants REQUESTED = " + str(numLines)
    resultsInfo += "\n    - Total test cases including all subvariants PASSED = " + str(numPassed)
    f.write(resultsInfo)
    batchPDAugVariations = [s.replace('Tensor', 'BatchPD') for s in tensorAugVariations]
    df['Tensor_Augmentation_Type'] = tensorAugVariations
    df['BatchPD_Augmentation_Type'] = batchPDAugVariations
    df['Performance Speedup (%)'] = achievedPerf
    df['Test_Result'] = status
    # Calculate the number of cases passed and failed
    passedCases = df['Test_Result'].eq('PASSED').sum()
    failedCases = df['Test_Result'].eq('FAILED').sum()

    summaryRow = {'BatchPD_Augmentation_Type': None,
                   'Tensor_Augmentation_Type': None,
                   'Performance Speedup (%)': None,
                   'Test_Result': 'Final Results of Tests: Passed: ' + str(passedCases) + ', Failed: ' + str(failedCases)}

    print("\n" + dataframe_to_markdown(df))

    # Append the summary row to the DataFrame
    # Convert the dictionary to a DataFrame
    summaryRow = pd.DataFrame([summaryRow])
    df = pd.concat([df, summaryRow], ignore_index=True, sort = True)

    df.to_excel(excelFilePath, index=False)
    print("\n-------------------------------------------------------------------" + resultsInfo + "\n\n-------------------------------------------------------------------")
    print("\nIMPORTANT NOTE:")
    print("- The following performance comparison shows Performance Speedup percentages between times measured on previous generation RPP-BatchPD APIs against current generation RPP-Tensor APIs.")
    print("- All APIs have been improved for performance ranging from " + str(0) + "% (almost same) to " + str(100) + "% faster.")
    print("- Random observations of negative speedups might always occur due to current test machine temperature/load variances or other CPU/GPU state-dependent conditions.")
    print("\n-------------------------------------------------------------------\n")
elif (testType == 1 and qaMode == 0):
    logFileList = get_log_file_list(preserveOutput)

    functionalityGroupList = [
        "color_augmentations",
        "data_exchange_operations",
        "effects_augmentations",
        "geometric_augmentations",
        "arithmetic_operations",
        "statistical_operations",
    ]

    for logFile in logFileList:
        print_performance_tests_summary(logFile, functionalityGroupList, numRuns)

if len(errorLog) > 1 or errorLog[0]["notExecutedFunctionality"] != 0:
    print("\n---------------------------------- Log of function variants requested but not run - Tensor_image_host ----------------------------------\n")
    for i in range(1,len(errorLog)):
        print(errorLog[i])
    if(errorLog[0]["notExecutedFunctionality"] != 0):
        print(str(errorLog[0]["notExecutedFunctionality"]) + " functionality variants requested by test_suite_image_host were not executed since these sub-variants are not currently supported in RPP.\n")
    print("-----------------------------------------------------------------------------------------------")