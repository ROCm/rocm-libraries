"""
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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
caseMin = min(imageAugmentationMap.keys())
caseMax = max(imageAugmentationMap.keys())
errorLog = [{"notExecutedFunctionality" : 0}]

# Get a list of log files based on a flag for preserving output
def get_log_file_list(preserveOutput, singleImageFlag=False):
    binName = "Tensor_single_image_host" if singleImageFlag else "Tensor_image_host"
    return [
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/" + binName + "_pkd3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/" + binName + "_pln3_raw_performance_log.txt",
        outFolderPath + "/OUTPUT_PERFORMANCE_LOGS_HOST_" + timestamp + "/" + binName + "_pln1_raw_performance_log.txt"
    ]

def run_unit_test(srcPath1, srcPath2, dstPathTemp, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList, singleImageFlag):
    if singleImageFlag and int(case) not in SINGLE_IMAGE_SUPPORTED_CASES:
        return
    binName = "Tensor_single_image_host" if singleImageFlag else "Tensor_image_host"
    augName = imageAugmentationMap[int(case)][0]
    bitDepths = list(BitDepthTestMode)
    outputFormatToggles = list(OutputFormat)
    if qaMode:
        bitDepths = [BitDepthTestMode.U8_TO_U8, BitDepthTestMode.F32_TO_F32]
    if singleImageFlag:
        bitDepths = [bd for bd in bitDepths if bd in SINGLE_IMAGE_SUPPORTED_BIT_DEPTHS]
    for bitDepth in bitDepths:
        for outputFormatToggle in outputFormatToggles:
            # There is no layout toggle for PLN1 case, so skip this case
            if layout == Layout.PLN1.value and outputFormatToggle == OutputFormat.TOGGLE:
                continue

            if augName in {"box_filter", "median_filter", "gaussian_filter", "erode", "dilate", "emboss"}:
                for kernelSize in range(3, 10, 2):
                    print("./" + binName + " " + srcPath1 + " " + srcPath2 + " " + dstPathTemp + " " + str(bitDepth.value) + " " + str(outputFormatToggle.value) + " " + str(case) + " " + str(kernelSize) + " 0")
                    result = subprocess.Popen([buildFolderPath + "/build/" + binName, srcPath1, srcPath2, dstPathTemp, str(bitDepth.value), str(outputFormatToggle.value), str(case), str(kernelSize), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
                    log_detected(result, errorLog, augName, get_bit_depth(int(bitDepth.value)), get_image_layout_type(layout, outputFormatToggle.value, "HOST"))
            elif augName == "noise":
                # Run all variants of noise type functions with additional argument of noiseType = gausssianNoise / shotNoise / saltandpepperNoise
                for noiseType in range(3):
                    print("./" + binName + " " + srcPath1 + " " + srcPath2 + " " + dstPathTemp + " " + str(bitDepth.value) + " " + str(outputFormatToggle.value) + " " + str(case) + " " + str(noiseType) + " 0")
                    result = subprocess.Popen([buildFolderPath + "/build/" + binName, srcPath1, srcPath2, dstPathTemp, str(bitDepth.value), str(outputFormatToggle.value), str(case), str(noiseType), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
                    log_detected(result, errorLog, augName, get_bit_depth(int(bitDepth.value)), get_image_layout_type(layout, outputFormatToggle.value, "HOST"))
            elif augName in {"resize", "rotate", "warp_affine", "remap", "warp_perspective"}:
                # Run all variants of interpolation functions with additional argument of interpolationType = bicubic / bilinear / gaussian / nearestneigbor / lanczos / triangular
                if singleImageFlag:
                    interpolationRange = 1  # nearestneighbor only (index 3 on HOST, but single image binary accepts index 0)
                elif augName in {"remap", "warp_perspective"}:
                    interpolationRange = 2
                else:
                    interpolationRange = 6
                for interpolationType in range(interpolationRange):
                    print("./" + binName + " " + srcPath1 + " " + srcPath2 + " " + dstPathTemp + " " + str(bitDepth.value) + " " + str(outputFormatToggle.value) + " " + str(case) + " " + str(interpolationType) + " 0")
                    result = subprocess.Popen([buildFolderPath + "/build/" + binName, srcPath1, srcPath2, dstPathTemp, str(bitDepth.value), str(outputFormatToggle.value), str(case), str(interpolationType), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
                    log_detected(result, errorLog, augName, get_bit_depth(int(bitDepth.value)), get_image_layout_type(layout, outputFormatToggle.value, "HOST"))
            elif augName == "channel_permute":
                # Run all 6 channel swap permutations (R-G-B, R-B-G, G-R-B, G-B-R, B-R-G, B-G-R) by varying swapOrder (0 to 5)
                for swapOrder in range(6):
                    print("./" + binName + " " + srcPath1 + " " + srcPath2 + " " + dstPathTemp + " " + str(bitDepth.value) + " " + str(outputFormatToggle.value) + " " + str(case) + " " + str(swapOrder))
                    result = subprocess.Popen([buildFolderPath + "/build/" + binName, srcPath1, srcPath2, dstPathTemp, str(bitDepth.value), str(outputFormatToggle.value), str(case), str(swapOrder), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
                    log_detected(result, errorLog, augName, get_bit_depth(int(bitDepth.value)), get_image_layout_type(layout, outputFormatToggle.value, "HOST"))
            elif augName == "sobel_filter":
                for kernelSizeAndGradient in range(9):
                    print("./" + binName + " " + srcPath1 + " " + srcPath2 + " " + dstPathTemp + " " + str(bitDepth.value) + " " + str(outputFormatToggle.value) + " " + str(case) + " " + str(kernelSizeAndGradient) + " 0")
                    result = subprocess.Popen([buildFolderPath + "/build/" + binName, srcPath1, srcPath2, dstPathTemp, str(bitDepth.value), str(outputFormatToggle.value), str(case), str(kernelSizeAndGradient), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
                    log_detected(result, errorLog, augName, get_bit_depth(int(bitDepth.value)), get_image_layout_type(layout, outputFormatToggle.value, "HOST"))
            else:
                print("./" + binName + " " + srcPath1 + " " + srcPath2 + " " + dstPathTemp + " " + str(bitDepth.value) + " " + str(outputFormatToggle.value) + " " + str(case) + " 0 " + str(numRuns) + " " + str(testType) + " " + str(layout) + " 0")
                result = subprocess.Popen([buildFolderPath + "/build/" + binName, srcPath1, srcPath2, dstPathTemp, str(bitDepth.value), str(outputFormatToggle.value), str(case), "0", str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
                log_detected(result, errorLog, augName, get_bit_depth(int(bitDepth.value)), get_image_layout_type(layout, outputFormatToggle.value, "HOST"))

            print("------------------------------------------------------------------------------------------")

def run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, additionalParam, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList, singleImageFlag=False):
    binName = "Tensor_single_image_host" if singleImageFlag else "Tensor_image_host"
    with open(loggingFolder + "/" + binName + "_" + logFileLayout + "_raw_performance_log.txt", "a") as logFile:
        logFile.write("./" + binName + " " + srcPath1 + " " + srcPath2 + " " + dstPath + " " + str(bitDepth.value) + " " + str(outputFormatToggle.value) + " " + str(case) + " " + str(additionalParam) + " 0\n")
        process = subprocess.Popen([buildFolderPath + "/build/" + binName, srcPath1, srcPath2, dstPath, str(bitDepth.value), str(outputFormatToggle.value), str(case), str(additionalParam), str(numRuns), str(testType), str(layout), "0", str(qaMode), str(decoderType), str(batchSize)] + roiList + [scriptPath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    # nosec
        read_from_subprocess_and_write_to_log(process, logFile)
        log_detected(process, errorLog, imageAugmentationMap[int(case)][0], get_bit_depth(int(bitDepth.value)), get_image_layout_type(layout, outputFormatToggle.value, "HOST"))

def run_performance_test(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, case, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList, singleImageFlag):
    if singleImageFlag and int(case) not in SINGLE_IMAGE_SUPPORTED_CASES:
        return
    augName = imageAugmentationMap[int(case)][0]
    print("\n")
    bitDepths = list(BitDepthTestMode)
    if qaMode:
        bitDepths = [BitDepthTestMode.U8_TO_U8]
    if singleImageFlag:
        bitDepths = [bd for bd in bitDepths if bd in SINGLE_IMAGE_SUPPORTED_BIT_DEPTHS]
    for bitDepth in bitDepths:
        for outputFormatToggle in list(OutputFormat):
            # There is no layout toggle for PLN1 case, so skip this case
            if layout == Layout.PLN1.value and outputFormatToggle == OutputFormat.TOGGLE:
                continue
            if augName in {"box_filter", "median_filter", "gaussian_filter", "erode", "dilate"}:
                for kernelSize in range(3, 10, 2):
                    run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, kernelSize, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList, singleImageFlag)
            elif augName == "noise":
                # Run all variants of noise type functions with additional argument of noiseType = gausssianNoise / shotNoise / saltandpepperNoise
                for noiseType in range(3):
                    run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, noiseType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList, singleImageFlag)
                    print("")
            elif augName in {"resize", "rotate", "warp_affine", "remap", "warp_perspective"}:
                # Run all variants of interpolation functions with additional argument of interpolationType = bicubic / bilinear / gaussian / nearestneigbor / lanczos / triangular
                if singleImageFlag:
                    interpolationRange = 1  # nearestneighbor only
                elif augName in {"remap", "warp_perspective"}:
                    interpolationRange = 2
                else:
                    interpolationRange = 6
                for interpolationType in range(interpolationRange):
                    run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, interpolationType, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList, singleImageFlag)
                    print("")
            elif augName == "channel_permute":
                # Run all variants of swap channel functions with additional argument of swapOrder (0 - 5)
                for swapOrder in range(6):
                    run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, swapOrder, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList, singleImageFlag)
                    print("")
            elif augName == "sobel_filter":
                for kernelSizeAndGradient in range(9):
                    run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, kernelSizeAndGradient, numRuns, testType, layout, qaMode, decoderType, batchSize, roiList, singleImageFlag)
                    print("")
            else:
                run_performance_test_cmd(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, bitDepth, outputFormatToggle, case, "0", numRuns, testType, layout, qaMode, decoderType, batchSize, roiList, singleImageFlag)
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
    parser.add_argument('--decoder_type', type = int, default = 0, help = "Input loader — 0 = packed .rgb (default) / 1 = OpenCV (requires Tensor_image built with OpenCV)")
    parser.add_argument('--num_runs', type = int, default = 1, help = "Specifies the number of runs for running the performance tests")
    parser.add_argument('--preserve_output', type = int, default = 1, help = "preserves the output of the program - (0 = override output / 1 = preserve output )" )
    parser.add_argument('--batch_size', type = int, default = 1, help = "Specifies the batch size to use for running tests. Default is 1.")
    parser.add_argument('--roi', nargs = 4, help = "specifies the roi values", required = False)
    parser.add_argument('--single_image', type = int, default = 0, help = "Processing type - (0 = batch processing / 1- single image processing)", required = False)
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
        print("Decoder Type must be 0 or 1 (0 = packed .rgb / 1 = OpenCV; OpenCV requires Tensor_image built with OpenCV). Aborting")
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
    elif args.single_image < 0 or args.single_image > 1:
        print(" Invalid parameter. Aborting")
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
singleImageFlag = args.single_image
roiList = ['0', '0', '0', '0'] if args.roi is None else args.roi

if qaMode and testType == TestType.UNIT_TEST.value and batchSize != 3:
    print("QA mode can only run with a batch size of 3.")
    exit(0)

if qaMode and testType == TestType.PERFORMANCE_TEST.value and batchSize != 8:
    print("Performance QA mode can only run with a batch size of 8.")
    exit(0)

# set the output folders and number of runs based on type of test (unit test / performance test)
if(testType == TestType.UNIT_TEST.value):
    if qaMode:
        outFilePath = outFolderPath + "/QA_RESULTS_HOST_" + timestamp
    else:
        outFilePath = outFolderPath + "/OUTPUT_IMAGES_HOST_" + timestamp
    numRuns = 1
elif(testType == TestType.PERFORMANCE_TEST.value):
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
run_cmake_build(scriptPath)

supportedCaseList = [key for key, values in imageAugmentationMap.items() if "HOST" in values]
if testType == TestType.UNIT_TEST.value:
    noCaseSupported = all(int(case) not in supportedCaseList for case in caseList)
    if noCaseSupported:
        print("\ncase numbers %s are not supported" % caseList)
        exit(0)
    for case in caseList:
        if int(case) not in imageAugmentationMap:
            continue
        if imageAugmentationMap[int(case)][0] == "ricap" and (("--input_path1" not in sys.argv and "--input_path2" not in sys.argv) or qaMode):
            srcPath1 = ricapInFilePath
            srcPath2 = ricapInFilePath
        elif imageAugmentationMap[int(case)][0] == "lens_correction" and (("--input_path1" not in sys.argv and "--input_path2" not in sys.argv) or qaMode):
            srcPath1 = lensCorrectionInFilePath
            srcPath2 = lensCorrectionInFilePath
        else:
            srcPath1 = inFilePath1
            srcPath2 = inFilePath2
        # if QA mode is enabled overwrite the input folders with the folders used for generating golden outputs
        if qaMode and (imageAugmentationMap[int(case)][0] not in {"ricap", "lens_correction"}):
            srcPath1 = inFilePath1
            srcPath2 = inFilePath2
        for layout in list(Layout):
            dstPathTemp, logFileLayout = process_layout(layout, qaMode, case, dstPath, "host", ImageAugmentationGroupMap, func_group_finder, imageAugmentationMap)

            if not qaMode:
                if not os.path.isdir(dstPathTemp):
                    os.mkdir(dstPathTemp)

            run_unit_test(srcPath1, srcPath2, dstPathTemp, case, numRuns, testType, layout.value, qaMode, decoderType, batchSize, roiList, singleImageFlag)
    if not qaMode:
        create_layout_directories(dstPath)
else:
    noCaseSupported = all(int(case) not in imageAugmentationMap for case in caseList)
    if noCaseSupported:
        print("case numbers %s are not supported" % caseList)
        exit(0)
    for case in caseList:
        if int(case) not in imageAugmentationMap:
            continue
        # if QA mode is enabled overwrite the input folders with the folders used for generating golden outputs
        func_name = imageAugmentationMap[int(case)][0]

        if qaMode and func_name != "ricap":
            srcPath1 = inFilePath1
            srcPath2 = inFilePath2
        elif func_name == "ricap" and "--input_path1" not in sys.argv and "--input_path2" not in sys.argv:
            srcPath1 = ricapInFilePath
            srcPath2 = ricapInFilePath
        elif func_name == "lens_correction" and "--input_path1" not in sys.argv and "--input_path2" not in sys.argv:
            srcPath1 = lensCorrectionInFilePath
            srcPath2 = lensCorrectionInFilePath
        for layout in list(Layout):
            dstPathTemp, logFileLayout = process_layout(layout, qaMode, case, dstPath, "host", ImageAugmentationGroupMap, func_group_finder, imageAugmentationMap)
            run_performance_test(loggingFolder, logFileLayout, srcPath1, srcPath2, dstPath, case, numRuns, testType, layout.value, qaMode, decoderType, batchSize, roiList, singleImageFlag)

# List of augmentation names without QA support
nonQAaugs = ["jitter", "noise", "fog", "rain", "warp_affine", "warp_perspective", "gaussian_filter", "spatter"]

# Find all case keys matching these names
nonQACaseList = [str(k) for k, v in imageAugmentationMap.items() if v[0] in nonQAaugs]

if qaMode and testType == TestType.UNIT_TEST.value:
    qaFilePath = os.path.join(outFilePath, "QA_results.txt")
    checkFile = os.path.isfile(qaFilePath)
    if checkFile:
        if singleImageFlag:
            # For single image mode, use only the supported single image cases
            # Filter nonQACaseList to only include cases that are in SINGLE_IMAGE_SUPPORTED_CASES
            singleImageNonQACaseList = [case for case in nonQACaseList if int(case) in SINGLE_IMAGE_SUPPORTED_CASES]
            print("---------------------------------- Results of QA Test - Tensor_single_image_host ----------------------------------\n")
            print_qa_tests_summary(qaFilePath, list(SINGLE_IMAGE_SUPPORTED_CASES), singleImageNonQACaseList, "Tensor_single_image_host")
        else:
            print("---------------------------------- Results of QA Test - Tensor_image_host ----------------------------------\n")
            print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList, "Tensor_image_host")

# unit tests and QA mode disabled
if testType == TestType.UNIT_TEST.value and not qaMode:
    create_layout_directories(dstPath)
# Performance tests
elif (testType == TestType.PERFORMANCE_TEST.value):
    logFileList = get_log_file_list(preserveOutput, singleImageFlag)

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
