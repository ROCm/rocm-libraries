# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import sys
import subprocess  # nosec
import shutil
import argparse

test_case_augmentation_map = {
    0: "audio_decoder",
    1: "preemphasis_filter",
    2: "spectrogram",
    3: "downmix",
    4: "to_decibels",
    5: "resample",
    6: "tensor_add_tensor",
    7: "tensor_mul_scalar",
    8: "non_silent_region",
    9: "slice",
    10: "mel_filter_bank",
    11: "normalize"
}

def run_unit_test(rocal_data_path, qa_mode, gpu, downmix, build_folder_path, case_list):
    passed_cases = []
    failed_cases = []
    num_passed = 0
    num_failed = 0
    for case in case_list:
        if case == 3:
            src_path = rocal_data_path + "/rocal_data/multi_channel_wav"
            downmix = 1
        else:
            src_path = rocal_data_path + "/rocal_data/audio"
        print("\n\n")
        result = subprocess.run([build_folder_path + "/build/audio_tests", src_path, str(case), str(downmix), str(gpu), str(qa_mode)], stdout=subprocess.PIPE)    # nosec
        try:
            decoded_stdout = result.stdout.decode('utf-8')
        except UnicodeDecodeError as e:
            # Handle the error by replacing or ignoring problematic characters
            decoded_stdout = result.stdout.decode('utf-8', errors='replace')

        # check the QA status of the test case
        if "PASSED" in decoded_stdout:
            num_passed += 1
            passed_cases.append(test_case_augmentation_map[case])
        else:
            num_failed += 1
            failed_cases.append(test_case_augmentation_map[case])
        print(decoded_stdout)

    if qa_mode:
        print("Number of PASSED:", num_passed)
        print(passed_cases)
        print("Number of FAILED:", num_failed)
        print(failed_cases)
    print("------------------------------------------------------------------------------------------")

def audio_test_suite_parser_and_validator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type = int, default = 0, help = "Backend to run (1 - GPU / 0 - CPU)")
    parser.add_argument("--downmix", type = bool, default = False, help = "Runs the Audio with downMix option (True / False)")
    parser.add_argument("--test_case", type = int, default = None, help = "Testing case")
    parser.add_argument("--qa_mode", type = int, default = 1, help = "Compare outputs with reference outputs (0 - disabled / 1 - enabled)")
    args = parser.parse_args()
    return args

def main():
    script_path = os.path.dirname(os.path.realpath(__file__))
    rocal_data_path = os.environ.get("ROCAL_DATA_PATH")
    if rocal_data_path is None:
        print("Need to export ROCAL_DATA_PATH")
        sys.exit()

    sys.dont_write_bytecode = True
    build_folder_path = os.getcwd()

    args = audio_test_suite_parser_and_validator()
    gpu = args.gpu
    downmix = args.downmix
    test_case = args.test_case
    qa_mode = args.qa_mode

    case_list = list(test_case_augmentation_map.keys())

    if test_case is not None: 
        if test_case not in case_list:
            print(" Invalid Test Case! ")
            exit()
        else:
            case_list = [test_case]

    # Enable extglob
    if os.path.exists(build_folder_path + "/build"):
        shutil.rmtree(build_folder_path + "/build")
    os.makedirs(build_folder_path + "/build")
    os.chdir(build_folder_path + "/build")

    # Run cmake and make commands
    subprocess.run(["cmake", script_path], cwd=".")   # nosec
    subprocess.run(["make", "-j16"], cwd=".")    # nosec

    run_unit_test(rocal_data_path, qa_mode, gpu, downmix, build_folder_path, case_list)

if __name__ == "__main__":
    main()
