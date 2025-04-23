# Copyright (c) 2020 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

from datetime import datetime
from subprocess import Popen, PIPE
import argparse
import os
import shutil
import sys
import platform

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2021 - 2024, AMD MIVisionX - AMD OpenCV EXT Vision Test Full Report"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "mivisionx.support@amd.com"
__status__ = "Shipping"


def shell(cmd):
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    output = p.communicate()[0][0:-1]
    return output


def write_formatted(output, f):
    f.write("````\n")
    f.write("%s\n\n" % output)
    f.write("````\n")


def strip_libtree_addresses(lib_tree):
    return lib_tree


# OpenCV Ext GDF Tests
gdfListOpenCV = [
    'absdiff.gdf',
    'adaptiveThreshold.gdf',
    'add.gdf',
    'addWeighted.gdf',
    'bilateralFilter.gdf',
    'bitwise_and.gdf',
    'bitwise_not.gdf',
    'bitwise_or.gdf',
    'bitwise_xor.gdf',
    'blur.gdf',
    'boxFilter.gdf',
    'canny.gdf',
    'compare.gdf',
    'convertScaleAbs.gdf',
    'cornerMinEigenVal.gdf',
    'cornerharris.gdf',
    'cvBuildPyramid.gdf',
    'cvtColor.gdf',
    'dilate.gdf',
    'distanceTransform.gdf',
    'divide.gdf',
    'erode.gdf',
    'fastNlMeansDenoising.gdf',
    'fastNlMeansDenoisingColored.gdf',
    'filter2D.gdf',
    'flip.gdf',
    'gaussianBlur.gdf',
    'goodFeatures.gdf',
    'laplacian.gdf',
    'medianBlur.gdf',
    'morphologyEX.gdf',
    'multiply.gdf',
    'pyrDown.gdf',
    'pyrUp.gdf',
    'resize.gdf',
    'scharr.gdf',
    'sepFilter2D.gdf',
    'simple_blob.gdf',
    'sobel.gdf',
    'subtract.gdf',
    'threshold.gdf',
    'transpose.gdf',
    'warpAffine.gdf',
    'warpPerspective.gdf',
    'BRISK_Compute.gdf',
    'BRISK_Detect.gdf',
    'FAST.gdf',
    'MSER_Detect.gdf',
    'ORB_Compute.gdf',
    'ORB_Detect.gdf'
    # contrib nodes - turn on with contrib build
    #'star_Detect.gdf',
    #'SIFT_Compute.gdf',
    #'SIFT_Detect.gdf',
    #'SURF_Compute.gdf',
    #'SURF_Detect.gdf'
]

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--runvx_directory',    type=str, default='',
                    help='RunVX Executable Directory - required')
parser.add_argument('--list_tests',         type=str, default='no',
                    help='List Vision Performance Tests - optional (default:no [options:no/yes])')
parser.add_argument('--num_frames',         type=int, default=1000,
                    help='Run Test for X number of frames - optional (default:1000 [range:1 - N])')

args = parser.parse_args()

runvxDir = args.runvx_directory
listTest = args.list_tests
numFrames = args.num_frames


# check arguments
if runvxDir == '':
    print("\nERROR: RunVX Executable Directory Required\n")
    print("USAGE: python runVisionTests.py --help\n")
    exit()

if listTest not in ('no', 'yes'):
    print(
        "ERROR: List Vision Performance Tests options supported - [no or yes]")
    exit()

if not 1 <= numFrames <= 10000:
    print("\nERROR: Vision Test Number of Frames not in range - [1 - 10000]\n")
    exit()


# List Vision Functionality tests
if listTest == 'yes':
    print(" %-5s - %-30s\n" % ('Test ID', 'Test Name'))
    for i in range(len(gdfListOpenCV)):
        gdfName = gdfListOpenCV[i]
        print("   %-5d - %-30s\n" % ((i+1), gdfName))
    exit()


print("\nMIVisionX runVisionTests V-"+__version__+"\n")

# RunVX Application
runVX_exe = runvxDir+'/runvx'
RunVXapp = os.path.abspath(runVX_exe)
scriptPath = os.path.dirname(os.path.realpath(__file__))
if (os.path.isfile(RunVXapp)):
    print("STATUS: RunVX path - "+RunVXapp)
else:
    print("\nERROR: RunVX Executable Not Found\n")
    exit()

# Get cwd
cwd = os.getcwd()

# OpenCV Vision Functionality Tests
outputDirectory = 'gdf_test_results'
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)
else:
    shutil.rmtree(outputDirectory)
    os.makedirs(outputDirectory)

outputDataDirectory = scriptPath+'/gdfs/amd_opencv_test_results'
if not os.path.exists(outputDataDirectory):
    os.makedirs(outputDataDirectory)
else:
    shutil.rmtree(outputDataDirectory)
    os.makedirs(outputDataDirectory)

# create directory to store vision accurarcy test results
print("\nrunOpenCVTests - OpenVX - OpenCV EXT Vision Tests\n")
for i in range(len(gdfListOpenCV)):
    testFileName = gdfListOpenCV[i]
    echoGDF = 'Running Test GDF - '+testFileName
    os.system('echo '+echoGDF +
            ' | tee -a gdf_test_results/openCV_visionOutput.log')
    print(RunVXapp+' -frames:'+str(numFrames)+' -dump-profile file '+scriptPath+'/gdfs/' +
            testFileName+' | tee -a gdf_test_results/openCV_visionOutput.log')
    os.system(RunVXapp+' -frames:'+str(numFrames)+' -dump-profile file '+scriptPath+'/gdfs/' +
            testFileName+' | tee -a gdf_test_results/openCV_visionOutput.log')
    print("\n")
print("\nSTATUS: OpenCV EXT Vision Test Results - " +
    scriptPath+"/gdfs/gdf_test_results\n")


orig_stdout = sys.stdout
sys.stdout = open('gdf_test_results/openCV_visionOutput.md', 'a')
echo_1 = '| OpenVX Node                             |Frames Count| tmp (ms) | avg (ms) | min (ms) | max (ms) |'
print(echo_1)
echo_2 = '|-----------------------------------------|------------|----------|----------|----------|----------|'
print(echo_2)
sys.stdout = orig_stdout
print(echo_1)
print(echo_2)
runAwk_csv = r'''awk 'BEGIN { node = "xxx"; } /Running Test GDF - / { node = $5; } /CPU,GRAPH/ { printf("| %-39s | %10d | %8.3f | %8.3f | %8.3f | %8.3f |\n", node, $1, $2, $3, $4, $5) }' gdf_test_results/openCV_visionOutput.log | tee -a gdf_test_results/openCV_visionOutput.md'''
os.system(runAwk_csv)

# get data
platform_name = platform.platform()
platform_name_fq = shell('hostname --all-fqdns')
platform_ip = shell('hostname -I')[0:-1]  # extra trailing space

file_dtstr = datetime.now().strftime("%Y%m%d")
reportFilename = 'opencv_vision_report_%s_%s.md' % (
    platform_name, file_dtstr)
report_dtstr = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
sys_info = shell('inxi -c0 -S')

cpu_info = shell('inxi -c0 -C')
# cpu_info = cpu_info.split('\n')[0]  # strip out clock speeds

gpu_info = shell('inxi -c0 -G')
# gpu_info = gpu_info.split('\n')[0]  # strip out X info

memory_info = shell('inxi -c 0 -m')
board_info = shell('inxi -c0 -M')

lib_tree = shell('ldd '+RunVXapp)
lib_tree = strip_libtree_addresses(lib_tree)

# Write Report
with open(reportFilename, 'w') as f:
    f.write("MIVisionX - OpenCV Extension Report\n")
    f.write("================================\n")
    f.write("\n")

    f.write("Generated: %s\n" % report_dtstr)
    f.write("\n")

    f.write("Platform: %s (%s)\n" % (platform_name_fq, platform_ip))
    f.write("--------\n")
    f.write("\n")

    write_formatted(sys_info, f)
    write_formatted(cpu_info, f)
    write_formatted(gpu_info, f)
    write_formatted(board_info, f)
    write_formatted(memory_info, f)

    f.write("\n\nBenchmark Report\n")
    f.write("--------\n")
    f.write("\n")
    f.write("\n")
    with open('gdf_test_results/openCV_visionOutput.md') as benchmarkFile:
        for line in benchmarkFile:
            f.write("%s" % line)
    f.write("\n")
    f.write("\n")
    f.write("Dynamic Libraries Report\n")
    f.write("-----------------\n")
    f.write("\n")
    write_formatted(lib_tree, f)
    f.write("\n")

    f.write(
        "\n\n---\n**Copyright AMD ROCm MIVisionX 2021 - 2024 -- runOpenCVTests.py V-"+__version__+"**\n")
    f.write("\n")

    # report file
    reportFileDir = os.path.abspath(reportFilename)
    print("\nSTATUS: Output Report File - "+reportFileDir)

print("\nrunOpenCVTests.py completed - V:"+__version__+"\n")
