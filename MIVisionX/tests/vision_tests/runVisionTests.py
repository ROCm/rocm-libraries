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
__copyright__ = "Copyright 2018 - 2024, AMD MIVisionX - Vision Test Full Report"
__license__ = "MIT"
__version__ = "1.3.0"
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


# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--runvx_directory',    type=str, default='/opt/rocm/bin',
                    help='RunVX Executable Directory - optional (default:/opt/rocm/bin)')
parser.add_argument('--hardware_mode',      type=str, default='CPU',
                    help='OpenVX Vision Function Target - optional (default:CPU [options:CPU/GPU])')
parser.add_argument('--list_tests',         type=str, default='no',
                    help='List Vision Performance Tests - optional (default:no [options:no/yes])')
parser.add_argument('--test_filter',        type=int, default=0,
                    help='Vision Performance Test Filter - optional (default:0 [range:1 - N])')
parser.add_argument('--num_frames',         type=int, default=1000,
                    help='Run Test for X number of frames - optional (default:1000 [range:1 - N])')
parser.add_argument('--functionality',      type=str, default='yes',
                    help='Vision Functionality Tests Enabled - optional (default:yes [options:no/yes])')
parser.add_argument('--backend_type',       type=str, default='CPU',
                    help='Backend type - optional (default:CPU [options:CPU/HIP/OCL])')
parser.add_argument('--profiling',          type=str, default='no',
                    help='GPU profiling with rocprof - optional (default:no [options:yes/no])')
parser.add_argument('--width',              type=int, default='1920',
                    help='Image width for Vision Performance tests - optional (default:1920) [range: 1 - 7680]')
parser.add_argument('--height',             type=int, default='1080',
                    help='Image height for Vision Performance tests - optional (default:1080) [range: 1 - 7680]')
parser.add_argument('--perf_counters',        type=str, default='no',
                    help='Collect performance counters with rocprof - optional (default:no [options:yes/no])')


args = parser.parse_args()

runvxDir = args.runvx_directory
hardwareMode = args.hardware_mode
listTest = args.list_tests
testFilter = args.test_filter
numFrames = args.num_frames
functionalityTests = args.functionality
backendType = args.backend_type
profilingOption = args.profiling
width = args.width
height = args.height
widthDiv2 = int(width / 2)
heightDiv2 = int(height / 2)
perfCounters = args.perf_counters

# Vision Accuracy Tests
visionTestConfig = [
    '01_absDiff.gdf',
    '02_accumulate.gdf',
    '03_accumulateSquared.gdf',
    '04_accumulateWeighted.gdf',
    '05_add.gdf',
    '06_and.gdf',
    '07_box.gdf',
    '08_canny.gdf',
    '09_channelCombine.gdf',
    '10_channelExtract.gdf',
    '11_colorConvert.gdf',
    '12_convertDepth.gdf',
    '13_convolve.gdf',
    '14_dilate.gdf',
    '15_equalizeHistogram.gdf',
    '16_erode.gdf',
    '17_fastCorners.gdf',
    '18_gaussian.gdf',
    '19_harrisCorners.gdf',
    '20_halfScaleGaussian.gdf',
    '21_histogram.gdf',
    '22_integralImage.gdf',
    '23_magnitude.gdf',
    '24_meanStdDev.gdf',
    '25_median.gdf',
    '26_minMaxLoc.gdf',
    '27_multiply.gdf',
    '28_not.gdf',
    '29_opticalFlowLK.gdf',
    '30_or.gdf',
    '31_phase.gdf',
    '32_gaussianPyramid.gdf',
    '33_remap.gdf',
    '34_scaleImage.gdf',
    '35_sobel.gdf',
    '36_subtract.gdf',
    '37_tableLookup.gdf',
    '38_threshold.gdf',
    '39_warpAffine.gdf',
    '40_warpPerspective.gdf',
    '41_xor.gdf',
    '42_scale_nv12_to_rgb.gdf'
]

# OpenVX Vision Functions 1080P
openvxNodes = [
    # Arithmetic kernels
    ('AbsDiff_U8_U8U8',                           'org.khronos.openvx.absdiff uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb image:' + str(width) + ',' + str(height) + ',U008'),
    ('AbsDiff_S16_S16S16_Sat',                    'org.khronos.openvx.absdiff uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',S016,0xbb image:' + str(width) + ',' + str(height) + ',S016'),
    ('Add_U8_U8U8_Wrap',                          'org.khronos.openvx.add uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb !WRAP image:' + str(width) + ',' + str(height) + ',U008'),
    ('Add_U8_U8U8_Sat',                           'org.khronos.openvx.add uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb !SATURATE image:' + str(width) + ',' + str(height) + ',U008'),
    ('Add_S16_U8U8_Wrap',                         'org.khronos.openvx.add uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb !WRAP image:' + str(width) + ',' + str(height) + ',S016'),
    ('Add_S16_S16U8_Wrap',                        'org.khronos.openvx.add uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb !WRAP image:' + str(width) + ',' + str(height) + ',S016'),
    ('Add_S16_S16U8_Sat',                         'org.khronos.openvx.add uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb !SATURATE image:' + str(width) + ',' + str(height) + ',S016'),
    ('Add_S16_S16S16_Wrap',                       'org.khronos.openvx.add uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',S016,0xbb !WRAP image:' + str(width) + ',' + str(height) + ',S016'),
    ('Add_S16_S16S16_Sat',                        'org.khronos.openvx.add uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',S016,0xbb !SATURATE image:' + str(width) + ',' + str(height) + ',S016'),
    ('Sub_U8_U8U8_Wrap',                          'org.khronos.openvx.subtract uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb !WRAP image:' + str(width) + ',' + str(height) + ',U008'),
    ('Sub_U8_U8U8_Sat',                           'org.khronos.openvx.subtract uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb !SATURATE image:' + str(width) + ',' + str(height) + ',U008'),
    ('Sub_S16_U8U8_Wrap',                         'org.khronos.openvx.subtract uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb !WRAP image:' + str(width) + ',' + str(height) + ',S016'),
    ('Sub_S16_S16U8_Wrap',                        'org.khronos.openvx.subtract uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb !WRAP image:' + str(width) + ',' + str(height) + ',S016'),
    ('Sub_S16_S16U8_Sat',                         'org.khronos.openvx.subtract uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb !SATURATE image:' + str(width) + ',' + str(height) + ',S016'),
    ('Sub_S16_U8S16_Wrap',                        'org.khronos.openvx.subtract uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',S016,0xbb !WRAP image:' + str(width) + ',' + str(height) + ',S016'),
    ('Sub_S16_U8S16_Sat',                         'org.khronos.openvx.subtract uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',S016,0xbb !SATURATE image:' + str(width) + ',' + str(height) + ',S016'),
    ('Sub_S16_S16S16_Wrap',                       'org.khronos.openvx.subtract uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',S016,0xbb !WRAP image:' + str(width) + ',' + str(height) + ',S016'),
    ('Sub_S16_S16S16_Sat',                        'org.khronos.openvx.subtract uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',S016,0xbb !SATURATE image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_U8_U8U8_Wrap_Trunc',                    'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',U008,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:' + str(width) + ',' + str(height) + ',U008'),
    ('Mul_U8_U8U8_Wrap_Round',                    'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',U008,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:' + str(width) + ',' + str(height) + ',U008'),
    ('Mul_U8_U8U8_Sat_Trunc',                     'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',U008,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:' + str(width) + ',' + str(height) + ',U008'),
    ('Mul_U8_U8U8_Sat_Round',                     'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',U008,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:' + str(width) + ',' + str(height) + ',U008'),
    ('Mul_S16_U8U8_Wrap_Trunc',                   'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',U008,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_U8U8_Wrap_Round',                   'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',U008,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_U8U8_Sat_Trunc',                    'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',U008,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_U8U8_Sat_Round',                    'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',U008,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_S16U8_Wrap_Trunc',                  'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',S016,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_S16U8_Wrap_Round',                  'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',S016,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_S16U8_Sat_Trunc',                   'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',S016,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_S16U8_Sat_Round',                   'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',S016,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_S16S16_Wrap_Trunc',                 'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',S016,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',S016,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_ZERO image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_S16S16_Wrap_Round',                 'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',S016,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',S016,0xbb scalar:FLOAT32,1.0 !WRAP !ROUND_POLICY_TO_NEAREST_EVEN image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_S16S16_Sat_Trunc',                  'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',S016,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',S016,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_ZERO image:' + str(width) + ',' + str(height) + ',S016'),
    ('Mul_S16_S16S16_Sat_Round',                  'org.khronos.openvx.multiply uniform-image:' + str(width) + ',' + str(height) + ',S016,0xaa uniform-image:' + \
     str(width) + ',' + str(height) + ',S016,0xbb scalar:FLOAT32,1.0 !SATURATE !ROUND_POLICY_TO_NEAREST_EVEN image:' + str(width) + ',' + str(height) + ',S016'),
    ('Magnitude_S16_S16S16',                      'org.khronos.openvx.magnitude uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',S016,0xbb image:' + str(width) + ',' + str(height) + ',S016'),
    ('Phase_U8_S16S16',                           'org.khronos.openvx.phase uniform-image:' + str(width) + ',' + str(height) + \
     ',S016,0xaa uniform-image:' + str(width) + ',' + str(height) + ',S016,0xbb image:' + str(width) + ',' + str(height) + ',U008'),
    ('WeightedAverage_U8_U8U8',                   'org.khronos.openvx.weighted_average uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa scalar:FLOAT32,0.25 uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb image:' + str(width) + ',' + str(height) + ',U008'),
    # logical kernels
    ('And_U8_U8U8',                               'org.khronos.openvx.and uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb image:' + str(width) + ',' + str(height) + ',U008'),
    ('Or_U8_U8U8',                                'org.khronos.openvx.or uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb image:' + str(width) + ',' + str(height) + ',U008'),
    ('Xor_U8_U8U8',                               'org.khronos.openvx.xor uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb image:' + str(width) + ',' + str(height) + ',U008'),
    ('Not_U8_U8',                                 'org.khronos.openvx.not uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xaa image:' + str(width) + ',' + str(height) + ',U008'),
    # Color kernels
    ('Lut_U8_U8',                                 'org.khronos.openvx.table_lookup uniform-image:' + str(width) + \
     ',' + str(height) + ',U008,0xaa lut:UINT8,256 image:' + str(width) + ',' + str(height) + ',U008'),
    ('ColorDepth_U8_S16_Wrap',                    'org.khronos.openvx.convertdepth uniform-image:' + str(width) + \
     ',' + str(height) + ',S016,0xaa image:' + str(width) + ',' + str(height) + ',U008 !WRAP scalar:INT32,1'),
    ('ColorDepth_U8_S16_Sat',                     'org.khronos.openvx.convertdepth uniform-image:' + str(width) + \
     ',' + str(height) + ',S016,0xaa image:' + str(width) + ',' + str(height) + ',U008 !SATURATE scalar:INT32,1'),
    ('ColorDepth_S16_U8',                         'org.khronos.openvx.convertdepth uniform-image:' + str(width) + \
     ',' + str(height) + ',U008,0xaa image:' + str(width) + ',' + str(height) + ',S016 !WRAP scalar:INT32,1'),
    ('ChannelExtract_U8_U16_Pos0',                'org.khronos.openvx.channel_extract uniform-image:' + str(width) + ',' + str(height) + \
     ',YUYV !CHANNEL_Y image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U16_Pos1',                'org.khronos.openvx.channel_extract uniform-image:' + str(width) + ',' + str(height) + \
     ',YUYV !CHANNEL_Y image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U24_Pos0',                'org.khronos.openvx.channel_extract uniform-image:' + \
     str(width) + ',' + str(height) + ',RGB2,0xaabbcc !CHANNEL_R image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U24_Pos1',                'org.khronos.openvx.channel_extract uniform-image:' + \
     str(width) + ',' + str(height) + ',RGB2,0xaabbcc !CHANNEL_G image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U24_Pos2',                'org.khronos.openvx.channel_extract uniform-image:' + \
     str(width) + ',' + str(height) + ',RGB2,0xaabbcc !CHANNEL_B image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U32_Pos0_UYVY',           'org.khronos.openvx.channel_extract uniform-image:' + str(width) + \
     ',' + str(height) + ',UYVY,0xaabbcc !CHANNEL_U image:' + str(widthDiv2) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U32_Pos1_YUYV',           'org.khronos.openvx.channel_extract uniform-image:' + str(width) + \
     ',' + str(height) + ',YUYV,0xaabbcc !CHANNEL_U image:' + str(widthDiv2) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U32_Pos2_UYVY',           'org.khronos.openvx.channel_extract uniform-image:' + str(width) + \
     ',' + str(height) + ',UYVY,0xaabbcc !CHANNEL_V image:' + str(widthDiv2) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U32_Pos3_YUYV',           'org.khronos.openvx.channel_extract uniform-image:' + str(width) + \
     ',' + str(height) + ',YUYV,0xaabbcc !CHANNEL_V image:' + str(widthDiv2) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U32_Pos0_RGBX',           'org.khronos.openvx.channel_extract uniform-image:' + str(width) + \
     ',' + str(height) + ',RGBA,0xaabbccdd !CHANNEL_R image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U32_Pos1_RGBX',           'org.khronos.openvx.channel_extract uniform-image:' + str(width) + \
     ',' + str(height) + ',RGBA,0xaabbccdd !CHANNEL_G image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U32_Pos2_RGBX',           'org.khronos.openvx.channel_extract uniform-image:' + str(width) + \
     ',' + str(height) + ',RGBA,0xaabbccdd !CHANNEL_B image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8_U32_Pos3_RGBX',           'org.khronos.openvx.channel_extract uniform-image:' + str(width) + \
     ',' + str(height) + ',RGBA,0xaabbccdd !CHANNEL_A image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8U8U8_U24',                 'org.khronos.openvx.channel_extract uniform-image:' + \
     str(width) + ',' + str(height) + ',RGB2,0xaabbcc !CHANNEL_R image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8U8U8_U32',                 'org.khronos.openvx.channel_extract uniform-image:' + str(width) + \
     ',' + str(height) + ',RGBA,0xaabbccdd !CHANNEL_R image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelExtract_U8U8U8U8_U32',               'org.khronos.openvx.channel_extract uniform-image:' + str(width) + \
     ',' + str(height) + ',RGBA,0xaabbccdd !CHANNEL_R image:' + str(width) + ',' + str(height) + ',U008'),
    ('ChannelCombine_U32_U8U8U8U8_RGBX',          'org.khronos.openvx.channel_combine uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa uniform-image:' + str(width) + ',' + str(height) + ',U008,0xbb uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xcc uniform-image:' + str(width) + ',' + str(height) + ',U008,0xdd image:' + str(width) + ',' + str(height) + ',RGBA'),
    ('ColorConvert_RGB_RGBX',                     'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',RGBA,0xaabbccdd image:' + str(width) + ',' + str(height) + ',RGB2'),
    ('ColorConvert_RGB_UYVY',                     'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',UYVY,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGB2'),
    ('ColorConvert_RGB_YUYV',                     'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',YUYV,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGB2'),
    ('ColorConvert_RGB_IYUV',                     'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',IYUV,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGB2'),
    ('ColorConvert_RGB_NV12',                     'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',NV12,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGB2'),
    ('ColorConvert_RGB_NV21',                     'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',NV21,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGB2'),
    ('ColorConvert_RGBX_RGB',                     'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',RGB2,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGBA'),
    ('ColorConvert_RGBX_UYVY',                    'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',UYVY,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGBA'),
    ('ColorConvert_RGBX_YUYV',                    'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',YUYV,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGBA'),
    ('ColorConvert_RGBX_IYUV',                    'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',IYUV,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGBA'),
    ('ColorConvert_RGBX_NV12',                    'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',NV12,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGBA'),
    ('ColorConvert_RGBX_NV21',                    'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',NV21,0xaabbcc image:' + str(width) + ',' + str(height) + ',RGBA'),
    ('ColorConvert_IYUV_RGB',                     'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',RGB2,0xaabbcc image:' + str(width) + ',' + str(height) + ',IYUV'),
    ('ColorConvert_IYUV_RGBX',                    'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',RGBA,0xaabbccdd image:' + str(width) + ',' + str(height) + ',IYUV'),
    ('FormatConvert_IYUV_UYVY',                   'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',UYVY,0xaabbcc image:' + str(width) + ',' + str(height) + ',IYUV'),
    ('FormatConvert_IYUV_YUYV',                   'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',YUYV,0xaabbcc image:' + str(width) + ',' + str(height) + ',IYUV'),
    ('ColorConvert_NV12_RGB',                     'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',RGB2,0xaabbcc image:' + str(width) + ',' + str(height) + ',NV12'),
    ('ColorConvert_NV12_RGBX',                    'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',RGBA,0xaabbccdd image:' + str(width) + ',' + str(height) + ',NV12'),
    ('FormatConvert_NV12_UYVY',                   'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',UYVY,0xaabbcc image:' + str(width) + ',' + str(height) + ',NV12'),
    ('FormatConvert_NV12_YUYV',                   'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',YUYV,0xaabbcc image:' + str(width) + ',' + str(height) + ',NV12'),
    ('ColorConvert_YUV4_RGB',                     'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',RGB2,0xaabbcc image:' + str(width) + ',' + str(height) + ',YUV4'),
    ('ColorConvert_YUV4_RGBX',                    'org.khronos.openvx.color_convert uniform-image:' + \
     str(width) + ',' + str(height) + ',RGBA,0xaabbccdd image:' + str(width) + ',' + str(height) + ',YUV4'),
    ('FormatConvert_IUV_UV12',                    'org.khronos.openvx.color_convert uniform-image:' + str(width) + ',' + str(height) + \
     ',NV12,0xaabbcc image:' + str(width) + ',' + str(height) + ',IYUV'),
    ('FormatConvert_UV12_IUV',                    'org.khronos.openvx.color_convert uniform-image:' + str(width) + ',' + str(height) + \
     ',IYUV,0xaabbcc image:' + str(width) + ',' + str(height) + ',NV12'),
    ('FormatConvert_UV_UV12',                     'org.khronos.openvx.color_convert uniform-image:' + str(width) + ',' + str(height) + \
     ',NV12,0xaabbcc image:' + str(width) + ',' + str(height) + ',YUV4'),
    ('ScaleUp2x2_U8_U8',                          'org.khronos.openvx.color_convert uniform-image:' + str(width) + ',' + str(height) + \
     ',IYUV,0xaabbcc image:' + str(width) + ',' + str(height) + ',YUV4'),
    # filter kernels
    ('Box_U8_U8_3x3',                             'org.khronos.openvx.box_3x3 uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xaa image:' + str(width) + ',' + str(height) + ',U008'),
    ('Dilate_U8_U8_3x3',                          'org.khronos.openvx.dilate_3x3 uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xaa image:' + str(width) + ',' + str(height) + ',U008'),
    ('Erode_U8_U8_3x3',                           'org.khronos.openvx.erode_3x3 uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xaa image:' + str(width) + ',' + str(height) + ',U008'),
    ('Median_U8_U8_3x3',                          'org.khronos.openvx.median_3x3 uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xaa image:' + str(width) + ',' + str(height) + ',U008'),
    ('Gaussian_U8_U8_3x3',                        'org.khronos.openvx.gaussian_3x3 uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xaa image:' + str(width) + ',' + str(height) + ',U008'),
    ('ScaleGaussianHalf_U8_U8_3x3',               'org.khronos.openvx.halfscale_gaussian uniform-image:' + str(width) + \
     ',' + str(height) + ',U008,0xaa image:' + str(widthDiv2) + ',' + str(heightDiv2) + ',U008 scalar:INT32,3'),
    ('ScaleGaussianHalf_U8_U8_5x5',               'org.khronos.openvx.halfscale_gaussian uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa image:' + str(widthDiv2) + ',' + str(heightDiv2) + ',U008 scalar:INT32,5'),
    ('Convolve_U8_U8_3x3',                        'org.khronos.openvx.custom_convolution uniform-image:' + str(width) + ',' + \
     str(height) + ',U008,0xaa "convolution:3,3:INIT,{-1;-1;-1;-1;16;-1;-1;-1;-1}" image:' + str(width) + ',' + str(height) + ',U008'),
    ('Convolve_S16_U8_3x3',                       'org.khronos.openvx.custom_convolution uniform-image:' + str(width) + ',' + \
     str(height) + ',U008,0xaa "convolution:3,3:INIT,{-1;-1;-1;-1;16;-1;-1;-1;-1}" image:' + str(width) + ',' + str(height) + ',S016'),
    ('Sobel_S16S16_U8_3x3_GXY',                   'org.khronos.openvx.sobel_3x3 uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xaa image:' + str(width) + ',' + str(height) + ',S016 image:' + str(width) + ',' + str(height) + ',S016'),
    ('Sobel_S16_U8_3x3_GX',                       'org.khronos.openvx.sobel_3x3 uniform-image:' + \
     str(width) + ',' + str(height) + ',U008,0xaa image:' + str(width) + ',' + str(height) + ',S016'),
    # Statistical kernels
    ('Threshold_U8_U8_Binary',                    'org.khronos.openvx.threshold uniform-image:' + str(width) + ',' + \
     str(height) + ',U008,0xaa threshold:BINARY,U008,U008:INIT,127 image:' + str(width) + ',' + str(height) + ',U008'),
    ('Threshold_U8_U8_Range',                     'org.khronos.openvx.threshold uniform-image:' + str(width) + ',' + \
     str(height) + ',U008,0xaa threshold:RANGE,U008,U008:INIT,100,200 image:' + str(width) + ',' + str(height) + ',U008'),
    ('Threshold_U8_S16_Binary',                   'org.khronos.openvx.threshold uniform-image:' + str(width) + ',' + \
     str(height) + ',S016,0xaa threshold:BINARY,S016,U008:INIT,127 image:' + str(width) + ',' + str(height) + ',U008'),
    ('Threshold_U8_S16_Range',                    'org.khronos.openvx.threshold uniform-image:' + str(width) + ',' + \
     str(height) + ',S016,0xaa threshold:RANGE,S016,U008:INIT,100,200 image:' + str(width) + ',' + str(height) + ',U008'),
    # geometric kernels
    ('ScaleImage_U8_U8_Nearest',                  'org.khronos.openvx.scale_image image:' + str(width) + \
     ',' + str(height) + ',U008 image:' + str(widthDiv2) + ',' + str(heightDiv2) + ',U008 !NEAREST_NEIGHBOR'),
    ('ScaleImage_U8_U8_Bilinear',                 'org.khronos.openvx.scale_image image:' + str(width) + \
     ',' + str(height) + ',U008 image:' + str(widthDiv2) + ',' + str(heightDiv2) + ',U008 !BILINEAR'),
    ('ScaleImage_U8_U8_Bilinear_Replicate',       'org.khronos.openvx.scale_image image:' + str(width) + ',' + \
     str(height) + ',U008 image:' + str(widthDiv2) + ',' + str(heightDiv2) + ',U008 !BILINEAR attr:BORDER_MODE:REPLICATE'),
    ('ScaleImage_U8_U8_Bilinear_Constant',        'org.khronos.openvx.scale_image image:' + str(width) + ',' + \
     str(height) + ',U008 image:' + str(widthDiv2) + ',' + str(heightDiv2) + ',U008 !BILINEAR attr:BORDER_MODE:CONSTANT,0'),
    ('ScaleImage_U8_U8_Area',                     'org.khronos.openvx.scale_image image:' + str(width) + \
     ',' + str(height) + ',U008 image:' + str(widthDiv2) + ',' + str(heightDiv2) + ',U008 !AREA'),
    ('WarpAffine_U8_U8_Nearest',                  'org.khronos.openvx.warp_affine image:' + str(width) + ',' + str(height) + \
     ',U008 "matrix:FLOAT32,2,3:INIT,{0.25;0;0;0.5;20;20}" !NEAREST_NEIGHBOR image:' + str(width) + ',' + str(height) + ',U008'),
    ('WarpAffine_U8_U8_Nearest_Constant',         'org.khronos.openvx.warp_affine image:' + str(width) + ',' + str(height) + \
     ',U008 "matrix:FLOAT32,2,3:INIT,{0.25;0;0;0.5;20;20}" !NEAREST_NEIGHBOR image:' + str(width) + ',' + str(height) + ',U008 attr:BORDER_MODE:CONSTANT,0'),
    ('WarpAffine_U8_U8_Bilinear',                 'org.khronos.openvx.warp_affine image:' + str(width) + ',' + str(height) + \
     ',U008 "matrix:FLOAT32,2,3:INIT,{0.25;0;0;0.5;20;20}" !BILINEAR image:' + str(width) + ',' + str(height) + ',U008'),
    ('WarpAffine_U8_U8_Bilinear_Constant',        'org.khronos.openvx.warp_affine image:' + str(width) + ',' + str(height) + \
     ',U008 "matrix:FLOAT32,2,3:INIT,{0.25;0;0;0.5;20;20}" !BILINEAR image:' + str(width) + ',' + str(height) + ',U008 attr:BORDER_MODE:CONSTANT,0'),
    # TBD -- Failure on GFX 1030 -- TURN OFF for OpenCL Backend
    #('WarpPerspective_U8_U8_Nearest',             'org.khronos.openvx.warp_perspective image:' + str(width) + ',' + str(height) + \
    # ',U008 "matrix:FLOAT32,3,3:INIT,{2;0.1;0;2;1.9;0;-1200;-360;1}" !NEAREST_NEIGHBOR image:' + str(width) + ',' + str(height) + ',U008'),
    #('WarpPerspective_U8_U8_Nearest_Constant',    'org.khronos.openvx.warp_perspective image:' + str(width) + ',' + str(height) + \
    # ',U008 "matrix:FLOAT32,3,3:INIT,{2;0.1;0;2;1.9;0;-1200;-360;1}" !NEAREST_NEIGHBOR image:' + str(width) + ',' + str(height) + ',U008 attr:BORDER_MODE:CONSTANT,0'),
    #('WarpPerspective_U8_U8_Bilinear',            'org.khronos.openvx.warp_perspective image:' + str(width) + ',' + str(height) + \
    # ',U008 "matrix:FLOAT32,3,3:INIT,{2;0.1;0;2;1.9;0;-1200;-360;1}" !BILINEAR image:' + str(width) + ',' + str(height) + ',U008'),
    #('WarpPerspective_U8_U8_Bilinear_Constant',   'org.khronos.openvx.warp_perspective image:' + str(width) + ',' + str(height) + \
    # ',U008 "matrix:FLOAT32,3,3:INIT,{2;0.1;0;2;1.9;0;-1200;-360;1}" !BILINEAR image:' + str(width) + ',' + str(height) + ',U008 attr:BORDER_MODE:CONSTANT,0'),
    #('Remap_S16_nearest',                         'org.khronos.openvx.remap image:' + str(width) + ',' + str(height) + \
    # ',U008 remap:' + str(width) + ',' + str(height) + ',' + str(width) + ',' + str(height) + ' !NEAREST_NEIGHBOR image:' + str(width) + ',' + str(height) + ',U008'),
    #('Remap_S16_nearest_constant',                'org.khronos.openvx.remap image:' + str(width) + ',' + str(height) + \
    # ',U008 remap:' + str(width) + ',' + str(height) + ',' + str(width) + ',' + str(height) + ' !NEAREST_NEIGHBOR image:' + str(width) + ',' + str(height) + ',U008 attr:BORDER_MODE:CONSTANT,0'),
    #('Remap_S16_bilinear',                        'org.khronos.openvx.remap image:' + str(width) + ',' + str(height) + \
    # ',U008 remap:' + str(width) + ',' + str(height) + ',' + str(width) + ',' + str(height) + ' !BILINEAR image:' + str(width) + ',' + str(height) + ',U008'),
    #('Remap_S16_bilinear_constant',               'org.khronos.openvx.remap image:' + str(width) + ',' + str(height) + \
    # ',U008 remap:' + str(width) + ',' + str(height) + ',' + str(width) + ',' + str(height) + ' !BILINEAR image:' + str(width) + ',' + str(height) + ',U008 attr:BORDER_MODE:CONSTANT,0'),
    # vision kernels
    ('FastCorners_XY_U8_NoSupression',            'org.khronos.openvx.fast_corners uniform-image:' + str(width) + ',' + str(height) + \
     ',U008 scalar:FLOAT32,80.0 scalar:BOOL,1 array:KEYPOINT,1000 scalar:SIZE,0'),
    ('FastCorners_XY_U8_Supression',              'org.khronos.openvx.fast_corners uniform-image:' + str(width) + ',' + str(height) + \
     ',U008 scalar:FLOAT32,80.0 scalar:BOOL,0 array:KEYPOINT,1000 scalar:SIZE,0'),
    ('Canny_3x3_L1Norm',                          'org.khronos.openvx.canny_edge_detector uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xab threshold:RANGE,U008,U008:INIT,80,100 scalar:INT32,3 !NORM_L1 image:' + str(width) + ',' + str(height) + ',U008'),
    ('Canny_3x3_L2Norm',                          'org.khronos.openvx.canny_edge_detector uniform-image:' + str(width) + ',' + str(height) + \
     ',U008,0xab threshold:RANGE,U008,U008:INIT,80,100 scalar:INT32,3 !NORM_L2 image:' + str(width) + ',' + str(height) + ',U008'),
]

#  Popular Video Sizes
# { 2160p: 3840x2160, 1440p: 2560x1440, 1080p: 1920x1080, 720p: 1280x720, 480p: 854x480, 360p: 640x360, 240p: 426x240 }
openvxNodeTestConfig = [
    # absdiff U8 - U8
    ('absdiff-240p-u8', 'org.khronos.openvx.absdiff image:426,240,U008 image:426,240,U008 image:426,240,U008'),
    ('absdiff-360p-u8', 'org.khronos.openvx.absdiff image:640,360,U008 image:640,360,U008 image:640,360,U008'),
    ('absdiff-480p-u8', 'org.khronos.openvx.absdiff image:854,480,U008 image:854,480,U008 image:854,480,U008'),
    ('absdiff-720p-u8', 'org.khronos.openvx.absdiff image:1280,720,U008 image:1280,720,U008 image:1280,720,U008'),
    ('absdiff-1080p-u8', 'org.khronos.openvx.absdiff image:1920,1080,U008 image:1920,1080,U008 image:1920,1080,U008'),
    ('absdiff-1440p-u8', 'org.khronos.openvx.absdiff image:2560,1440,U008 image:2560,1440,U008 image:2560,1440,U008'),
    ('absdiff-2160p-u8', 'org.khronos.openvx.absdiff image:3840,2160,U008 image:3840,2160,U008 image:3840,2160,U008'),
    # absdiff S16 - S16
    ('absdiff-240p-s16', 'org.khronos.openvx.absdiff image:426,240,S016 image:426,240,S016 image:426,240,S016'),
    ('absdiff-360p-s16', 'org.khronos.openvx.absdiff image:640,360,S016 image:640,360,S016 image:640,360,S016'),
    ('absdiff-480p-s16', 'org.khronos.openvx.absdiff image:854,480,S016 image:854,480,S016 image:854,480,S016'),
    ('absdiff-720p-s16', 'org.khronos.openvx.absdiff image:1280,720,S016 image:1280,720,S016 image:1280,720,S016'),
    ('absdiff-1080p-s16', 'org.khronos.openvx.absdiff image:1920,1080,S016 image:1920,1080,S016 image:1920,1080,S016'),
    ('absdiff-1440p-s16', 'org.khronos.openvx.absdiff image:2560,1440,S016 image:2560,1440,S016 image:2560,1440,S016'),
    ('absdiff-2160p-s16', 'org.khronos.openvx.absdiff image:3840,2160,S016 image:3840,2160,S016 image:3840,2160,S016')
]


# check arguments
if hardwareMode not in ('CPU', 'GPU'):
    print("ERROR: OpenVX Hardware supported - CPU or GPU]")
    exit()
if listTest not in ('no', 'yes'):
    print(
        "ERROR: List Vision Performance Tests options supported - [no or yes]")
    exit()
if functionalityTests not in ('no', 'yes'):
    print("ERROR: Vision functionality Tests option supported - [no or yes]")
    exit()
if backendType not in ('CPU', 'HIP', 'OCL'):
    print("ERROR: OpenVX Backends supported - CPU or HIP or OCL]")
    exit()
if profilingOption not in ('no', 'yes'):
    print("ERROR: Profiling options supported - [no or yes]")
    exit()

if perfCounters not in ('no', 'yes'):
    print("ERROR: perf_counters options supported - [no or yes]")
    exit()

if profilingOption == "no" and perfCounters == "yes":
    print("ERROR: To collect Performance counters both profiling and perfCounters must be yes")
    exit()

if not 0 <= testFilter <= len(openvxNodes):
    print(
        "\nERROR: Vision Performance Filter not in range - [1 - %d]\n" % (len(openvxNodes)))
    exit()
if not 1 <= numFrames <= 10000:
    print("\nERROR: Vision Test Number of Frames not in range - [1 - 10000]\n")
    exit()

if not 1 <= width <= 7680:
    print("\nERROR: image width not in range - [1 - 7680]\n")
    exit()

if not 1 <= height <= 7680:
    print("\nERROR: image width not in range - [1 - 7680]\n")
    exit()

if hardwareMode == "CPU":
    if backendType != "CPU" or profilingOption != "no":
        print("For hardware_mode=CPU, the backend_type must be 'CPU' and profiling must be 'no'")
        exit()
if hardwareMode == "GPU":
    if backendType == "CPU":
        print("For hardware_mode=GPU, the backend_type must be either 'HIP' or 'OCL'")
        exit()

backendTypeValue = 0
if backendType == "OCL":
    backendTypeValue = 0
elif backendType == "HIP":
    backendTypeValue = 1

# List Vision Functionality tests
if listTest == 'yes':
    print(" %-5s - %-30s\n" % ('Test ID', 'Test Name'))
    for i in range(len(openvxNodes)):
        nodeName, nodeFormat = openvxNodes[i]
        print("   %-5d - %-30s\n" % ((i+1), nodeName))
    exit()
if runvxDir == '':
    print("\nERROR: RunVX Executable Directory Required\n")
    print("USAGE: python runVisionTests.py --help\n")
    exit()

print("\nMIVisionX runVisionTests V-"+__version__+"\n")

# RunVX Application
runVX_exe = runvxDir+'/runvx'
RunVXapp = os.path.abspath(runVX_exe)
scriptPath = os.path.dirname(os.path.realpath(__file__))
if(os.path.isfile(RunVXapp)):
    print("STATUS: RunVX path - "+RunVXapp)
else:
    print("\nERROR: RunVX Executable Not Found\n")
    exit()

# Get cwd
cwd = os.getcwd()

# OpenVX Vision Functionality Tests

if testFilter == 0 and functionalityTests == 'yes':
    # create directory to store vision accurarcy test results
    os.system('(cd '+scriptPath+'/gdfs; mkdir -p openvx_test_results)')
    print("\nrunVisionTests - OpenVX Vision Functionality Tests\n")
    for i in range(len(visionTestConfig)):
        testFileName = visionTestConfig[i]
        print("Running Test Script: "+testFileName)
        os.system('echo '+testFileName)
        os.system(RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
                  hardwareMode+' -dump-profile file '+scriptPath+'/gdfs/'+testFileName+' | tee -a '+scriptPath+'/gdfs/openvx_test_results/VisionOutput.log')
        print("\n")
    print("\nSTATUS: Vision Accuracy Results - " +
          scriptPath+"/gdfs/openvx_test_results\n")

# OpenVX Performance Tests
print("\nrunVisionTests - OpenVX Node Performance\n")
outputDirectory = 'openvx_node_results'
if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)
else:
    shutil.rmtree(outputDirectory)
    os.makedirs(outputDirectory)

# Option A - All cases / single case with GPU profiling

nodeList = []
case_num_list = []


def multiCaseProfilerOCL(nodeList, case_num_list):

    for i in nodeList:
        nodeName, nodeFormat = openvxNodes[i]
        echo1 = 'Running OpenVX Node - '+nodeName
        os.system('echo '+echo1 +
                  ' | tee -a openvx_node_results/nodePerformanceOutput.log')
        os.system('mkdir '+cwd+'/rocprof_vision_tests_outputs/case_'+str(i+1))
        if perfCounters == "yes":
            print('rocprof -i rocprof_counters.txt -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
            os.system('rocprof -i rocprof_counters.txt -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
        else:
            print('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
            os.system('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)

        print("\n")

    RESULTS_DIR = "rocprof_vision_tests_outputs"
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE = RESULTS_DIR + "/consolidated_results.stats.csv"
    new_file = open(CONSOLIDATED_FILE, 'w')
    new_file.write(
        '"OCL Kernel Name","Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

    for case_num in case_num_list:
        nodeName, nodeFormat = openvxNodes[case_num-1]
        CASE_RESULTS_DIR = RESULTS_DIR + "/case_" + str(case_num)
        print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)
        CASE_FILE_PATH = CASE_RESULTS_DIR + \
            "/output_case_" + str(case_num) + ".stats.csv"
        print("CASE_FILE_PATH = " + CASE_FILE_PATH)
        try:
            case_file = open(CASE_FILE_PATH, 'r')
            for line in case_file:
                print(line)
                if line.startswith('"OpenVX_kernel'):
                    new_file.write('Ocl_'+nodeName+","+line)
            case_file.close()
        except IOError:
            print("Unable to open case results")
            continue

    new_file.close()
    os.system('chown $USER:$USER '+RESULTS_DIR +
              '/consolidated_results.stats.csv')

    try:
        import pandas as pd
        pd.options.display.max_rows = None
        df = pd.read_csv(CONSOLIDATED_FILE)
        df["AverageMs"] = df["AverageNs"] / 1000000
        dfPrint = df.drop(['Name', 'Percentage'], axis=1)
        dfPrint["OCL Kernel Name"] = dfPrint.iloc[:, 0].str.lstrip("Ocl_")
        print(dfPrint)
    except ImportError:
        print("\nPandas not available! Results of GPU profiling experiment are available in " + CONSOLIDATED_FILE)
    except IOError:
        print("Unable to open results in " + CONSOLIDATED_FILE)


def multiCaseProfilerHIP(nodeList, case_num_list):

    for i in nodeList:
        nodeName, nodeFormat = openvxNodes[i]
        echo1 = 'Running OpenVX Node - '+nodeName
        os.system('echo '+echo1 +
                  ' | tee -a openvx_node_results/nodePerformanceOutput.log')
        os.system('mkdir '+cwd+'/rocprof_vision_tests_outputs/case_'+str(i+1))
        if perfCounters == "yes":
            print('rocprof -i rocprof_counters.txt -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats  '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
            os.system('rocprof -i rocprof_counters.txt -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats  '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
        else:
            print('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats  '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
            os.system('rocprof -o "rocprof_vision_tests_outputs/case_'+str(i+1)+'/output_case_'+str(i+1)+'.csv" --basenames on --timestamp on --stats  '+RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
                hardwareMode+' -dump-profile node '+nodeFormat)
        print("\n")

    RESULTS_DIR = "rocprof_vision_tests_outputs"
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE = RESULTS_DIR + "/consolidated_results.stats.csv"
    new_file = open(CONSOLIDATED_FILE, 'w')
    new_file.write(
        '"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')

    for case_num in case_num_list:
        nodeName, nodeFormat = openvxNodes[case_num-1]
        CASE_RESULTS_DIR = RESULTS_DIR + "/case_" + str(case_num)
        print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)
        CASE_FILE_PATH = CASE_RESULTS_DIR + \
            "/output_case_" + str(case_num) + ".stats.csv"
        print("CASE_FILE_PATH = " + CASE_FILE_PATH)
        try:
            case_file = open(CASE_FILE_PATH, 'r')
            for line in case_file:
                print(line)
                if line.startswith('"Hip'):
                    new_file.write(line)
            case_file.close()
        except IOError:
            print("Unable to open case results")
            continue

    new_file.close()
    os.system('chown $USER:$USER '+RESULTS_DIR +
              '/consolidated_results.stats.csv')

    try:
        import pandas as pd
        pd.options.display.max_rows = None
        df = pd.read_csv(CONSOLIDATED_FILE)
        df["AverageMs"] = df["AverageNs"] / 1000000
        dfPrint = df.drop(['Percentage'], axis=1)
        dfPrint["HIP Kernel Name"] = dfPrint.iloc[:, 0].str.lstrip("Hip_")
        print(dfPrint)
    except ImportError:
        print("\nPandas not available! Results of GPU profiling experiment are available in " + CONSOLIDATED_FILE)
    except IOError:
        print("Unable to open results in " + CONSOLIDATED_FILE)


if testFilter == 0:
    totalCount = len(openvxNodes[:])
    nodeList = range(totalCount)
    case_num_list = range(1, totalCount + 1, 1)
else:
    totalCount = testFilter
    nodeList = range((totalCount - 1), totalCount, 1)
    case_num_list = range(totalCount, totalCount + 1, 1)

if profilingOption == "yes":

    os.system('rm -rvf '+cwd+'/rocprof_vision_tests_outputs')
    os.system('mkdir '+cwd+'/rocprof_vision_tests_outputs')

    if backendType == "OCL":
        multiCaseProfilerOCL(nodeList=nodeList, case_num_list=case_num_list)
    else:
        multiCaseProfilerHIP(nodeList=nodeList, case_num_list=case_num_list)

# Option B - All cases / single case without GPU profiling
else:
    for i in nodeList:
        nodeName, nodeFormat = openvxNodes[i]
        echo1 = 'Running OpenVX Node - '+nodeName
        os.system('echo '+echo1 +
                  ' | tee -a openvx_node_results/nodePerformanceOutput.log')
        print(RunVXapp+' -frames:'+str(numFrames)+' -affinity:' +
              hardwareMode+' -dump-profile node '+nodeFormat)
        os.system(RunVXapp+' -frames:'+str(numFrames)+' -affinity:'+hardwareMode +
                  ' -dump-profile node '+nodeFormat+' | tee -a openvx_node_results/nodePerformanceOutput.log')
        print("\n")
    orig_stdout = sys.stdout
    sys.stdout = open('openvx_node_results/nodePerformance.md', 'a')
    echo_1 = '| OpenVX Node                             |Frames Count| tmp (ms) | avg (ms) | min (ms) | max (ms) |'
    print(echo_1)
    echo_2 = '|-----------------------------------------|------------|----------|----------|----------|----------|'
    print(echo_2)
    sys.stdout = orig_stdout
    print(echo_1)
    print(echo_2)
    if hardwareMode == 'CPU':
        runAwk_csv = r'''awk 'BEGIN { node = "xxx"; } /Running OpenVX Node - / { node = $5; } /CPU,GRAPH/ { printf("| %-39s | %10d | %8.3f | %8.3f | %8.3f | %8.3f |\n", node, $1, $2, $3, $4, $5) }' openvx_node_results/nodePerformanceOutput.log | tee -a openvx_node_results/nodePerformance.md'''
    elif hardwareMode == 'GPU':
        runAwk_csv = r'''awk 'BEGIN { node = "xxx"; } /Running OpenVX Node - / { node = $5; } /GPU,GRAPH/ { printf("| %-39s | %10d | %8.3f | %8.3f | %8.3f | %8.3f |\n", node, $1, $2, $3, $4, $5) }' openvx_node_results/nodePerformanceOutput.log | tee -a openvx_node_results/nodePerformance.md'''
    os.system(runAwk_csv)

    # get data
    platform_name = platform.platform()
    if os.path.exists('/usr/bin/yum'):
        if not "centos" in platform_name or not "redhat" in platform_name:
            platfromInfo = platform_name+'-CentOS-RedHat'
    elif os.path.exists('/usr/bin/apt-get'):
        if not "Ubuntu" in platform_name:
            platform_name = platform_name+'-Ubuntu'
    elif os.path.exists('/usr/bin/zypper'):
        if not "SLES" in platform_name:
            platform_name = platform_name+'-SLES'
    else:
        print("\nMIVisionX Vision Test on "+platform_name+" is unsupported")
        print("MIVisionX Vision Test Supported on: Ubuntu 20/22; CentOS 7/8; RedHat 8/9; & SLES 15 SP3")
        print("\nMIVisionX Vision Test on "+platform_name+" is unreliable")

    platform_name_fq = shell('hostname --all-fqdns')
    platform_ip = shell('hostname -I')[0:-1]  # extra trailing space

    file_dtstr = datetime.now().strftime("%Y%m%d")
    reportFilename = 'vision_report_%s_%s_%s.md' % (
        platform_name, file_dtstr, backendType)
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
        f.write("MIVisionX - OpenVX Function Report\n")
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
        f.write("### Hardware: %s\n" % hardwareMode)
        f.write("\n")
        with open('openvx_node_results/nodePerformance.md') as benchmarkFile:
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
            "\n\n---\n**Copyright AMD ROCm MIVisionX 2018 - 2024 -- runVisionTests.py V-"+__version__+"**\n")
        f.write("\n")

        # report file
        reportFileDir = os.path.abspath(reportFilename)
        print("\nSTATUS: Output Report File - "+reportFileDir)

print("\nrunVisionTests.py completed - V:"+__version__+"\n")
