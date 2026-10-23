# MIT License
#
# Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

add_hipcub_example("hipcub.example.DevicePartitionFlagged"            example_device_partition_flagged.cpp)
add_hipcub_example("hipcub.example.DevicePartitionIf"                 example_device_partition_if.cpp)
add_hipcub_example("hipcub.example.DeviceRadixSort"                   example_device_radix_sort.cpp)
add_hipcub_example("hipcub.example.DeviceReduce"                      example_device_reduce.cpp)
add_hipcub_example("hipcub.example.DeviceScan"                        example_device_scan.cpp)
add_hipcub_example("hipcub.example.DeviceSelectFlagged"               example_device_select_flagged.cpp)
add_hipcub_example("hipcub.example.DeviceSelectIf"                    example_device_select_if.cpp)
add_hipcub_example("hipcub.example.DeviceSelectUnique"                example_device_select_unique.cpp)
add_hipcub_example("hipcub.example.DeviceSortFindNonTrivalRuns.cpp"   example_device_sort_find_non_trivial_runs.cpp)
