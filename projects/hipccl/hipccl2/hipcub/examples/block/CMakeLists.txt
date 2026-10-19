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

file(GLOB_RECURSE example_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  example_*.cu
)

foreach (cub_target IN LISTS CUB_TARGETS)
  foreach (example_src IN LISTS example_srcs)
    get_filename_component(example_name "${example_src}" NAME_WE)
    string(REGEX REPLACE
      "^example_block_" "block."
      example_name "${example_name}"
    )
    add_hipcub_example(target_name ${example_name} "${example_src}" ${cub_target})
  endforeach()
endforeach()
