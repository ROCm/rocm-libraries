# MIT License
#
# Copyright (c) 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
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

include_guard(GLOBAL)

macro(miopen_workaround_amdhip64_missing TARGET_NAME)
    if(WIN32 AND TARGET hip::amdhip64)
        set(_miopen_hip_compat_dir "${CMAKE_CURRENT_BINARY_DIR}/hip_compat")
        add_custom_command(TARGET ${TARGET_NAME} PRE_LINK
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_miopen_hip_compat_dir}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_LINKER_FILE:hip::amdhip64>" "${_miopen_hip_compat_dir}/amdhip64.lib"
            VERBATIM
        )
        target_link_directories(${TARGET_NAME} PRIVATE "${_miopen_hip_compat_dir}")
    endif()
endmacro()
