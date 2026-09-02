################################################################################
#
# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#
################################################################################

# Prefer a project-prefixed analyzer aggregator so the target does not collide
# with sibling projects when MIOpen is included via add_subdirectory in a
# superbuild. The unprefixed `analyze` target is preserved as a back-compat
# alias when MIOpen is built standalone.
if(NOT TARGET miopen-analyze)
    add_custom_target(miopen-analyze COMMENT "Aggregate target for all MIOpen static analyzers")
endif()
if(NOT ROCM_LIBS_SUPERBUILD AND NOT TARGET analyze)
    add_custom_target(analyze DEPENDS miopen-analyze COMMENT "Back-compat alias for miopen-analyze")
endif()

# Register one or more targets as dependencies of the miopen-analyze aggregate target.
function(mark_as_analyzer)
    add_dependencies(miopen-analyze ${ARGN})
endfunction()
