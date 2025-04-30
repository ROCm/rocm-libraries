# ########################################################################
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################


# Enables increasingly expensive runtime correctness checks
# 0 - Nothing
# 1 - Inexpensive correctness checks (extra assertions, etc..)
#     Note: Some checks are added by the optimizer, so it can help to build
#           with optimizations enabled. e.g. -Og
# 2 - Expensive correctness checks (debug iterators)
macro(add_armor_flags target level)
  if(UNIX AND "${level}" GREATER "0")
    if("${level}" GREATER "1")
      # Building with std debug iterators is enabled by the defines below, but
      # requires building C++ dependencies with the same defines.
      target_compile_definitions(${target} PRIVATE
        _GLIBCXX_DEBUG
      )
    endif()
    target_compile_definitions(${target} PRIVATE
      $<$<NOT:$<BOOL:${BUILD_ADDRESS_SANITIZER}>>:_FORTIFY_SOURCE=1> # requires optimizations to work
      _GLIBCXX_ASSERTIONS
    )
  endif()
endmacro()
