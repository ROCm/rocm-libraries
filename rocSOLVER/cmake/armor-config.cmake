# ########################################################################
# Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
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
    # Note that _FORTIFY_SOURCE does not work unless optimizations are enabled
    target_compile_definitions(${target} PRIVATE
      $<$<NOT:$<BOOL:${BUILD_ADDRESS_SANITIZER}>>:_FORTIFY_SOURCE=1>
      _GLIBCXX_ASSERTIONS
      ROCSOLVER_VERIFY_ASSUMPTIONS
    )
  endif()
endmacro()
