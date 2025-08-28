# Disable unreachable code warnings.
# This test unconditionally throws in some places, the compiler will detect that
# control flow will never reach some instructions. This is intentional.
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(${TEST_TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/wd4702>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/wd4702>
  )
endif()
