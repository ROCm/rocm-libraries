# MIT License
#
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

set(CMAKE_CXX_STANDARD 17)

include(CheckCXXSourceCompiles)

set(HAVE_STD_FILESYSTEM_TEST [[
  #include <filesystem>

  int main()
  {
  std::filesystem::path p{"/"};
  return 0;
  }
  ]])
check_cxx_source_compiles("${HAVE_STD_FILESYSTEM_TEST}" HAVE_STD_FILESYSTEM)

add_custom_target(tools)

function(rocrand_add_tool TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    add_dependencies(tools ${TARGET_NAME})
    if(NOT HAVE_STD_FILESYSTEM)
      # if filesystem library is not available in the standard library,
      # link to the separate library instead
      target_link_options( ${TARGET_NAME} PRIVATE -lstdc++fs )
    endif()
endfunction()

rocrand_add_tool(bin2typed bin2typed.cpp)
if(HIP_COMPILER STREQUAL "nvcc")
    message(STATUS "Target lfsr113_precomputed_generator cannot be built for CUDA")
else()
    rocrand_add_tool(lfsr113_precomputed_generator lfsr113_precomputed_generator.cpp)
    target_link_libraries(lfsr113_precomputed_generator hip::device)
endif()
rocrand_add_tool(mrg31k3p_precomputed_generator mrg31k3p_precomputed_generator.cpp)
rocrand_add_tool(mrg32k3a_precomputed_generator mrg32k3a_precomputed_generator.cpp)
rocrand_add_tool(mt19937_precomputed_generator mt19937_precomputed_generator.cpp)
rocrand_add_tool(scrambled_sobol32_constants_generator scrambled_sobol32_constants_generator.cpp)
rocrand_add_tool(scrambled_sobol32_direction_vector_generator
    scrambled_sobol32_direction_vector_generator.cpp)
rocrand_add_tool(scrambled_sobol64_constants_generator scrambled_sobol64_constants_generator.cpp)
rocrand_add_tool(scrambled_sobol64_direction_vector_generator
    scrambled_sobol64_direction_vector_generator.cpp)
rocrand_add_tool(sobol32_direction_vector_generator sobol32_direction_vector_generator.cpp)
rocrand_add_tool(sobol64_direction_vector_generator sobol64_direction_vector_generator.cpp)
rocrand_add_tool(xorwow_precomputed_generator xorwow_precomputed_generator.cpp)

# Run all (scrabmbled_)sobol(32,64) precomputations and generate all output formats
# (binary files, cpp files, assembly files)
# reads the input direction numbers from a file named direction_numbers.txt in the current
# working directory
add_custom_target(precompute_sobol_directions)

foreach(NAME IN ITEMS "sobol32" "sobol64")
    foreach(PREFIX IN ITEMS "" "scrambled_")
        string(TOUPPER "${PREFIX}" PREFIX_UPPER)
        set(INNER_NAME "${PREFIX}${NAME}")

        set(${PREFIX_UPPER}DIRECTIONS_TARGET precompute_${INNER_NAME}_directions)
        set(${PREFIX_UPPER}DIRECTIONS_EXE ${INNER_NAME}_direction_vector_generator)
        set(${PREFIX_UPPER}SYMBOL rocrand_h_${INNER_NAME}_direction_vectors)
        set(${PREFIX_UPPER}OUT_H
            ${PROJECT_SOURCE_DIR}/library/include/rocrand/rocrand_${INNER_NAME}_precomputed.h)
        set(${PREFIX_UPPER}OUT_BIN_NAME rocrand_${INNER_NAME}_precomputed.bin)
        set(${PREFIX_UPPER}OUT_BIN
            ${PROJECT_SOURCE_DIR}/library/src/${${PREFIX_UPPER}OUT_BIN_NAME})
        set(${PREFIX_UPPER}NAME ${INNER_NAME})
    endforeach()

    add_custom_target(${DIRECTIONS_TARGET} ${DIRECTIONS_EXE}
        ${CMAKE_CURRENT_SOURCE_DIR}/direction_numbers.txt
        ${SYMBOL}
        ${OUT_H}
        ${OUT_BIN})
    add_dependencies(${DIRECTIONS_TARGET} ${DIRECTIONS_EXE})

    add_custom_target(${SCRAMBLED_DIRECTIONS_TARGET} ${SCRAMBLED_DIRECTIONS_EXE}
        ${OUT_BIN}
        ${SCRAMBLED_SYMBOL}
        ${SCRAMBLED_OUT_H}
        ${SCRAMBLED_OUT_BIN})
    add_dependencies(${SCRAMBLED_DIRECTIONS_TARGET}
        ${SCRAMBLED_DIRECTIONS_EXE} ${DIRECTIONS_TARGET})

    if(NAME STREQUAL sobol32)
        set(SYMBOL_TYPE "unsigned int")
    else()
        set(SYMBOL_TYPE "unsigned long long")
    endif()

    foreach(PREFIX IN ITEMS "" "scrambled_")
        string(TOUPPER "${PREFIX}" PREFIX_UPPER)
        set(INNER_NAME "${PREFIX}${NAME}")

        add_custom_target(precompute_${INNER_NAME}_directions_asm ${CMAKE_COMMAND}
            -DSYMBOL=${${PREFIX_UPPER}SYMBOL}
            -DINPUT_BIN=${${PREFIX_UPPER}OUT_BIN_NAME}
            -DOUTPUT=${PROJECT_SOURCE_DIR}/library/src/rocrand_${INNER_NAME}_precomputed.S
            -P ${CMAKE_CURRENT_LIST_DIR}/AsmEmbedSymbol.cmake)

        add_custom_target(precompute_${INNER_NAME}_directions_cpp bin2typed
            ${${PREFIX_UPPER}OUT_BIN}
            ${${PREFIX_UPPER}SYMBOL}
            ${SYMBOL_TYPE}
            ${PROJECT_SOURCE_DIR}/library/src/rocrand_${INNER_NAME}_precomputed.cpp)
        add_dependencies(precompute_${INNER_NAME}_directions_cpp
                         bin2typed ${${PREFIX_UPPER}DIRECTIONS_TARGET})
    endforeach()

    add_dependencies(precompute_sobol_directions
                     precompute_${NAME}_directions
                     precompute_${SCRAMBLED_NAME}_directions
                     precompute_${NAME}_directions_asm
                     precompute_${NAME}_directions_cpp
                     precompute_${SCRAMBLED_NAME}_directions_asm
                     precompute_${SCRAMBLED_NAME}_directions_cpp)
endforeach()
