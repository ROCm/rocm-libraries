# MIT License
#
# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

# Function to add a configured source file to a target.
# It parses arguments, prepares the output file name, and configures the file.
function(add_configured_source)
  # Parse arguments and ensure proper usage
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "INPUT;TARGET;OUTPUT_PATTERN" "NAMES;VALUES")
  list(LENGTH ARG_NAMES NAMES_LEN)
  list(LENGTH ARG_VALUES VALS_LEN)
  if (NOT NAMES_LEN EQUAL VALS_LEN)
    message(FATAL_ERROR "add_configured_source: The same number of names (${NAMES_LEN}) and values (${VALS_LEN}) must be provided!")
  endif()

  # Loop through the names and values, preparing the output pattern
  set(max ${VALS_LEN})
  math(EXPR max "${max} - 1")
  foreach(i RANGE ${max})
    list(GET ARG_NAMES ${i} curr_name)
    list(GET ARG_VALUES ${i} "${curr_name}")
  endforeach()

  # Configure the output file and add it to the target
  string(CONFIGURE "${ARG_OUTPUT_PATTERN}" output @ONLY)
  string(MAKE_C_IDENTIFIER ${output} output)
  set(output_path "${ARG_TARGET}.parallel/${output}.cpp")
  configure_file("${ARG_INPUT}" "${output_path}" @ONLY)
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${ARG_TARGET}.parallel")
  target_sources("${ARG_TARGET}" PRIVATE "${output_path}")
  target_include_directories("${ARG_TARGET}" PRIVATE "../benchmark")

  # Ensure reconfiguration if necessary
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARG_INPUT}" "${output_path}")
endfunction()

# Function to divide two numbers and round up.
function(div_round_up dividend divisor result_var)
  math(EXPR result "(${dividend} + ${divisor} - 1) / ${divisor}")
  set("${result_var}" "${result}" PARENT_SCOPE)
endfunction()

# Function to add a matrix of configured sources.
# It handles permutations of input parameters and calls add_configured_source accordingly.
function(add_matrix)
  set(single_value_args "TARGET" "INPUT" "OUTPUT_PATTERN" "SHARDS" "CURRENT_SHARD")
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${single_value_args}" "NAMES;LISTS")

  # Validate argument lengths
  list(LENGTH ARG_NAMES NAMES_LEN)
  list(LENGTH ARG_LISTS LISTS_LEN)
  if (NOT NAMES_LEN EQUAL LISTS_LEN)
    message(FATAL_ERROR "add_matrix: The same number of names (${NAMES_LEN}) and lists (${LISTS_LEN}) must be provided!")
  endif()

  # Calculate the total number of permutations
  set(total_len 1)
  foreach(LIST IN LISTS ARG_LISTS)
    string(REPLACE " " ";" list ${LIST})
    list(LENGTH list LIST_LEN)
    math(EXPR total_len "${total_len} * ${LIST_LEN}")
  endforeach()

  # Handle sharding
  if(NOT DEFINED ARG_SHARDS)
    set(ARG_SHARDS 1)
  endif()
  div_round_up("${total_len}" "${ARG_SHARDS}" per_shard)

  # Determine the range of permutations for the current shard
  math(EXPR start "${ARG_CURRENT_SHARD} * ${per_shard}")
  math(EXPR stop "${start} + ${per_shard} - 1")

  # Process each permutation
  foreach(i RANGE ${start} ${stop})
    set(index ${i})
    set(values "")
    foreach(input_list IN LISTS ARG_LISTS)
      string(REPLACE " " ";" curr_list ${input_list})
      list(LENGTH curr_list curr_length)
      math(EXPR curr_index "${index} % ${curr_length}")
      list(GET curr_list ${curr_index} curr_item)
      list(APPEND values "${curr_item}")
      math(EXPR index "${index} / ${curr_length}")
    endforeach()

    # Add the configured source for each permutation
    add_configured_source(TARGET "${ARG_TARGET}"
            INPUT "${ARG_INPUT}"
            OUTPUT_PATTERN "${ARG_OUTPUT_PATTERN}"
            NAMES ${ARG_NAMES}
            VALUES ${values})
  endforeach()
endfunction()

# Function to filter out odd block sizes.
# It sets a variable in the parent scope based on the condition.
function(reject_odd_blocksize RESULT BlockSize)
  math(EXPR res "${BlockSize} % 2")
  if(res EQUAL 0)
    set("${RESULT}" ON PARENT_SCOPE)
  else()
    set("${RESULT}" OFF PARENT_SCOPE)
  endif()
endfunction()