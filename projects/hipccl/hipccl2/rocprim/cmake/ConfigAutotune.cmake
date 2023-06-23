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

function(add_configured_source)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "INPUT;TARGET;OUTPUT_PATTERN" "NAMES;VALUES")
  list(LENGTH ARG_NAMES NAMES_LEN)
  list(LENGTH ARG_VALUES VALS_LEN)
  if (NOT NAMES_LEN EQUAL VALS_LEN)
    message("NAMES_LEN: ${NAMES_LEN}, VALS_LEN: ${VALS_LEN}")
    message(FATAL_ERROR "The same number of names and values must be provided!")
  endif()

  math(EXPR max "${VALS_LEN} - 1")
  foreach(i RANGE ${max})
    list(GET ARG_NAMES ${i} curr_name)
    list(GET ARG_VALUES ${i} "${curr_name}")
  endforeach()

  string(CONFIGURE "${ARG_OUTPUT_PATTERN}" output @ONLY)
  string(MAKE_C_IDENTIFIER ${output} output)
  configure_file("${ARG_INPUT}" "${ARG_TARGET}.parallel/${output}.cpp" @ONLY)
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${ARG_TARGET}.parallel")
  target_sources("${ARG_TARGET}" PRIVATE "${ARG_TARGET}.parallel/${output}.cpp")
  target_include_directories("${ARG_TARGET}" PRIVATE "../benchmark")

  # Rerun configuration if the input template changes or if the configured file is cleaned
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARG_INPUT}" "${ARG_TARGET}.parallel/${output}.cpp")
endfunction()

function(div_round_up dividend divisor result_var)
  math(EXPR result "(${dividend} + ${divisor} - 1) / ${divisor}")
  set("${result_var}" "${result}" PARENT_SCOPE)
endfunction()

function(add_matrix)
  set(single_value_args "TARGET" "INPUT" "OUTPUT_PATTERN" "SHARDS" "CURRENT_SHARD")
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${single_value_args}" "NAMES;LISTS")
  list(LENGTH ARG_NAMES NAMES_LEN)
  list(LENGTH ARG_LISTS LISTS_LEN)
  if (NOT NAMES_LEN EQUAL LISTS_LEN)
    message("NAMES_LEN: ${NAMES_LEN}, LISTS_LEN: ${LISTS_LEN}")
    message(FATAL_ERROR "The same number of names and lists must be provided!")
  endif()

  # Calculate the total number of permutations
  set(total_len 1)
  foreach(LIST IN LISTS ARG_LISTS)
    string(REPLACE " " ";" list ${LIST})
    list(LENGTH list LIST_LEN)
    math(EXPR total_len "${total_len} * ${LIST_LEN}")
  endforeach()

  if(NOT DEFINED ARG_SHARDS)
    set(ARG_SHARDS 1)
  endif()
  div_round_up("${total_len}" "${ARG_SHARDS}" per_shard)
  message("per_shard: ${per_shard}")
  math(EXPR start "${ARG_CURRENT_SHARD} * ${per_shard}")
  math(EXPR stop "${start} + ${per_shard} - 1")
  message("start: ${start}, stop: ${stop}")
  # Run for each permutation of input paramters
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

    add_configured_source(TARGET "${ARG_TARGET}"
            INPUT "${ARG_INPUT}"
            OUTPUT_PATTERN "${ARG_OUTPUT_PATTERN}"
            NAMES ${ARG_NAMES}
            VALUES ${values})
  endforeach()
endfunction()

# example of a FILTER rule
function(reject_odd_blocksize RESULT BlockSize)
  math(EXPR res "${BlockSize} % 2")
  if(res EQUAL 0)
    set("${RESULT}" ON PARENT_SCOPE)
  else()
    set("${RESULT}" OFF PARENT_SCOPE)
  endif()
endfunction()
