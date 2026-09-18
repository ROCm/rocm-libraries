# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Filter GPU offload targets so they are not passed into Tensile, while leaving
# GPU_TARGETS unchanged for rocBLAS library compilation (--offload-arch).
#
# A skiplist entry matches an exact target or that target with a ':' feature
# ID suffix (for example gfx1250 matches gfx1250:xnack+).

function(rocblas_tensile_arch_is_skipped arch_ skiplist_list_name_ out_var_)
  set(_hit FALSE)
  foreach(_pat IN LISTS ${skiplist_list_name_})
    string(STRIP "${_pat}" _pat)
    if(_pat STREQUAL "")
      continue()
    endif()
    if(arch_ STREQUAL _pat)
      set(_hit TRUE)
      break()
    endif()
    string(LENGTH "${_pat}" _pat_len)
    string(SUBSTRING "${arch_}" 0 ${_pat_len} _prefix)
    if(_prefix STREQUAL _pat)
      string(LENGTH "${arch_}" _arch_len)
      if(_arch_len GREATER _pat_len)
        string(SUBSTRING "${arch_}" ${_pat_len} 1 _sep)
        if(_sep STREQUAL ":")
          set(_hit TRUE)
          break()
        endif()
      endif()
    endif()
  endforeach()
  set(${out_var_} ${_hit} PARENT_SCOPE)
endfunction()

function(rocblas_filter_tensile_arch_skiplist input_list_name_ skiplist_list_name_ kept_var_ excluded_var_)
  set(_kept "")
  set(_excluded "")
  foreach(_arch IN LISTS ${input_list_name_})
    string(STRIP "${_arch}" _arch)
    if(_arch STREQUAL "")
      continue()
    endif()
    rocblas_tensile_arch_is_skipped("${_arch}" ${skiplist_list_name_} _skipped)
    if(_skipped)
      list(APPEND _excluded "${_arch}")
    else()
      list(APPEND _kept "${_arch}")
    endif()
  endforeach()
  set(${kept_var_} "${_kept}" PARENT_SCOPE)
  set(${excluded_var_} "${_excluded}" PARENT_SCOPE)
endfunction()

# Build a C initializer fragment: "gfx1250","gfx1201",
# Used as -DROCBLAS_TENSILE_ARCH_SKIPLIST_C_LITERALS=...
function(rocblas_tensile_arch_skiplist_c_literals skiplist_list_name_ out_var_)
  set(_lits "")
  foreach(_pat IN LISTS ${skiplist_list_name_})
    string(STRIP "${_pat}" _pat)
    if(NOT _pat STREQUAL "")
      string(APPEND _lits "\"${_pat}\",")
    endif()
  endforeach()
  set(${out_var_} "${_lits}" PARENT_SCOPE)
endfunction()

# Tensile placeholder architecture (architectureMap gfx000 -> none). It exists when
# the selected logic tree has HIP fallback YAML (ArchitectureName fallback or gfx000).
function(rocblas_tensile_find_stub_gfx_target logic_dir_ out_var_)
  set(_stub "")
  if(IS_DIRECTORY "${logic_dir_}")
    file(GLOB _candidates
      LIST_DIRECTORIES false
      "${logic_dir_}/hip/*.yaml"
      "${logic_dir_}/hip/*.yml"
      "${logic_dir_}/hip_*.yaml"
      "${logic_dir_}/hip_*.yml")
    if(_candidates)
      set(_stub "gfx000")
    endif()
  endif()
  set(${out_var_} "${_stub}" PARENT_SCOPE)
endfunction()

# When every GPU_TARGET is skipped, keep Tensile enabled and build the stub
# gfx target if the logic tree has one; otherwise fail configure.
function(rocblas_tensile_use_stub_if_empty kept_var_ logic_dir_ gpu_targets_ skiplist_)
  if(NOT "${${kept_var_}}" STREQUAL "")
    return()
  endif()
  rocblas_tensile_find_stub_gfx_target("${logic_dir_}" _stub)
  if(_stub)
    message(STATUS
      "All GPU_TARGETS (${gpu_targets_}) are skipped by ROCBLAS_TENSILE_ARCH_SKIPLIST "
      "(${skiplist_}); using Tensile stub architecture ${_stub}")
    set(${kept_var_} "${_stub}" PARENT_SCOPE)
  else()
    message(FATAL_ERROR
      "All GPU_TARGETS (${gpu_targets_}) are skipped by ROCBLAS_TENSILE_ARCH_SKIPLIST "
      "(${skiplist_}) and no Tensile stub gfx target was found in ${logic_dir_}.")
  endif()
endfunction()
