# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# hipSPARSE Supported GPU Architectures
# ==============================================================================
# This file defines the GPU architectures supported by hipSPARSE.
#
# Architecture support is managed through version control (Git branches/tags)
# rather than build-time conditionals. Each branch/tag in version control
# defines the appropriate architecture list for that ROCm release.
#
# ASAN builds use a restricted set of architectures with xnack+ support.
# ==============================================================================
if(NOT HIPSPARSE_ENABLE_ASAN)
  # Standard build: all supported GPU architectures
  set(SUPPORTED_TARGETS
    "gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+"
    "gfx940;gfx941;gfx942;gfx950"
    "gfx1030;gfx1100;gfx1101;gfx1102;gfx1103"
    "gfx1150;gfx1151;gfx1200;gfx1201"
  )
else()
  # Address sanitizer builds: architectures with xnack+ support
  set(SUPPORTED_TARGETS
    "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"
  )
endif()

function(hipsparse_validate_gpu_targets targets output_var)
  # Expand gfx90a to xnack variants for backward compatibility
  set(target_list_expanded "")
  foreach(target IN LISTS targets)
    if(target STREQUAL "gfx90a")
      if(HIPSPARSE_ENABLE_ASAN)
        # ASAN builds only support xnack+
        list(APPEND target_list_expanded "gfx90a:xnack+")
      else()
        # Standard builds support both xnack variants
        list(APPEND target_list_expanded "gfx90a:xnack-")
        list(APPEND target_list_expanded "gfx90a:xnack+")
      endif()
    else()
      list(APPEND target_list_expanded "${target}")
    endif()
  endforeach()

  set(supported_list ${SUPPORTED_TARGETS})
  set(target_list ${target_list_expanded})

  string(REGEX REPLACE ";" " " supported_flat "${supported_list}")
  string(REGEX REPLACE " +" ";" supported_list "${supported_flat}")

  string(REGEX REPLACE ";" " " target_flat "${target_list}")
  string(REGEX REPLACE " +" ";" target_list "${target_flat}")

  foreach(target IN LISTS target_list)
    list(FIND supported_list "${target}" idx)
    if(idx EQUAL -1)
      message(FATAL_ERROR "Unsupported GPU target: ${target}\nSupported targets are: ${supported_list}")
    endif()
  endforeach()

  # Return the expanded target list
  set(${output_var} ${target_list} PARENT_SCOPE)
endfunction()

function(hipsparse_get_base_architectures output_var)
  set(${output_var} ${SUPPORTED_TARGETS} PARENT_SCOPE)
endfunction()

function(hipsparse_get_supported_architectures output_var)
  set(${output_var} ${SUPPORTED_TARGETS} PARENT_SCOPE)
endfunction()
