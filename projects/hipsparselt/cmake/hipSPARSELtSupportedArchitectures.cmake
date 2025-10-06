# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Base architectures - used when "all" is specified for GPU_TARGETS
set(BASE_ARCHITECTURES "")

# All supported architectures including xnack variants - used for validation of GPU_TARGETS
set(SUPPORTED_ARCHITECTURES "")

if(NOT BUILD_ADDRESS_SANITIZER)
    list(APPEND BASE_ARCHITECTURES
        "gfx942"
        "gfx950")

    set(SUPPORTED_ARCHITECTURES ${BASE_ARCHITECTURES})
    list(APPEND SUPPORTED_ARCHITECTURES
        "gfx942:xnack+"
        "gfx942:xnack-"
        "gfx950:xnack+"
        "gfx950:xnack-")
else()
    # For address sanitizer builds, base and supported are the same
    list(APPEND BASE_ARCHITECTURES
        "gfx942:xnack+"
        "gfx950:xnack+")
    set(SUPPORTED_ARCHITECTURES ${BASE_ARCHITECTURES})
endif()

function(hipsparselt_validate_gpu_targets targets)
    set(supported_list ${SUPPORTED_ARCHITECTURES})
    set(target_list ${targets})

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
endfunction()

function(hipsparselt_get_base_architectures output_var)
    set(${output_var} ${BASE_ARCHITECTURES} PARENT_SCOPE)
endfunction()

function(hipsparselt_get_supported_architectures output_var)
    set(${output_var} ${SUPPORTED_ARCHITECTURES} PARENT_SCOPE)
endfunction()
