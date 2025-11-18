# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Base architectures - used when "all" is specified for GPU_TARGETS
set(BASE_ARCHITECTURES "")

# All supported architectures including xnack variants - used for validation of GPU_TARGETS
set(SUPPORTED_ARCHITECTURES "")

if(ROCSPARSE_ENABLE_ASAN OR BUILD_ADDRESS_SANITIZER)
    # For address sanitizer builds, base and supported are the same
    list(APPEND BASE_ARCHITECTURES "gfx908:xnack+" "gfx90a:xnack+" "gfx942:xnack+")
    set(SUPPORTED_ARCHITECTURES ${BASE_ARCHITECTURES})
else()
    list(APPEND BASE_ARCHITECTURES
        "gfx803"
        "gfx900:xnack-"
        "gfx906:xnack-"
        "gfx908:xnack-"
        "gfx90a:xnack-"
        "gfx90a:xnack+"
        "gfx942"
        "gfx950"
        "gfx1030"
        "gfx1100"
        "gfx1101"
        "gfx1102"
        "gfx1150"
        "gfx1151"
        "gfx1152"
        "gfx1153"
        "gfx1200"
        "gfx1201"
    )

    set(SUPPORTED_ARCHITECTURES ${BASE_ARCHITECTURES})
    # Add additional xnack variants for validation
    list(APPEND SUPPORTED_ARCHITECTURES
        "gfx900:xnack+"
        "gfx906:xnack+"
        "gfx908:xnack+"
        "gfx942:xnack+"
        "gfx942:xnack-"
        "gfx950:xnack+"
        "gfx950:xnack-"
    )
endif()

# .rst: Validates that all specified GPU targets are supported.
#
# ``rocsparse_validate_gpu_targets(<targets>)``
#
# Checks each target in the list against supported architectures. Throws FATAL_ERROR if any
# unsupported target is found.
function(rocsparse_validate_gpu_targets targets)
    set(supported_list ${SUPPORTED_ARCHITECTURES})
    set(target_list ${targets})

    string(REGEX REPLACE ";" " " supported_flat "${supported_list}")
    string(REGEX REPLACE " +" ";" supported_list "${supported_flat}")

    string(REGEX REPLACE ";" " " target_flat "${target_list}")
    string(REGEX REPLACE " +" ";" target_list "${target_flat}")

    foreach(target IN LISTS target_list)
        list(FIND supported_list "${target}" idx)
        if(idx EQUAL -1)
            message(
                FATAL_ERROR
                    "Unsupported GPU target: ${target}\nSupported targets are: ${supported_list}"
            )
        endif()
    endforeach()
endfunction()

# .rst: Returns the list of base GPU architectures.
#
# ``rocsparse_get_base_architectures(<output_var>)``
#
# Sets <output_var> to the list of base architectures used when "all" is specified.
function(rocsparse_get_base_architectures output_var)
    set(${output_var} ${BASE_ARCHITECTURES} PARENT_SCOPE)
endfunction()

# .rst: Returns the list of all supported GPU architectures.
#
# ``rocsparse_get_supported_architectures(<output_var>)``
#
# Sets <output_var> to the full list including xnack variants.
function(rocsparse_get_supported_architectures output_var)
    set(${output_var} ${SUPPORTED_ARCHITECTURES} PARENT_SCOPE)
endfunction()

