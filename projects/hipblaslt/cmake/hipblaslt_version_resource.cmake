# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

# Embeds Windows PE VERSIONINFO into a shipped hipBLASLt-family DLL or EXE through
# the shared rocm_libraries_add_version_resource, stamped with the hipBLASLt product
# name and the git commit the binary was built from. Skips static libraries and
# targets this configuration does not define; on non-Windows platforms it only marks
# the target for rocm_libraries_check_version_resources.
#
#   hipblaslt_add_version_resource(<target> <file-description>)

include("${ROCM_LIBRARIES_ROOT}/cmake/modules/rocm_libraries_version_resource.cmake")

if(WIN32)
    set(HIPBLASLT_VERSION_RESOURCE_COMMIT "unknown")
    if(GIT)
        execute_process(
            COMMAND "${GIT}" log -1 --format=%H .
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE _hipblaslt_version_resource_commit
            RESULT_VARIABLE _hipblaslt_version_resource_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_hipblaslt_version_resource_result EQUAL 0 AND _hipblaslt_version_resource_commit)
            set(HIPBLASLT_VERSION_RESOURCE_COMMIT "${_hipblaslt_version_resource_commit}")
        endif()
    endif()
endif()

function(hipblaslt_add_version_resource TARGET DESCRIPTION)
    if(NOT TARGET ${TARGET})
        return()
    endif()
    get_target_property(_type ${TARGET} TYPE)
    if(NOT _type MATCHES "^(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY)$")
        return()
    endif()
    rocm_libraries_add_version_resource(${TARGET} "${TARGET}" "${DESCRIPTION}"
        PRODUCT_NAME "AMD ROCm hipBLASLt ${hipblaslt_VERSION_MAJOR}.${hipblaslt_VERSION_MINOR} (${hipblaslt_VERSION_TWEAK})"
        COMMENTS "Git commit: ${HIPBLASLT_VERSION_RESOURCE_COMMIT}")
endfunction()
