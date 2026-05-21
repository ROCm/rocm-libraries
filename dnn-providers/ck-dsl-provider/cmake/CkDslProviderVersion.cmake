# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Thin wrapper around the shared dnn-providers version helpers so the
# CK DSL provider can read its version.json and emit version.h with
# CK_DSL_PROVIDER_VERSION_STRING.
#
# Usage:
#   include(${CMAKE_CURRENT_LIST_DIR}/cmake/CkDslProviderVersion.cmake)
#   ck_dsl_provider_setup_version()
#   project(ck-dsl-provider VERSION ${CK_DSL_PROVIDER_VERSION} LANGUAGES CXX)
#   ...
#   ck_dsl_provider_generate_version_header(<target>)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/VersionUtils.cmake)

set(CK_DSL_PROVIDER_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/..")

# Read version.json under the provider root and expose
# CK_DSL_PROVIDER_VERSION{,_MAJOR,_MINOR,_PATCH,_TWEAK,_STRING} in the caller's
# scope. Must run before project(...) so VERSION can be passed through.
function(ck_dsl_provider_setup_version)
    dnn_provider_setup_version(ck_dsl_provider "${CK_DSL_PROVIDER_ROOT_DIR}")

    # Re-export to the caller's scope (one frame above this function).
    set(CK_DSL_PROVIDER_VERSION_MAJOR  ${CK_DSL_PROVIDER_VERSION_MAJOR}  PARENT_SCOPE)
    set(CK_DSL_PROVIDER_VERSION_MINOR  ${CK_DSL_PROVIDER_VERSION_MINOR}  PARENT_SCOPE)
    set(CK_DSL_PROVIDER_VERSION_PATCH  ${CK_DSL_PROVIDER_VERSION_PATCH}  PARENT_SCOPE)
    set(CK_DSL_PROVIDER_VERSION_TWEAK  ${CK_DSL_PROVIDER_VERSION_TWEAK}  PARENT_SCOPE)
    set(CK_DSL_PROVIDER_VERSION_STRING ${CK_DSL_PROVIDER_VERSION_STRING} PARENT_SCOPE)
    set(CK_DSL_PROVIDER_VERSION        ${CK_DSL_PROVIDER_VERSION}        PARENT_SCOPE)
endfunction()

# Generate version.h from templates/version.h.in and attach it to TARGET_NAME
# so every translation unit in the target sees CK_DSL_PROVIDER_VERSION_STRING.
# Call after add_library(<TARGET_NAME> ...).
function(ck_dsl_provider_generate_version_header TARGET_NAME)
    dnn_provider_generate_version_header(ck_dsl_provider
                                         ${TARGET_NAME}
                                         "${CK_DSL_PROVIDER_ROOT_DIR}")
endfunction()
