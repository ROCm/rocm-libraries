# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Experimental API registry: per-feature deadlines, the feature cap, and generation of
# hipblaslt-experimental-features.h. Policy: docs/reference/experimental-api.rst.
#
# Usage (see library/experimental/features.cmake):
#   set(ROCM_FEATURE_RELEASES <ordered list of ROCm major.minor releases>)
#   set(HIPBLASLT_ROCM_RELEASE <ROCm release this build targets>)
#   set(HIPBLASLT_EXPERIMENTAL_HEADER_DIR <dir holding hipblaslt-experimental-*.h>)
#   hipblaslt_experimental_feature(<NAME> HEADER ... OWNER ... TRACKING ...
#                                  INTRODUCED <rocm release> LAST_RELEASE <rocm release>
#                                  [EXTENDED_TO <rocm release> APPROVED_BY ...])
#   hipblaslt_experimental_finalize(TEMPLATE <in> OUTPUT <out> REGISTRY <out>)
#
# Releases are counted on ROCM_FEATURE_RELEASES, not in hipBLASLt versions.
include_guard(GLOBAL)

# Position of a ROCm release in ROCM_FEATURE_RELEASES. Patch versions are ignored.
function(_hipblaslt_release_index release what out)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" mm "${release}")
    list(FIND ROCM_FEATURE_RELEASES "${mm}" i)
    if(i EQUAL -1)
        message(FATAL_ERROR "${what}: ROCm release '${release}' is not in ROCM_FEATURE_RELEASES "
                            "(${ROCM_FEATURE_RELEASES}); add it to cmake/rocm_feature_releases.cmake")
    endif()
    set(${out} ${i} PARENT_SCOPE)
endfunction()

# Register one experimental feature. Fails the configure once the feature is past its deadline.
function(hipblaslt_experimental_feature name)
    cmake_parse_arguments(PARSE_ARGV 1 F ""
        "HEADER;OWNER;TRACKING;INTRODUCED;LAST_RELEASE;EXTENDED_TO;APPROVED_BY" "")
    set(what "experimental feature ${name}")
    foreach(req HEADER OWNER TRACKING INTRODUCED LAST_RELEASE)
        if(NOT F_${req})
            message(FATAL_ERROR "${what}: ${req} is required")
        endif()
    endforeach()

    # The header is named after the feature and defines its revision macro, so the checker,
    # the runtime query and consumers all agree on the name.
    string(TOLOWER "${name}" expected)
    string(REPLACE "_" "-" expected "hipblaslt-experimental-${expected}.h")
    if(NOT F_HEADER STREQUAL expected)
        message(FATAL_ERROR "${what}: HEADER must be ${expected}")
    endif()
    set(header "${HIPBLASLT_EXPERIMENTAL_HEADER_DIR}/${F_HEADER}")
    if(NOT EXISTS "${header}")
        message(FATAL_ERROR "${what}: ${F_HEADER} does not exist in "
                            "${HIPBLASLT_EXPERIMENTAL_HEADER_DIR}")
    endif()
    file(STRINGS "${header}" rev_line
         REGEX "^#[ \t]*define[ \t]+HIPBLASLT_EXPERIMENTAL_${name}[ \t]+[0-9]+[ \t]*$")
    if(NOT rev_line)
        message(FATAL_ERROR "${what}: ${F_HEADER} must #define HIPBLASLT_EXPERIMENTAL_${name} "
                            "to the feature's revision")
    endif()
    string(REGEX MATCH "[0-9]+[ \t]*$" revision "${rev_line}")
    string(STRIP "${revision}" revision)

    # A feature may ship in at most HIPBLASLT_EXPERIMENTAL_MAX_RELEASES ROCm releases, counted
    # from the release it entered. Revision bumps don't reset this.
    _hipblaslt_release_index(${F_INTRODUCED} "${what}: INTRODUCED" intro)
    _hipblaslt_release_index(${F_LAST_RELEASE} "${what}: LAST_RELEASE" last)
    math(EXPR span "${last} - ${intro}")
    if(span LESS 0 OR span GREATER_EQUAL HIPBLASLT_EXPERIMENTAL_MAX_RELEASES)
        message(FATAL_ERROR "${what}: ${F_INTRODUCED} through ${F_LAST_RELEASE} is more than "
                            "${HIPBLASLT_EXPERIMENTAL_MAX_RELEASES} releases")
    endif()

    # At most one extension, of exactly one release, naming who approved it.
    set(deadline ${F_LAST_RELEASE})
    set(deadline_index ${last})
    if(F_EXTENDED_TO)
        _hipblaslt_release_index(${F_EXTENDED_TO} "${what}: EXTENDED_TO" ext)
        math(EXPR ext_span "${ext} - ${last}")
        if(NOT ext_span EQUAL 1 OR NOT F_APPROVED_BY)
            message(FATAL_ERROR "${what}: EXTENDED_TO must be one release "
                                "after LAST_RELEASE (${F_LAST_RELEASE}) and needs APPROVED_BY")
        endif()
        set(deadline ${F_EXTENDED_TO})
        set(deadline_index ${ext})
    endif()

    _hipblaslt_release_index(${HIPBLASLT_ROCM_RELEASE} "HIPBLASLT_ROCM_RELEASE" current_index)
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" current "${HIPBLASLT_ROCM_RELEASE}")
    if(current_index GREATER deadline_index)
        message(FATAL_ERROR
            "hipBLASLt experimental feature ${name} was introduced in ROCm ${F_INTRODUCED} and "
            "may ship through ROCm ${deadline}, but this is ROCm ${current}.\n"
            "Graduate it to hipblaslt.h or remove it, or record a one-release extension in "
            "library/experimental/features.cmake.\n"
            "Owner: ${F_OWNER}  Tracking: ${F_TRACKING}")
    elseif(current_index EQUAL deadline_index)
        message(WARNING "hipBLASLt experimental feature ${name}: ROCm ${current} is the last "
                        "release that may ship it. Owner: ${F_OWNER}  Tracking: ${F_TRACKING}")
    endif()

    set_property(GLOBAL APPEND PROPERTY HIPBLASLT_EXPERIMENTAL_FEATURES ${name})
    set_property(GLOBAL APPEND PROPERTY HIPBLASLT_EXPERIMENTAL_HEADERS ${F_HEADER})
    set(line " *   ${name}: revision ${revision}, introduced ROCm ${F_INTRODUCED}, ")
    string(APPEND line "last release ROCm ${deadline}, owner ${F_OWNER}\n")
    set_property(GLOBAL APPEND_STRING PROPERTY HIPBLASLT_EXPERIMENTAL_BANNER "${line}")
endfunction()

# Check the feature cap and generate the public features header and the library's private
# feature list. Call after all features are registered. Sets HIPBLASLT_EXPERIMENTAL_FRAGMENTS
# (full paths of the registered feature headers) in the caller's scope.
function(hipblaslt_experimental_finalize)
    cmake_parse_arguments(PARSE_ARGV 0 F "" "TEMPLATE;OUTPUT;REGISTRY" "")
    get_property(features GLOBAL PROPERTY HIPBLASLT_EXPERIMENTAL_FEATURES)
    get_property(headers GLOBAL PROPERTY HIPBLASLT_EXPERIMENTAL_HEADERS)
    get_property(HIPBLASLT_EXPERIMENTAL_BANNER GLOBAL PROPERTY HIPBLASLT_EXPERIMENTAL_BANNER)

    list(LENGTH features n)
    if(n GREATER HIPBLASLT_EXPERIMENTAL_MAX_FEATURES)
        message(FATAL_ERROR "${n} experimental features (${features}); "
                            "the cap is ${HIPBLASLT_EXPERIMENTAL_MAX_FEATURES}")
    endif()

    # Every feature header on disk must be registered, so nothing ships without a deadline.
    set(dir "${HIPBLASLT_EXPERIMENTAL_HEADER_DIR}")
    file(GLOB on_disk RELATIVE "${dir}" "${dir}/hipblaslt-experimental-*.h")
    list(REMOVE_ITEM on_disk hipblaslt-experimental-features.h)
    foreach(h IN LISTS on_disk)
        if(NOT h IN_LIST headers)
            message(FATAL_ERROR "${h} is not registered in library/experimental/features.cmake")
        endif()
    endforeach()

    set(HIPBLASLT_EXPERIMENTAL_INCLUDES "")
    set(fragments "")
    foreach(h IN LISTS headers)
        string(APPEND HIPBLASLT_EXPERIMENTAL_INCLUDES "#include \"hipblaslt/${h}\"\n")
        list(APPEND fragments "${dir}/${h}")
    endforeach()
    configure_file("${F_TEMPLATE}" "${F_OUTPUT}" @ONLY)

    # The library's list of features, for hipblasLtExperimentalGetFeatureRevision. Not installed.
    if(F_REGISTRY)
        set(registry "// Generated from library/experimental/features.cmake; do not edit.\n")
        foreach(f IN LISTS features)
            string(APPEND registry "HIPBLASLT_EXPERIMENTAL_FEATURE(${f})\n")
        endforeach()
        file(CONFIGURE OUTPUT "${F_REGISTRY}" CONTENT "${registry}" @ONLY)
    endif()
    set(HIPBLASLT_EXPERIMENTAL_FRAGMENTS "${fragments}" PARENT_SCOPE)
endfunction()
