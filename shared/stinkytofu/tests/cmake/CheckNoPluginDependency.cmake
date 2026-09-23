# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Regression guard for #8342: libstinkytofu must never carry a DT_NEEDED on a
# plugin. A plugin is discovered and dlopen'd by the host at runtime, so linking
# one into libstinkytofu would invert the relationship — and if that plugin were
# absent (e.g. STINKYTOFU_BUILD_EXAMPLES=OFF) the loader would fail to load
# libstinkytofu itself, surfacing as "libstinkytofu not found".
#
# Invoked as a CTest with -DLIB=<path to libstinkytofu> -DREADELF=<readelf>.

if(NOT EXISTS "${LIB}")
    message(FATAL_ERROR "libstinkytofu not found at: ${LIB}")
endif()

execute_process(
    COMMAND "${READELF}" -d "${LIB}"
    OUTPUT_VARIABLE _dynamic
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "readelf failed on ${LIB} (exit ${_rc})")
endif()

# Collect every NEEDED line, then fail if any names a plugin.
string(REGEX MATCHALL "\\(NEEDED\\)[^\n]*" _needed "${_dynamic}")
set(_bad "")
foreach(_line IN LISTS _needed)
    if(_line MATCHES "plugin")
        list(APPEND _bad "${_line}")
    endif()
endforeach()

if(_bad)
    string(REPLACE ";" "\n  " _bad_str "${_bad}")
    message(FATAL_ERROR
        "libstinkytofu has a DT_NEEDED on a plugin — it must not (see #8342):\n  ${_bad_str}")
endif()

message(STATUS "OK: libstinkytofu has no plugin in its DT_NEEDED")
