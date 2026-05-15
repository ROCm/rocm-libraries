# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Apply security hardening flags to a specific target.
#
# Adds FORTIFY_SOURCE and stack protection compiler flags to the given target.
# FORTIFY_SOURCE is disabled when building with address sanitizer to avoid conflicts.
# Stack protection is only enabled in optimized builds to avoid performance overhead.
function(apply_security_hardening target)
    if(NOT TARGET ${target})
        message(WARNING "Target '${target}' does not exist. Skipping security hardening.")
        return()
    endif()

    if(NOT BUILD_ADDRESS_SANITIZER AND NOT THEROCK_SANITIZER STREQUAL "ASAN" AND NOT THEROCK_SANITIZER STREQUAL "HOST_ASAN")
        target_compile_definitions(${target} PRIVATE
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:_FORTIFY_SOURCE=2>
        )
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|MSVC")
        if(MSVC)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX,C>:/GS>)
        else()
            # Only apply stack protection in optimized builds to avoid test performance degradation
            target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX,C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>>:-fstack-protector-strong>)
        endif()
    endif()
endfunction()

# Apply security hardening flags globally to all targets.
#
# Adds FORTIFY_SOURCE and stack protection compiler flags globally.
# FORTIFY_SOURCE is disabled when building with address sanitizer to avoid conflicts.
# Stack protection is only enabled in optimized builds to avoid performance overhead.
macro(apply_security_hardening_globally)
    if(NOT BUILD_ADDRESS_SANITIZER AND NOT THEROCK_SANITIZER STREQUAL "ASAN" AND NOT THEROCK_SANITIZER STREQUAL "HOST_ASAN")
        add_compile_definitions(
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:_FORTIFY_SOURCE=2>
        )
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|MSVC")
        if(MSVC)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX,C>:/GS>)
        else()
            # Only apply stack protection in optimized builds to avoid test performance degradation
            add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX,C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>>:-fstack-protector-strong>)
        endif()
    endif()
endmacro()

option(ROCM_LIBS_ENABLE_SECURITY_HARDENING "Enable security hardening (FORTIFY_SOURCE and stack protection)" ON)

if(ROCM_LIBS_ENABLE_SECURITY_HARDENING)
    message(STATUS "Security hardening enabled: FORTIFY_SOURCE and stack protection")
endif()
