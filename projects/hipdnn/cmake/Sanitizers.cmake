# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Address Sanitizer and Thread Sanitizer are mutually exclusive
if(BUILD_ADDRESS_SANITIZER AND BUILD_THREAD_SANITIZER)
    message(FATAL_ERROR "BUILD_ADDRESS_SANITIZER and BUILD_THREAD_SANITIZER cannot both be enabled. "
                        "These sanitizers are mutually exclusive."
    )
endif()

# Standalone sanitizer builds require Linux-specific compiler runtime libraries and flags.
# See: https://github.com/ROCm/rocm-libraries/issues/5022
if(BUILD_ADDRESS_SANITIZER AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "BUILD_ADDRESS_SANITIZER is only supported on Linux. "
                        "The standalone sanitizer build flow requires Linux-specific "
                        "compiler runtime libraries and flags."
    )
endif()

if(BUILD_THREAD_SANITIZER AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "BUILD_THREAD_SANITIZER is only supported on Linux. "
                        "Thread Sanitizer is not available on Windows and the standalone "
                        "sanitizer build flow requires Linux-specific compiler runtime "
                        "libraries and flags."
    )
endif()

# Enable Address Sanitizer and set linker flags. This configuration is for standalone builds outside
# of TheRock
if(BUILD_ADDRESS_SANITIZER)

    # Address Sanitizer requires specific GPU targets which support XNACK.
    set(GPU_TARGETS
        gfx908:xnack+ # MI100 (Arcturus)
        gfx90a:xnack+ # MI200 series (MI210, MI250, MI250X)
        gfx942:xnack+ # MI300X (GPU)
    )

    # Query the compiler for the resource directory to locate sanitizer libraries reliably
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir OUTPUT_VARIABLE CLANG_RESOURCE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    link_directories(${CLANG_RESOURCE_DIR}/lib/linux)

    # Define sanitizer flags as variables for reuse
    set(SANITIZER_COMPILE_FLAGS -fsanitize=address -fno-omit-frame-pointer)

    set(SANITIZER_LINK_FLAGS -fsanitize=address -fno-omit-frame-pointer -shared-libasan)

    # Apply sanitizer flags globally (can be overridden per target)
    add_compile_options(${SANITIZER_COMPILE_FLAGS})
    add_link_options(${SANITIZER_LINK_FLAGS})

endif()

# Enable Thread Sanitizer and set linker flags. This configuration is for standalone builds outside
# of TheRock. TSAN is host-side only and does not require specific GPU targets.
if(BUILD_THREAD_SANITIZER)

    # Query the compiler for the resource directory to locate sanitizer libraries reliably
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir OUTPUT_VARIABLE CLANG_RESOURCE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    link_directories(${CLANG_RESOURCE_DIR}/lib/linux)

    # Define sanitizer flags as variables for reuse
    set(SANITIZER_COMPILE_FLAGS -fsanitize=thread -fno-omit-frame-pointer)

    set(SANITIZER_LINK_FLAGS -fsanitize=thread -fno-omit-frame-pointer -shared-libsan)

    # Apply sanitizer flags globally (can be overridden per target)
    add_compile_options(${SANITIZER_COMPILE_FLAGS})
    add_link_options(${SANITIZER_LINK_FLAGS})

endif()

# These settings are applied whether building with TheRock or standalone
if(BUILD_ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")

    message(STATUS "Building with Address Sanitizer: ON")

    # Add compile definition for conditional compilation
    add_compile_definitions(ADDRESS_SANITIZER)

    # Set environment variables for Address Sanitizer.
    # HSA_XNACK is only required for device-side ASAN (not HOST_ASAN).
    # ASAN_SYMBOLIZER_PATH is set to the LLVM symbolizer to make the output from leak detection
    # more readable.
    if(BUILD_ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN")
        set(TEST_ENVIRONMENT "ASAN_SYMBOLIZER_PATH=${CMAKE_SYMBOLIZER}" "HSA_XNACK=1"
                             # "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1"
        )
    else()
        # HOST_ASAN only needs the symbolizer, not HSA_XNACK
        set(TEST_ENVIRONMENT "ASAN_SYMBOLIZER_PATH=${CMAKE_SYMBOLIZER}"
                             # "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1"
        )
    endif()

endif()

# These settings are applied whether building with TheRock or standalone
if(BUILD_THREAD_SANITIZER OR THEROCK_SANITIZER STREQUAL "TSAN")

    message(STATUS "Building with Thread Sanitizer: ON")

    # Add compile definition for conditional compilation
    add_compile_definitions(THREAD_SANITIZER)

    # Both standalone (-fsanitize=thread) and TheRock builds link the TSAN runtime directly into
    # binaries, so LD_PRELOAD is not needed. Using LD_PRELOAD would double-load the runtime and
    # cause a segfault.

endif()
