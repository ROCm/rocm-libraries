# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Address Sanitizer and Thread Sanitizer are mutually exclusive
if(BUILD_ADDRESS_SANITIZER AND BUILD_THREAD_SANITIZER)
    message(FATAL_ERROR "BUILD_ADDRESS_SANITIZER and BUILD_THREAD_SANITIZER cannot both be enabled. "
                        "These sanitizers are mutually exclusive."
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
# of TheRock. Windows and Linux are supported.
if(BUILD_ADDRESS_SANITIZER)

    # Windows configuration
    if (WIN32)
        message(STATUS "Configuring Address Sanitizer for Windows")

        # ASAN is incompatible with the MSVC debug CRT (/MDd). Force the release CRT (/MD)
        # to avoid false "bad-free" errors from ucrtbased.dll during static initialization.
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

        # Windows: MSVC uses /fsanitize=address, Clang uses -fsanitize=address.
        if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(SANITIZER_COMPILE_FLAGS /fsanitize=address)
            set(SANITIZER_LINK_FLAGS    /fsanitize=address)
        else()
            set(SANITIZER_COMPILE_FLAGS -fsanitize=address -fno-omit-frame-pointer)
            set(SANITIZER_LINK_FLAGS    -fsanitize=address -fno-omit-frame-pointer)

            # -print-resource-dir is clang-specific; only query it for non-MSVC compilers.
            execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
                OUTPUT_VARIABLE CLANG_RESOURCE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            link_directories(${CLANG_RESOURCE_DIR}/lib/windows)
        endif()

        add_compile_options(${SANITIZER_COMPILE_FLAGS})
        add_link_options(${SANITIZER_LINK_FLAGS})

    # Linux configuration
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(STATUS "Configuring Address Sanitizer for Linux")

        # GPU targets for Linux ASAN
        set(GPU_TARGETS
            gfx908:xnack+
            gfx90a:xnack+
            gfx942:xnack+
        )

        # Query the compiler for the resource directory to locate sanitizer runtime libraries.
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
            OUTPUT_VARIABLE CLANG_RESOURCE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        link_directories(${CLANG_RESOURCE_DIR}/lib/linux)

        set(SANITIZER_COMPILE_FLAGS -fsanitize=address -fno-omit-frame-pointer)
        set(SANITIZER_LINK_FLAGS    -fsanitize=address -fno-omit-frame-pointer -shared-libasan)

        add_compile_options(${SANITIZER_COMPILE_FLAGS})
        add_link_options(${SANITIZER_LINK_FLAGS})

    # Unsupported platform
    else()
        message(FATAL_ERROR
            "BUILD_ADDRESS_SANITIZER is only supported on Windows or Linux."
        )
    endif()

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
