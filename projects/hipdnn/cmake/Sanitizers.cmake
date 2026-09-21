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
            # The MSVC STL emits /INFERASANLIBS + /DEFAULTLIB:stl_asan.lib for ASAN container
            # annotations, but the x64 MSVC toolset does not ship stl_asan.lib (x86 only), so the
            # link fails. Disabling STL annotation drops that dependency; heap/stack/global/UAF
            # detection is unaffected (only intra-container red zones are lost).
            add_compile_definitions(_DISABLE_STL_ANNOTATION)

            block(SCOPE_FOR VARIABLES PROPAGATE TEST_ENVIRONMENT_MODIFICATION)
                # Test executables dynamically load DLLs that are not on the default Windows search
                # path, so prepend the directories that hold them: the ASAN runtime
                # (clang_rt.asan_dynamic-x86_64.dll, in the clang resource dir under lib/windows),
                # the freshly built backend/plugin DLLs (in the build bin dir), and the ROCm runtime
                # that MIOpen pulls in (MIOpen.dll, hiprtc*.dll, rocblas.dll, in
                # <ROCM_CMAKE_PATH>/bin). None of these exist in a default search location, so the
                # prepend makes them discoverable. Test registration applies this via CTest's
                # ENVIRONMENT_MODIFICATION (see add_hipdnn_test). path_list_prepend uses the host
                # path separator and the *runtime* PATH, avoiding the ';' collision and
                # configure-time-frozen-PATH problems of a literal PATH= entry.
                execute_process(
                    COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
                    OUTPUT_VARIABLE CLANG_RESOURCE_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                )
                file(TO_CMAKE_PATH "${CLANG_RESOURCE_DIR}/lib/windows" _asan_runtime_dir)
                file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}" _build_bin_dir)
                set(TEST_ENVIRONMENT_MODIFICATION
                    "PATH=path_list_prepend:${_asan_runtime_dir}"
                    "PATH=path_list_prepend:${_build_bin_dir}"
                )

                set(_rocm_root "${ROCM_CMAKE_PATH}")
                if(NOT _rocm_root)
                    set(_rocm_root "${ROCM_PATH}")
                endif()
                if(_rocm_root)
                    file(TO_CMAKE_PATH "${_rocm_root}/bin" _rocm_bin_dir)
                    list(APPEND TEST_ENVIRONMENT_MODIFICATION
                         "PATH=path_list_prepend:${_rocm_bin_dir}")
                endif()
            endblock()
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

# DLL-shadowing workaround (remove once ROCm fixes this on Windows): a stale System32 amd_comgr.dll
# shadows the TheRock one and breaks MIOpen's kernel JIT. PATH can't fix it (System32 precedes PATH),
# but the exe's own dir wins, so stage the DLL there for any Windows build that JITs MIOpen kernels.
# Tests.cmake wires stage_shadowed_rocm_dlls to test targets; the GLOBAL guard defines it once.
if(WIN32)
    block(SCOPE_FOR VARIABLES)
        get_property(_dll_shadow_staged GLOBAL PROPERTY _rocm_dlls_staged_dll_shadow_workaround)
        if(NOT _dll_shadow_staged)
            # ROCM_CMAKE_PATH and ROCM_PATH are mutually-exclusive ways to point at the ROCm root
            # (see ClangToolChain.cmake); prefer the former, fall back to the latter.
            set(_rocm_root "${ROCM_CMAKE_PATH}")
            if(NOT _rocm_root)
                set(_rocm_root "${ROCM_PATH}")
            endif()
            if(_rocm_root)
                set_property(GLOBAL PROPERTY _rocm_dlls_staged_dll_shadow_workaround TRUE)
                file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}" _build_bin_dir)
                set(_shadowed_dlls amd_comgr.dll)
                set(_staged_dlls "")
                foreach(_dll_name IN LISTS _shadowed_dlls)
                    set(_dst "${_build_bin_dir}/${_dll_name}")
                    add_custom_command(
                        OUTPUT "${_dst}"
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                "${_rocm_root}/bin/${_dll_name}" "${_dst}"
                        DEPENDS "${_rocm_root}/bin/${_dll_name}"
                        COMMENT "Staging ${_dll_name} into build bin (DLL-shadowing workaround)"
                        VERBATIM
                    )
                    list(APPEND _staged_dlls "${_dst}")
                endforeach()
                add_custom_target(stage_shadowed_rocm_dlls ALL DEPENDS ${_staged_dlls}
                    COMMENT "Staging shadowed ROCm DLLs into build bin")
            endif()
        endif()
    endblock()
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

    # Ensure the LLVM symbolizer is located before setting TEST_ENVIRONMENT.
    include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)
    findandcheckllvmsymbolizer()

    # Redirect MIOpen's kernel cache to a build-local dir, isolated from the developer's real
    # ~/.miopen and cleared once per ctest run (Tests.cmake registers the clearing fixture from this
    # variable) so a stale/poisoned entry cannot mask a failure across runs.
    set(HIPDNN_TEST_MIOPEN_CACHE_DIR "${CMAKE_BINARY_DIR}/miopen_test_cache")

    # Set environment variables for Address Sanitizer.
    # HSA_XNACK is only required for device-side ASAN (not HOST_ASAN).
    # ASAN_SYMBOLIZER_PATH is set to the LLVM symbolizer to make the output from leak detection
    # more readable.
    if(BUILD_ADDRESS_SANITIZER OR THEROCK_SANITIZER STREQUAL "ASAN")
        set(TEST_ENVIRONMENT "ASAN_SYMBOLIZER_PATH=${CMAKE_SYMBOLIZER}" "HSA_XNACK=1"
                             "MIOPEN_CUSTOM_CACHE_DIR=${HIPDNN_TEST_MIOPEN_CACHE_DIR}"
                             # "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1"
        )
    else()
        # HOST_ASAN only needs the symbolizer, not HSA_XNACK
        set(TEST_ENVIRONMENT "ASAN_SYMBOLIZER_PATH=${CMAKE_SYMBOLIZER}"
                             "MIOPEN_CUSTOM_CACHE_DIR=${HIPDNN_TEST_MIOPEN_CACHE_DIR}"
                             # "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1"
        )
    endif()
    message(VERBOSE "ASAN ${CMAKE_CURRENT_SOURCE_DIR} TEST_ENVIRONMENT=${TEST_ENVIRONMENT}")

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
