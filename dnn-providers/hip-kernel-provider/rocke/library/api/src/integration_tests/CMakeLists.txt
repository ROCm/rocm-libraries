# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    rocke_client_integration_tests
    main.cpp
    TestRockeClientLoad.cpp
    TestRockeClientApplicability.cpp
)

target_compile_options(rocke_client_integration_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# rocke-client is gated entirely behind HIPKERNELPROVIDER_ENABLE_ROCKE, so its
# integration tests always drive the SDPA graph path regardless of the backend's
# HIPDNN_ENABLE_SDPA setting. Define it unconditionally for this target.
target_compile_definitions(
    rocke_client_integration_tests
    PRIVATE
        PLUGIN_PATH="../$<IF:$<PLATFORM_ID:Windows>,bin,lib>/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/rocke-client"
        HIPDNN_ENABLE_SDPA
)

target_link_libraries(
    rocke_client_integration_tests PUBLIC GTest::gtest hip::host hipdnn_test_sdk
                                          hipdnn_plugin_sdk Threads::Threads
)

add_dependencies(rocke_client_integration_tests rocke_client)

# ---------------------------------------------------------------------------
# AOT probe integration tests: separate executable (temporary, AICK-1484)
# ---------------------------------------------------------------------------
# A plugin's log level is latched on first load and cannot be raised afterward.
# TestRockeClientAotLoad raises the level to INFO (ScopedGlobalLogLevel) before
# the first load and asserts on the AOT_PROBE_LOAD_OK/_FAILED markers that
# loadForDevice() emits at INFO. In a shared process another test can load the
# plugin first at the default level, latching the markers off; a dedicated
# process keeps the AOT test the first to load, so INFO takes effect.
add_executable(
    rocke_client_aot_probe_tests
    main.cpp
    TestRockeClientAotLoad.cpp
)

target_compile_options(rocke_client_aot_probe_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

target_compile_definitions(
    rocke_client_aot_probe_tests
    PRIVATE
        PLUGIN_PATH="../$<IF:$<PLATFORM_ID:Windows>,bin,lib>/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/rocke-client"
        HIPDNN_ENABLE_SDPA
)

target_link_libraries(
    rocke_client_aot_probe_tests PUBLIC GTest::gtest hip::host hipdnn_test_sdk
                                        hipdnn_plugin_sdk Threads::Threads
)

# Expose api/src/ for dispatcher/AotProbeMarkers.hpp (temporary, AICK-1484).
target_include_directories(rocke_client_aot_probe_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_dependencies(rocke_client_aot_probe_tests rocke_client)

# ---------------------------------------------------------------------------
# Build-time test bundles (test-only; installed + relocatable)
# ---------------------------------------------------------------------------
# Generate per-arch valid + corrupt probe bundles beside the installed test
# layout so the test finds them RELATIVE TO ITS OWN EXECUTABLE, both in the
# build tree and after a relocatable install:
#   <exe_dir>/hip_kernel_provider/tests/aot_test_bundles/{valid,corrupt}/<arch>/
#
# TestRockeClientAotLoad::bundleRootDir() resolves that via
# getCurrentExecutableDirectory(); each test sets ROCKE_CLIENT_AOT_BUNDLE_DIR to
# the valid|corrupt/<arch> subdir, which aotBundleDir() returns directly (no
# arch_content/rocke/<arch> suffix).
#
# These are dev/test content: installed under bin/hip_kernel_provider/tests/ and
# captured by the existing hipkernelprovider [test] artifact glob
# (bin/hip_kernel_provider/tests/**), so NO TheRock packaging change is needed.
# They are NEVER placed under arch_content and NEVER enter the runtime lib
# artifact.
#
# Requires the rocke pyenv (ROCKE_PYENV_PYTHON) and rocm_kpack Python source
# (ROCKE_KPACK_PYTHON_DIR) already resolved by the parent build system, plus
# hipcc and clang-offload-bundler on PATH.
if(DEFINED ROCKE_PYENV_PYTHON AND DEFINED ROCKE_KPACK_PYTHON_DIR)
    find_program(ROCKE_HIPCC hipcc HINTS "${ROCM_PATH}/bin" "${HIP_ROOT_DIR}/bin")
    find_program(ROCKE_CLANG_OFFLOAD_BUNDLER clang-offload-bundler
        HINTS "${ROCM_PATH}/lib/llvm/bin")

    if(ROCKE_HIPCC AND ROCKE_CLANG_OFFLOAD_BUNDLER)
        set(_probe_src      "${CMAKE_CURRENT_SOURCE_DIR}/../tests/probe.hip")
        set(_gen_valid      "${CMAKE_CURRENT_SOURCE_DIR}/../tests/generate_test_bundle.py")
        set(_gen_corrupt    "${CMAKE_CURRENT_SOURCE_DIR}/../tests/generate_corrupt_bundle.py")
        set(_bundle_root    "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/hip_kernel_provider/tests/aot_test_bundles")

        set(_all_bundle_outputs "")
        # Both generators run through the rocke pyenv interpreter
        # (ROCKE_PYENV_PYTHON), so each custom command MUST be ordered after the
        # venv is built. Depend on the pyenv STAMP FILE (${ROCKE_PYENV_STAMP}),
        # not the rocke-pyenv target: that target is created later in
        # rocke/CMakeLists.txt (after add_subdirectory(library)), so it does not
        # exist yet in this scope and if(TARGET rocke-pyenv) would be false. The
        # stamp path is an inherited variable here, and a cross-directory
        # add_custom_command OUTPUT file dependency resolves under Ninja
        # regardless of subdirectory processing order (same pattern as
        # rocke_aot.cmake). add_dependencies() on the aggregate target alone is
        # NOT sufficient: it orders only the aggregate's stamp, not the commands
        # that produce its DEPENDS, so ninja could otherwise run them before the
        # venv interpreter exists (seen on Windows: "system cannot find the path
        # specified").

        foreach(_arch IN LISTS GPU_TARGETS)
            set(_probe_work  "${CMAKE_CURRENT_BINARY_DIR}/probe_work/${_arch}")
            set(_probe_co    "${_probe_work}/probe.co")
            set(_probe_hsaco "${_probe_work}/probe_${_arch}.hsaco")
            set(_valid_dir   "${_bundle_root}/valid/${_arch}")
            set(_corrupt_dir "${_bundle_root}/corrupt/${_arch}")
            set(_valid_kpack "${_valid_dir}/rocke_client_${_arch}.kpack")
            set(_valid_mfst  "${_valid_dir}/rocke_client_${_arch}.json")
            set(_corr_kpack  "${_corrupt_dir}/rocke_client_${_arch}.kpack")
            set(_corr_mfst   "${_corrupt_dir}/rocke_client_${_arch}.json")

            # Step 1: compile probe.hip -> unbundled bare HSACO
            add_custom_command(
                OUTPUT "${_probe_hsaco}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_probe_work}"
                COMMAND "${ROCKE_HIPCC}" --genco --offload-arch=${_arch}
                        -o "${_probe_co}" "${_probe_src}"
                COMMAND "${ROCKE_CLANG_OFFLOAD_BUNDLER}"
                        --type=o
                        "--inputs=${_probe_co}"
                        "--targets=hipv4-amdgcn-amd-amdhsa--${_arch}"
                        "--unbundle"
                        "--output=${_probe_hsaco}"
                DEPENDS "${_probe_src}"
                COMMENT "Compiling rocKE probe HSACO for ${_arch}"
            )

            # Step 2a: pack into valid kpack + manifest
            add_custom_command(
                OUTPUT "${_valid_kpack}" "${_valid_mfst}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_valid_dir}"
                COMMAND "${ROCKE_PYENV_PYTHON}" "${_gen_valid}"
                        --hsaco "${_probe_hsaco}"
                        --arch  "${_arch}"
                        --outdir "${_valid_dir}"
                        --kpack-python-dir "${ROCKE_KPACK_PYTHON_DIR}"
                DEPENDS "${_probe_hsaco}" "${_gen_valid}" "${ROCKE_PYENV_STAMP}"
                COMMENT "Generating valid rocKE test bundle for ${_arch}"
            )

            # Step 2b: generate corrupt kpack + valid manifest (no Python deps beyond stdlib)
            add_custom_command(
                OUTPUT "${_corr_kpack}" "${_corr_mfst}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_corrupt_dir}"
                COMMAND "${ROCKE_PYENV_PYTHON}" "${_gen_corrupt}"
                        --arch  "${_arch}"
                        --outdir "${_corrupt_dir}"
                DEPENDS "${_gen_corrupt}" "${ROCKE_PYENV_STAMP}"
                COMMENT "Generating corrupt rocKE test bundle for ${_arch}"
            )

            list(APPEND _all_bundle_outputs
                "${_valid_kpack}" "${_valid_mfst}"
                "${_corr_kpack}"  "${_corr_mfst}")
        endforeach()

        add_custom_target(rocke_client_aot_test_bundles
            DEPENDS ${_all_bundle_outputs}
            COMMENT "Generating rocKE AOT integration-test bundles")
        add_dependencies(rocke_client_aot_probe_tests rocke_client_aot_test_bundles)

        # Install the generated bundles as test-only content. This lands under
        # bin/hip_kernel_provider/tests/aot_test_bundles/ and is captured by the
        # existing hipkernelprovider [test] artifact glob
        # (bin/hip_kernel_provider/tests/**), so no TheRock packaging change.
        install(DIRECTORY "${_bundle_root}/"
                DESTINATION "${CMAKE_INSTALL_BINDIR}/hip_kernel_provider/tests/aot_test_bundles")
    else()
        message(STATUS
            "rocke-client: hipcc or clang-offload-bundler not found; "
            "AOT integration tests will SKIP (no probe bundle)")
    endif()
else()
    message(STATUS
        "rocke-client: ROCKE_PYENV_PYTHON/ROCKE_KPACK_PYTHON_DIR not set; "
        "AOT integration tests will SKIP (no probe bundle)")
endif()

clang_tidy_check(rocke_client_integration_tests)
clang_tidy_check(rocke_client_aot_probe_tests)

add_integration_test_target(rocke_client_integration_tests ${CMAKE_CURRENT_BINARY_DIR}
    LABELS quick)

add_integration_test_target(rocke_client_aot_probe_tests ${CMAKE_CURRENT_BINARY_DIR}
    LABELS quick)
