# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_library(
    rocke_client_impl OBJECT
    RockeClientContainer.cpp
    RockeClientHandle.cpp
    PluginPublic.cpp
    engines/RockeClientEngine.cpp
    dispatcher/AotCatalog.cpp
    dispatcher/KpackModuleLoader.cpp
    dispatcher/PluginModuleDir.cpp
    dispatcher/RockeClientDispatcher.cpp
    dispatcher/SdpaGraphAdapter.cpp
    dispatcher/SdpaProblem.cpp
    dispatcher/SelectionConstraints.cpp
)

target_include_directories(rocke_client_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
dnn_provider_generate_version_header(rocke_client rocke_client_impl ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(rocke_client_impl PUBLIC hip::host hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk rocm::rocm_kpack)
target_link_libraries(rocke_client_impl PRIVATE nlohmann_json::nlohmann_json)

target_compile_options(rocke_client_impl PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(rocke_client_impl PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
clang_tidy_check(rocke_client_impl)

add_library(rocke_client_private STATIC $<TARGET_OBJECTS:rocke_client_impl>)
target_include_directories(rocke_client_private PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_link_libraries(rocke_client_private PUBLIC hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk rocke_client_impl rocm::rocm_kpack)

add_library(rocke_client SHARED $<TARGET_OBJECTS:rocke_client_impl>)
target_link_libraries(rocke_client PRIVATE hip::host hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk rocm::rocm_kpack ${CMAKE_DL_LIBS})

if(NOT WIN32)
    target_link_libraries(rocke_client PRIVATE "-Xlinker --exclude-libs=ALL")
endif()

target_compile_options(rocke_client PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    rocke_client
    PROPERTIES OUTPUT_NAME "rocke-client"
               CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

set_target_properties(rocke_client PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")

if(ROCKE_CLIENT_ENABLE_TESTS)
    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    # Build-time probe bundle: compile a trivial HIP kernel cross-arch and pack
    # it into rocke_client_<arch>.kpack + rocke_client_<arch>.json beside the
    # plugin.  loadDefault() discovers and loads this bundle in the integration
    # tests.  NOT installed / NOT shipped -- pure test artifact.
    #
    # Requires ROCKE_PYENV_PYTHON (from rocke pyenv) and
    # ROCKE_KPACK_PYTHON_DIR (-DROCKE_KPACK_PYTHON_DIR=<rocm-systems>/shared/kpack/python).
    if(DEFINED ROCKE_PYENV_PYTHON AND DEFINED ROCKE_KPACK_PYTHON_DIR)
        find_program(ROCKE_HIPCC hipcc HINTS "${ROCM_PATH}/bin" "${HIP_ROOT_DIR}/bin")
        find_program(ROCKE_CLANG_OFFLOAD_BUNDLER clang-offload-bundler
            HINTS "${ROCM_PATH}/lib/llvm/bin")

        if(ROCKE_HIPCC AND ROCKE_CLANG_OFFLOAD_BUNDLER)
            set(_probe_src "${CMAKE_CURRENT_SOURCE_DIR}/tests/probe.hip")
            set(_gen_script "${CMAKE_CURRENT_SOURCE_DIR}/tests/generate_test_bundle.py")
            set(_bundle_base "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/arch_content/rocke")

            set(_all_probe_outputs "")
            foreach(_arch IN LISTS GPU_TARGETS)
                set(_probe_work "${CMAKE_CURRENT_BINARY_DIR}/probe_work/${_arch}")
                set(_probe_co   "${_probe_work}/probe.co")
                set(_probe_hsaco "${_probe_work}/probe_${_arch}.hsaco")
                set(_out_kpack  "${_bundle_base}/${_arch}/rocke_client_${_arch}.kpack")
                set(_out_mfst   "${_bundle_base}/${_arch}/rocke_client_${_arch}.json")

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

                add_custom_command(
                    OUTPUT "${_out_kpack}" "${_out_mfst}"
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${_bundle_base}/${_arch}"
                    COMMAND "${ROCKE_PYENV_PYTHON}" "${_gen_script}"
                            --hsaco "${_probe_hsaco}"
                            --arch "${_arch}"
                            --outdir "${_bundle_base}/${_arch}"
                            --kpack-python-dir "${ROCKE_KPACK_PYTHON_DIR}"
                    DEPENDS "${_probe_hsaco}" "${_gen_script}" rocke_client
                    COMMENT "Generating rocKE test bundle for ${_arch}"
                )

                list(APPEND _all_probe_outputs "${_out_kpack}" "${_out_mfst}")
            endforeach()

            add_custom_target(rocke_client_probe_bundles
                DEPENDS ${_all_probe_outputs}
                COMMENT "Generate rocKE test probe bundles for all GPU_TARGETS")
            add_dependencies(rocke_client_integration_tests rocke_client_probe_bundles)
            # The pack step invokes the rocke pyenv interpreter; make sure the
            # pyenv is built first (avoids a clean-build race where the bundle
            # command runs before rocke-pyenv/bin/python exists).
            if(TARGET rocke-pyenv)
                add_dependencies(rocke_client_probe_bundles rocke-pyenv)
            endif()
        else()
            message(STATUS
                "rocke-client: hipcc or clang-offload-bundler not found; "
                "skipping probe bundle generation (integration tests will SKIP)")
        endif()
    else()
        message(STATUS
            "rocke-client: ROCKE_PYENV_PYTHON/ROCKE_KPACK_PYTHON_DIR not set; "
            "skipping probe bundle generation (integration tests will SKIP)")
    endif()
endif()
