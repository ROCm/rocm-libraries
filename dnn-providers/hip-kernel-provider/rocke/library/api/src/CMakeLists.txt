# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_library(
    rocke_client_impl OBJECT
    RockeClientContainer.cpp
    RockeClientHandle.cpp
    PluginPublic.cpp
    engines/RockeClientEngine.cpp
    dispatcher/AotCatalog.cpp
    dispatcher/RockeClientDispatcher.cpp
    dispatcher/SdpaGraphAdapter.cpp
    dispatcher/SdpaProblem.cpp
    dispatcher/SelectionConstraints.cpp
    # Generated model registry (gen_model_registry.py). Provides
    # rocke_lookup_model(); regenerate + keep listed when predictors are added.
    dispatcher/sdpa_fwd/rocke_model_registry.c
    # Committed predictors (one per op/arch/dtype). Example: bf16/gfx950 (smoke
    # model, 12 shapes) demonstrating the layout + registry wiring.
    dispatcher/sdpa_fwd/gfx950/model_tflops_bf16.c
)

# The generated registry + predictors are emitted as .c (plain C, valid as C++).
# Compile them with the CXX toolchain so the project needs no separate C language
# enabled (the ClangToolChain configures CXX only). rocke_lookup_model keeps C
# linkage via the extern "C" guard in its header; the predictor symbols are
# referenced only through the registry's function-pointer table, so consistent
# CXX compilation of both keeps linkage matched.
#
# SKIP_LINTING: these are GENERATED files (lgbm_to_c.py, gen_model_registry.py,
# gen_fmha_featurizer.py); linting them is pointless and causes noise on regen.
set_source_files_properties(
    dispatcher/sdpa_fwd/rocke_model_registry.c
    dispatcher/sdpa_fwd/gfx950/model_tflops_bf16.c
    dispatcher/sdpa_fwd/FmhaFeaturizer.hpp
    PROPERTIES
        LANGUAGE CXX
        SKIP_LINTING ON
)

target_include_directories(rocke_client_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
dnn_provider_generate_version_header(rocke_client rocke_client_impl ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(rocke_client_impl PUBLIC hip::host hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk)

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
target_link_libraries(rocke_client_private PUBLIC hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk rocke_client_impl)

add_library(rocke_client SHARED $<TARGET_OBJECTS:rocke_client_impl>)
target_link_libraries(rocke_client PRIVATE hip::host hipdnn_data_sdk hipdnn_flatbuffers_sdk hipdnn_plugin_sdk)

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

# Custom target: regenerate FmhaFeaturizer.hpp from gen_fmha_featurizer.py.
# Usage: cmake --build . --target rocke_regenerate_fmha_featurizer
# The generator lives in platform/python/rocke/heuristics; invoke it via python3 -m.
add_custom_target(
    rocke_regenerate_fmha_featurizer
    COMMAND ${CMAKE_COMMAND} -E echo "Regenerating FmhaFeaturizer.hpp..."
    COMMAND python3 -m rocke.heuristics.gen_fmha_featurizer
            --out ${CMAKE_CURRENT_SOURCE_DIR}/dispatcher/sdpa_fwd/FmhaFeaturizer.hpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../platform/python
    COMMENT "Regenerate dispatcher/sdpa_fwd/FmhaFeaturizer.hpp from gen_fmha_featurizer.py"
    VERBATIM
)

# Custom target: regenerate rocke_model_registry.c from gen_model_registry.py.
# Usage: cmake --build . --target rocke_regenerate_model_registry
# Scans dispatcher/sdpa_fwd/*/*.meta.json and emits the registry + extern decls.
add_custom_target(
    rocke_regenerate_model_registry
    COMMAND ${CMAKE_COMMAND} -E echo "Regenerating rocke_model_registry.c/.h..."
    COMMAND python3 -m rocke.heuristics.gen_model_registry
            --models-dir ${CMAKE_CURRENT_SOURCE_DIR}/dispatcher/sdpa_fwd
            --out-c ${CMAKE_CURRENT_SOURCE_DIR}/dispatcher/sdpa_fwd/rocke_model_registry.c
            --out-h ${CMAKE_CURRENT_SOURCE_DIR}/dispatcher/sdpa_fwd/rocke_model_registry.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../platform/python
    COMMENT "Regenerate rocke_model_registry.{c,h} from committed predictors"
    VERBATIM
)

if(ROCKE_CLIENT_ENABLE_TESTS)
    add_subdirectory(tests)
    add_subdirectory(integration_tests)
endif()
