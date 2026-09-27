# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The plugin resolves kpack descriptors, which read their kernel code from an archive on
# disk, so nothing is embedded there. The unit-test binary resolves both kinds: it reads
# kpack descriptors from a staged archive, and for embedded_source descriptors it links
# this engine statically and embeds its own copy of the kernel sources.

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/IngestorPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelIngestorEngine.cpp
    # Descriptors naming these packs are not in this tree. They arrive from the install tree:
    # test_descriptors/ for the test binaries, and descriptor-packaging/examples/descriptors
    # where the superbuild CI lane points the production source root.
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/PointwiseNative.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/ConvNative.cpp
)

# Sources above are entirely inside `#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR`; the
# definition must reach this object library, not only the static library tests link.
target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENABLE_KERNEL_INGESTOR=1)

if(HIPKERNELPROVIDER_ENABLE_ROCKE_RECIPE_EXAMPLE)
    if(NOT HIPKERNELPROVIDER_ENABLE_ROCKE)
        message(FATAL_ERROR "The recipe example requires HIPKERNELPROVIDER_ENABLE_ROCKE")
    endif()
    find_path(ROCKE_COMGR_INCLUDE amd_comgr.h HINTS "${ROCM_PATH}/include" PATH_SUFFIXES amd_comgr REQUIRED)
    find_library(ROCKE_COMGR_LIBRARY amd_comgr HINTS "${ROCM_PATH}/lib" REQUIRED)
    target_sources(hip_kernel_provider_impl PRIVATE
        packs/RockeSdpaNative.cpp
        ../../compilation/RockeRecipe.cpp)
    target_include_directories(hip_kernel_provider_impl PRIVATE
        "${ROCKE_COMGR_INCLUDE}" "${PROJECT_SOURCE_DIR}/rocke/platform/cpp/include")
    target_link_libraries(hip_kernel_provider_impl PUBLIC rocke_core "${ROCKE_COMGR_LIBRARY}")
    target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPKERNELPROVIDER_ENABLE_ROCKE_RECIPE_EXAMPLE=1)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    include("${PROJECT_SOURCE_DIR}/descriptor-packaging/cmake/HkpPackaging.cmake")
    hkp_selected_arches(_recipe_arches _recipe_arch_source)
    if(NOT _recipe_arches)
        message(FATAL_ERROR "The SDPA recipe example requires GPU_TARGETS or AMDGPU_TARGETS")
    endif()
    foreach(_requested IN LISTS ${_recipe_arch_source})
        string(REGEX REPLACE ":.*$" "" _bare "${_requested}")
        if(NOT _bare IN_LIST _recipe_arches)
            message(FATAL_ERROR "The SDPA recipe example requires concrete targets: ${_requested}")
        endif()
    endforeach()
    foreach(_arch IN LISTS _recipe_arches)
        if(NOT _arch MATCHES "^gfx(942|950)$")
            message(FATAL_ERROR "Unsupported SDPA recipe target: ${_arch}; supported: gfx942, gfx950")
        endif()
    endforeach()
    set(HIPKERNELPROVIDER_ROCKE_RECIPE_LLVM_FLAVOR "llvm23" CACHE STRING
        "LLVM lowerer matching the runtime COMGR toolchain for the SDPA recipe example")
    set(_recipe_fixture "${PROJECT_SOURCE_DIR}/src/integration_tests/kernel_ingestor_engine/fixtures/rocke_recipe")
    set(_recipe_staging "${HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR}/rocke_sdpa")
    # HKP installs the descriptor tree wholesale. Keep reference LLVM and
    # authoring provenance outside it; only descriptors and CBOR are runtime inputs.
    set(_recipe_authoring "${CMAKE_CURRENT_BINARY_DIR}/rocke_sdpa_authoring")
    file(MAKE_DIRECTORY "${_recipe_staging}")
    foreach(_descriptor IN ITEMS sdpa.ued.json sdpa.kmd.json sdpa.uhd.json
            sdpa.umd.json sdpa.udd.json)
        configure_file("${_recipe_fixture}/descriptors/${_descriptor}"
            "${_recipe_staging}/${_descriptor}" COPYONLY)
    endforeach()
    # Reconfiguration must invalidate outputs for Make as well as Ninja when
    # targets or the authoring flavor change.
    configure_file("${_recipe_fixture}/authoring-config.txt.in"
        "${_recipe_authoring}/authoring-config.txt" @ONLY)
    # Conservative authoring dependencies: imported builder/lowerer changes must
    # regenerate the bundle. This glob tracks data generation, not compiled sources.
    file(GLOB_RECURSE _recipe_authoring_sources CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/rocke/platform/python/rocke/*.py"
        "${PROJECT_SOURCE_DIR}/rocke/library/*.py")
    add_custom_command(OUTPUT "${_recipe_staging}/sdpa_dense.cbor" "${_recipe_staging}/sdpa.kdp.json"
        BYPRODUCTS "${_recipe_authoring}/manifest.json"
        COMMAND "${CMAKE_COMMAND}" -E env ROCKE_BACKEND=python
            "ROCKE_LLVM_FLAVOR=${HIPKERNELPROVIDER_ROCKE_RECIPE_LLVM_FLAVOR}"
            "${Python3_EXECUTABLE}" "${_recipe_fixture}/produce_sdpa.py" "${_recipe_authoring}"
            --arches ${_recipe_arches} --llvm-flavor "${HIPKERNELPROVIDER_ROCKE_RECIPE_LLVM_FLAVOR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_recipe_authoring}/sdpa_dense.cbor" "${_recipe_authoring}/sdpa.kdp.json" "${_recipe_staging}"
        DEPENDS "${_recipe_fixture}/produce_sdpa.py" ${_recipe_authoring_sources}
            "${_recipe_fixture}/descriptors/sdpa.kdp.json.in" "${_recipe_authoring}/authoring-config.txt"
        COMMENT "Generate the bounded SDPA recipe bundle offline"
        VERBATIM)
    add_custom_target(rocke_sdpa_recipe DEPENDS "${_recipe_staging}/sdpa_dense.cbor" "${_recipe_staging}/sdpa.kdp.json"
        COMMENT "Stage the SDPA recipe")
    add_dependencies(hip_kernel_provider_impl rocke_sdpa_recipe)
    install(DIRECTORY "${_recipe_fixture}/descriptors/"
        DESTINATION "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}/rocke_sdpa"
        FILES_MATCHING PATTERN "*.json")
    install(FILES "${_recipe_staging}/sdpa_dense.cbor" "${_recipe_staging}/sdpa.kdp.json"
        DESTINATION "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}/rocke_sdpa")
endif()
