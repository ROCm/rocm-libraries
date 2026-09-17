# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The plugin resolves kpack descriptors, which read their kernel code from an archive on
# disk, so nothing is embedded there. The unit-test binary resolves both kinds: it reads
# kpack descriptors from a staged archive, and for embedded_source descriptors it links
# this engine statically and embeds its own copy of the kernel sources.

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/IngestorPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelIngestorEngine.cpp
    # One file per pack, the native seam; its descriptors are installed JSON. Listed
    # explicitly rather than globbed -- CMake's own docs discourage GLOB for sources,
    # and CONFIGURE_DEPENDS does not re-run on removal under every generator.
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
    set(_recipe_fixture "${PROJECT_SOURCE_DIR}/src/integration_tests/kernel_ingestor_engine/fixtures/rocke_recipe")
    set(_recipe_staging "${HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR}/rocke_sdpa")
    file(MAKE_DIRECTORY "${_recipe_staging}")
    foreach(_descriptor IN ITEMS sdpa.ued.json sdpa.kmd.json sdpa.uhd.json
            sdpa.umd.json sdpa.udd.json sdpa.kdp.json)
        configure_file("${_recipe_fixture}/descriptors/${_descriptor}"
            "${_recipe_staging}/${_descriptor}" COPYONLY)
    endforeach()
    # Conservative authoring dependencies: imported builder/lowerer changes must
    # regenerate the bundle. This glob tracks data generation, not compiled sources.
    file(GLOB_RECURSE _recipe_authoring_sources CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/rocke/platform/python/rocke/*.py"
        "${PROJECT_SOURCE_DIR}/rocke/library/*.py")
    add_custom_command(OUTPUT "${_recipe_staging}/sdpa_dense.cbor"
        COMMAND "${CMAKE_COMMAND}" -E env ROCKE_BACKEND=python ROCKE_LLVM_FLAVOR=llvm23
            "${Python3_EXECUTABLE}" "${_recipe_fixture}/produce_sdpa.py" "${_recipe_staging}"
        DEPENDS "${_recipe_fixture}/produce_sdpa.py" ${_recipe_authoring_sources}
        COMMENT "Generate the bounded SDPA recipe bundle offline"
        VERBATIM)
    add_custom_target(rocke_sdpa_recipe DEPENDS "${_recipe_staging}/sdpa_dense.cbor"
        COMMENT "Stage the SDPA recipe")
    add_dependencies(hip_kernel_provider_impl rocke_sdpa_recipe)
    install(DIRECTORY "${_recipe_fixture}/descriptors/"
        DESTINATION "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}/rocke_sdpa")
    install(FILES "${_recipe_staging}/sdpa_dense.cbor"
        DESTINATION "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR}/rocke_sdpa")
endif()
