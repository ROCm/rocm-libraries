# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Triton AOT kernel set.
#
# The committed objects and descriptors under the provider's in-tree descriptor root,
# at <root>/Triton/, are packed by the shared packer like every other family there; the
# Triton folder is switched on and off by HIPKERNELPROVIDER_ENABLE_TRITON through
# HKP_DESCRIPTOR_FAMILIES (descriptor-packaging/cmake/HkpPackaging.cmake).
#
# Nothing here compiles Triton or imports it. This file wires three checks, each run with
# the interpreter the packer runs under; the first two fail the build:
#   * before the product pack, the committed tree is what the generator in this checkout
#     produces (gen_descriptors.py --check);
#   * after the product pack, every shipped shard carries exactly the committed objects
#     (tools/check_shards.py);
#   * the generator's own unit tests, registered with CTest.
# gen_descriptors.py, tools/check_shards.py and the unit tests each put the sibling
# descriptor-packaging/python on sys.path themselves, so none needs a PYTHONPATH.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(TRITON_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/gen_descriptors.py")
# The committed tree the generator reads and writes by default.
set(TRITON_CONTENT_DIR "${HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT}/Triton")

# ============================================================================
# Generator unit tests
# ============================================================================
# Host-only: no GPU and no triton. Tier labels come from HKP_PACK_test_categories, and
# apply_ctest_category_labels labels only the tests add_test()'d in its own directory
# scope, so it is called here.
if(HIPKERNELPROVIDER_ENABLE_TESTS)
    add_test(NAME hip-kernel-provider-triton-generator
             COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests -t .
             WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED
       AND COMMAND apply_ctest_category_labels)
        apply_ctest_category_labels("${HKP_PACK_CTEST_CATEGORIES_YAML}")
    endif()
endif()

if(NOT TARGET hkp_packaging_product)
    message(STATUS "triton: no product pack in this build; the descriptor checks are not wired.")
    return()
endif()

# ============================================================================
# The committed tree is what this checkout generates, before the pack reads it
# ============================================================================
# --check recomputes every committed descriptor and manifest from the kernel source, the
# op contract, the generator code and the pinned requirements, and fails on any
# difference. It re-runs whenever one of those inputs or the committed tree changes.
# Bytecode caches are not inputs: running the generator or its tests writes them.
file(GLOB_RECURSE _triton_check_inputs CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/kernels_src/*"
     "${CMAKE_CURRENT_SOURCE_DIR}/tritongen/*.py"
     "${TRITON_CONTENT_DIR}/*")
list(FILTER _triton_check_inputs EXCLUDE REGEX "/__pycache__/")
# The packer modules --check imports to parse objects and descriptors; an edit to them
# changes the verdict. Globbed like the pack step's own tool sources.
file(GLOB _triton_check_modules CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/../descriptor-packaging/python/hkp_pack/*.py")

set(_triton_check_stamp "${CMAKE_CURRENT_BINARY_DIR}/triton-descriptors.checked")
add_custom_command(
    OUTPUT "${_triton_check_stamp}"
    COMMAND "${Python3_EXECUTABLE}" "${TRITON_GEN_SCRIPT}" --check
    COMMAND ${CMAKE_COMMAND} -E touch "${_triton_check_stamp}"
    DEPENDS ${_triton_check_inputs}
            ${_triton_check_modules}
            "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
            "${TRITON_GEN_SCRIPT}"
    COMMENT "triton: checking the committed descriptors against the generator"
    VERBATIM
)
add_custom_target(triton_descriptor_check
    DEPENDS "${_triton_check_stamp}"
    COMMENT "Check the committed Triton descriptors against the generator")
add_dependencies(hkp_packaging_product triton_descriptor_check)

# ============================================================================
# The pack shipped what is committed, into every shard that should have it
# ============================================================================
# The arches whose product shard ships hipkernel:TritonRmsNorm. For each, the shard must
# hold one packed UKD per committed object, packed from that object and carrying its
# SHA256. Fails the build rather than leaving it to a test someone has to run.
get_property(_triton_product_arches GLOBAL PROPERTY HKP_PACK_ARCHES_product)
set(_triton_shard_arches "")
foreach(_arch IN LISTS _triton_product_arches)
    hkp_triton_rmsnorm_available(_available "${_arch}")
    if(_available)
        list(APPEND _triton_shard_arches "${_arch}")
    endif()
endforeach()

if(NOT _triton_shard_arches)
    message(STATUS "triton: the product pack ships no Triton kernels; the shard check is not wired.")
    return()
endif()

get_property(_triton_product_root GLOBAL PROPERTY HKP_PACK_OUT_ROOT_product)
add_custom_target(triton_shard_check ALL
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_shards.py"
            --content-dir "${TRITON_CONTENT_DIR}"
            --out-root "${_triton_product_root}"
            --arches ${_triton_shard_arches}
    COMMENT "triton: checking the packed shards against the committed objects"
    VERBATIM
)
add_dependencies(triton_shard_check hkp_packaging_product)

string(REPLACE ";" ", " _triton_arch_csv "${_triton_shard_arches}")
message(STATUS "triton: checking the product shards for ${_triton_arch_csv}")
