# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# FlyDSL kernel set.
#
# The checked-in AOT objects and their descriptors are a bundle of the
# provider's production descriptor root, at <root>/FlyDSL/, beside rocKE's. The
# shared packer packs them like every other bundle there: it resolves each
# authored `hsaco` UKD's object, packs it into the per-arch
# hip_kernel_provider_<arch>.kpack, rewrites the UKD to kpack form, and stages
# the shard into arch_content/hip-kernel-provider/, which is installed and found
# beside the plugin with no further configuration. The FlyDSL family folder is
# switched on and off by HIPKERNELPROVIDER_ENABLE_FLYDSL through
# HKP_DESCRIPTOR_FAMILIES (descriptor-packaging/cmake/HkpPackaging.cmake).
#
# Nothing here compiles, packs or installs. This file answers one question --
# does this build ship FlyDSL kernels at all -- for the engine and test gates,
# and checks before the pack that the descriptors still agree with the objects.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(FLYDSL_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/gen_descriptors.py")

# Read from the root the packer is actually packing. A build that points the
# production root elsewhere -- the superbuild CI lane packs the examples tree --
# ships no FlyDSL descriptors, and must not compile an engine whose kernels it
# does not ship.
set(FLYDSL_CONTENT_DIR "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}/FlyDSL")

# ============================================================================
# Architecture selection
# ============================================================================
# The checked-in object set is the ceiling and GPU_TARGETS is the request; the
# intersection is what ships. An empty GPU_TARGETS states no preference, so it
# takes everything checked in. The packer prunes by each pack's own `arch`, so
# this list does not decide what is packed; it decides whether the native seams
# and the tests are compiled, which must agree with it.
#
# A content directory is named for the target its objects were compiled for.
# An LLVM generic target (gfx11-generic) stands for every member of its family:
# the descriptors under it list them all and the packer ships the same bytes into
# each member's shard. Membership comes from generators/arch_families.json, the
# table the descriptor generator reads, so the two cannot disagree.
#
# GPU_TARGETS entries may carry target features (gfx942:sramecc+:xnack-). The
# content tree keys on the bare arch, so match on the part before the first colon.
set(_flydsl_families_json "${CMAKE_CURRENT_SOURCE_DIR}/generators/arch_families.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_flydsl_families_json}")
file(READ "${_flydsl_families_json}" _flydsl_families)

file(GLOB _flydsl_arch_dirs LIST_DIRECTORIES true RELATIVE "${FLYDSL_CONTENT_DIR}"
     CONFIGURE_DEPENDS "${FLYDSL_CONTENT_DIR}/*/gfx*")
set(FLYDSL_AVAILABLE_ARCHS "")
set(_flydsl_ops "")
foreach(_entry IN LISTS _flydsl_arch_dirs)
    if(NOT IS_DIRECTORY "${FLYDSL_CONTENT_DIR}/${_entry}")
        continue()
    endif()
    get_filename_component(_target "${_entry}" NAME)
    get_filename_component(_op "${_entry}" DIRECTORY)
    list(APPEND _flydsl_ops "${_op}")
    if(_target MATCHES "-generic$")
        string(JSON _members ERROR_VARIABLE _flydsl_json_error
               GET "${_flydsl_families}" "${_target}" members)
        if(_flydsl_json_error)
            message(FATAL_ERROR
                "flydsl: content directory '${_entry}' names generic target "
                "'${_target}', which generators/arch_families.json does not list")
        endif()
        string(JSON _count LENGTH "${_flydsl_families}" "${_target}" members)
        math(EXPR _last "${_count} - 1")
        # cmake-lint: disable=E1120
        foreach(_index RANGE ${_last})
            string(JSON _member GET "${_flydsl_families}" "${_target}" members ${_index})
            list(APPEND FLYDSL_AVAILABLE_ARCHS "${_member}")
            list(APPEND "_flydsl_dirs_for_${_member}" "${_target}")
            list(APPEND "_flydsl_op_archs_${_op}" "${_member}")
        endforeach()
    else()
        list(APPEND FLYDSL_AVAILABLE_ARCHS "${_target}")
        list(APPEND "_flydsl_dirs_for_${_target}" "${_target}")
        list(APPEND "_flydsl_op_archs_${_op}" "${_target}")
    endif()
endforeach()
list(REMOVE_DUPLICATES FLYDSL_AVAILABLE_ARCHS)
list(REMOVE_DUPLICATES _flydsl_ops)

set(_flydsl_archs "${FLYDSL_AVAILABLE_ARCHS}")
if(GPU_TARGETS)
    set(_flydsl_requested "")
    foreach(_target IN LISTS GPU_TARGETS)
        string(REGEX REPLACE ":.*$" "" _bare "${_target}")
        list(APPEND _flydsl_requested "${_bare}")
    endforeach()
    set(_flydsl_archs "")
    foreach(_arch IN LISTS FLYDSL_AVAILABLE_ARCHS)
        if("${_arch}" IN_LIST _flydsl_requested)
            list(APPEND _flydsl_archs "${_arch}")
        endif()
    endforeach()
endif()

if(NOT _flydsl_archs OR NOT TARGET hkp_packaging_product)
    string(REPLACE ";" ", " _flydsl_have "${FLYDSL_AVAILABLE_ARCHS}")
    string(REPLACE ";" ", " _flydsl_want "${GPU_TARGETS}")
    message(STATUS
        "flydsl: dormant -- no checked-in kernels for the requested targets in "
        "the production descriptor root (have: ${_flydsl_have}; GPU_TARGETS: "
        "${_flydsl_want}).")
    return()
endif()

# Past the dormant return, so the product pack ships FlyDSL kernels for every
# arch in the intersection. src/ and src/tests/ read these rather than the
# option, which says only that FlyDSL was asked for. INTERNAL implies FORCE, and
# the root cleared ACTIVE to FALSE before adding this directory.
set(HIPKERNELPROVIDER_FLYDSL_ACTIVE TRUE CACHE INTERNAL
    "The product pack ships FlyDSL kernels for at least one requested architecture")
set(HIPKERNELPROVIDER_FLYDSL_ARCHS "${_flydsl_archs}" CACHE INTERNAL
    "Architectures the product pack ships FlyDSL kernels for")
# Per op, since ops differ in the architectures they ship for (the RDNA and CDNA
# attention engines share none): a suite over one op's objects reads only the
# shards that carry them. Empty for an op with nothing for the requested targets.
foreach(_op IN LISTS _flydsl_ops)
    set(_flydsl_op_shipped "")
    foreach(_arch IN LISTS _flydsl_archs)
        if("${_arch}" IN_LIST "_flydsl_op_archs_${_op}")
            list(APPEND _flydsl_op_shipped "${_arch}")
        endif()
    endforeach()
    set("HIPKERNELPROVIDER_FLYDSL_ARCHS_${_op}" "${_flydsl_op_shipped}" CACHE INTERNAL
        "Architectures the product pack ships FlyDSL ${_op} kernels for")
endforeach()

# ============================================================================
# Descriptors agree with the objects, before the pack reads either
# ============================================================================
# The packer packs what a UKD names into every shard its `arch` lists, and checks
# neither the object's bytes nor the processor it was built for. So this does,
# before every pack: each object hashes to its manifest's SHA256, targets the
# processor its directory names (and, for a generic target, carries that generic
# machine in its ELF header), and each descriptor's metadata, priority, symbol and
# arch list agree with the manifest. Runs whenever any file in the FlyDSL folder
# changes, which is exactly when they can start disagreeing. One run per content
# directory a shipped arch draws on.
file(GLOB_RECURSE _flydsl_content CONFIGURE_DEPENDS "${FLYDSL_CONTENT_DIR}/*")
set(_flydsl_check_dirs "")
foreach(_arch IN LISTS _flydsl_archs)
    list(APPEND _flydsl_check_dirs ${_flydsl_dirs_for_${_arch}})
endforeach()
list(REMOVE_DUPLICATES _flydsl_check_dirs)
set(_flydsl_check_commands "")
foreach(_target IN LISTS _flydsl_check_dirs)
    list(APPEND _flydsl_check_commands
        COMMAND ${Python3_EXECUTABLE} "${FLYDSL_GEN_SCRIPT}"
                --content-dir "${FLYDSL_CONTENT_DIR}" --arch ${_target} --check
                --generic-targets-json "${HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON}")
endforeach()

set(_flydsl_check_stamp "${CMAKE_CURRENT_BINARY_DIR}/flydsl-descriptors.checked")
add_custom_command(
    OUTPUT "${_flydsl_check_stamp}"
    ${_flydsl_check_commands}
    COMMAND ${CMAKE_COMMAND} -E touch "${_flydsl_check_stamp}"
    DEPENDS ${_flydsl_content} "${FLYDSL_GEN_SCRIPT}" "${_flydsl_families_json}"
            "${HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON}"
    COMMENT "flydsl: checking descriptors against the checked-in objects"
    VERBATIM
)
add_custom_target(flydsl_descriptor_check
    DEPENDS "${_flydsl_check_stamp}"
    COMMENT "Check FlyDSL descriptors against the checked-in objects")
add_dependencies(hkp_packaging_product flydsl_descriptor_check)

# ============================================================================
# The pack shipped what is checked in, into every shard that should have it
# ============================================================================
# The other half of the check above: after the product pack, every arch in the
# intersection must have a shard holding one shipped UKD per checked-in object,
# packed from that object and carrying its SHA256. For a generic target that is
# what shows each member ships the one shared object set. Fails the build, as
# the descriptor check does, rather than leaving it to a test someone has to run.
get_property(_flydsl_product_root GLOBAL PROPERTY HKP_PACK_OUT_ROOT_product)
if(_flydsl_product_root)
    add_custom_target(flydsl_shard_check ALL
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_shards.py"
                --content-dir "${FLYDSL_CONTENT_DIR}"
                --out-root "${_flydsl_product_root}"
                --arches ${_flydsl_archs}
        COMMENT "flydsl: checking the packed shards against the checked-in objects"
        VERBATIM
    )
    add_dependencies(flydsl_shard_check hkp_packaging_product)
endif()

# The shipped objects are compiled from Apache-2.0 sources, which asks that a copy
# of the License go with them in object form.
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_LICENSES.md"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/hip-kernel-provider/flydsl"
)

string(REPLACE ";" ", " _flydsl_arch_csv "${_flydsl_archs}")
message(STATUS "flydsl: shipping through the product pack for ${_flydsl_arch_csv}")
