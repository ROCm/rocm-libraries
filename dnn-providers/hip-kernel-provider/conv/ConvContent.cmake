# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Which conv engine families this build serves.
#
# Each family's instance catalogs (`descriptors/conv/<family>/*.kdp.json`) are generated
# (conv/generators/) and stored in DVC, one output per catalog: git holds
# `<name>.kdp.json.dvc` beside the engine descriptors (UED, UDD, KMD, UMD) and the kernel
# sources, which stay in git. conv/REGEN.md is how they are regenerated and uploaded.
#
# A family is SERVED when the root the product pack reads holds every catalog its pointers
# name, at the size the pointer records, and at least one catalog. The provider's
# top-level CMakeLists.txt warns about an output that is missing or differs -- or with
# HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS fails -- as it does for rocKE and FlyDSL.
#
# A family that is not served ships nothing: an engine descriptor without its catalog
# would load as an engine with no kernels, which the loader drops with an error at every
# provider load. So this file withdraws `conv/<family>` from every pack
# (HKP_WITHDRAWN_FOLDERS, read by hkp_add_packaging), and src/ neither registers the
# family's native pack (HIPDNN_ENGINE_CONV_<FAMILY>, IngestorPacks.cpp) nor builds its
# device tests. The packs' host code still compiles: families share it (the implicit-GEMM
# core in ConvIgemmNative.cpp serves direct, wrw and stem_bwd too), and none of it is
# reachable without the registration.
#
# Kernel sources are embedded into the plugin at build time; catalogs and engine
# descriptors are packed into the per-arch shard and installed with it, so the same
# answer holds for the build tree and the install tree.
#
# Outputs, reset on every configure (INTERNAL implies FORCE):
#   HIPKERNELPROVIDER_CONV_FAMILIES           served families
#   HIPKERNELPROVIDER_CONV_UNSERVED_FAMILIES  families of the source tree that are not
#   HIPKERNELPROVIDER_CONV_ALL_FAMILIES_SERVED TRUE when the second list is empty

set(_conv_source_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engines/kernel_ingestor_engine/descriptors/conv")
# Read from the root the packer is actually packing, as the top-level DVC check does. A
# build pointed at another root (the superbuild CI lane packs the examples tree) serves
# only the families that root carries.
if(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT)
    set(_conv_root "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}/conv")
else()
    set(_conv_root "${HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT}/conv")
endif()

# The family set is the source tree's: one folder per family, each with its kernels/.
file(GLOB _conv_family_dirs LIST_DIRECTORIES true RELATIVE "${_conv_source_dir}"
     CONFIGURE_DEPENDS "${_conv_source_dir}/*/kernels")
set(_conv_families "")
foreach(_entry IN LISTS _conv_family_dirs)
    get_filename_component(_family "${_entry}" DIRECTORY)
    list(APPEND _conv_families "${_family}")
endforeach()
list(SORT _conv_families)

set(_conv_served "")
set(_conv_unserved "")
foreach(_family IN LISTS _conv_families)
    set(_dir "${_conv_root}/${_family}")
    set(_reason "")
    file(GLOB _pointers CONFIGURE_DEPENDS "${_dir}/*.kdp.json.dvc")
    file(GLOB _catalogs CONFIGURE_DEPENDS "${_dir}/*.kdp.json")
    foreach(_pointer IN LISTS _pointers)
        string(REGEX REPLACE "\\.dvc$" "" _catalog "${_pointer}")
        file(STRINGS "${_pointer}" _size_line REGEX "^  size: [0-9]+$")
        string(REGEX REPLACE "[^0-9]" "" _expected "${_size_line}")
        set(_actual -1)
        if(EXISTS "${_catalog}")
            file(SIZE "${_catalog}" _actual)
        endif()
        if(NOT _expected OR NOT _actual EQUAL _expected)
            get_filename_component(_name "${_catalog}" NAME)
            set(_reason "${_name} not pulled")
            break()
        endif()
    endforeach()
    if(NOT _reason AND NOT _catalogs)
        set(_reason "no catalog in ${_dir}")
    endif()
    if(_reason)
        list(APPEND _conv_unserved "${_family}")
        message(STATUS "conv: family ${_family} not served -- ${_reason}")
        # Withdraw it from every pack, so its engine descriptors do not ship without kernels.
        set_property(GLOBAL APPEND PROPERTY HKP_WITHDRAWN_FOLDERS "conv/${_family}")
    else()
        list(APPEND _conv_served "${_family}")
    endif()
endforeach()

set(HIPKERNELPROVIDER_CONV_FAMILIES "${_conv_served}" CACHE INTERNAL
    "Conv engine families whose catalogs this build packs")
set(HIPKERNELPROVIDER_CONV_UNSERVED_FAMILIES "${_conv_unserved}" CACHE INTERNAL
    "Conv engine families of the source tree whose catalogs this build does not have")
if(_conv_unserved)
    set(HIPKERNELPROVIDER_CONV_ALL_FAMILIES_SERVED FALSE CACHE INTERNAL
        "Every conv engine family is served")
    string(REPLACE ";" ", " _conv_unserved_csv "${_conv_unserved}")
    message(STATUS
        "conv: ${_conv_unserved_csv} not served; fetch with: dvc pull -r ingestor "
        "${_conv_root}/<family>/*.kdp.json.dvc (conv/REGEN.md)")
else()
    set(HIPKERNELPROVIDER_CONV_ALL_FAMILIES_SERVED TRUE CACHE INTERNAL
        "Every conv engine family is served")
endif()
string(REPLACE ";" ", " _conv_served_csv "${_conv_served}")
message(STATUS "conv: serving ${_conv_served_csv}")
