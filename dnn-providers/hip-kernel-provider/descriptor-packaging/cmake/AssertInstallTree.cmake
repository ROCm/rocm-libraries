# Staged-install assertion for the hip-kernel-provider descriptor packaging
# output. Invoked as:
#   cmake -DTREE=<staging>/<engine-dir>/arch_content/hip-kernel-provider/descriptors
#         -DARCHES=gfx942 -DEMPTY_ARCH=gfx950 -P AssertInstallTree.cmake
#
# Asserts, per populated arch, that the shipped tree contains the kpack under a
# kpack/ subfolder and KDP/generic JSON whose inline UKDs are all kpack-form;
# and that no manifest, no hsaco/hip inline UKD, and no loose .co leaked in.
# EMPTY_ARCH (optional) is an arch that prunes to nothing: its shard folder must
# be absent from the shipped tree.

if(NOT DEFINED TREE)
    message(FATAL_ERROR "TREE must be defined")
endif()
if(NOT DEFINED ARCHES)
    message(FATAL_ERROR "ARCHES must be defined")
endif()

foreach(_arch IN LISTS ARCHES)
    set(_dir "${TREE}/${_arch}")
    if(NOT IS_DIRECTORY "${_dir}")
        message(FATAL_ERROR "expected descriptor folder missing: ${_dir}")
    endif()

    set(_kpack "${_dir}/kpack/hip_kernel_provider_${_arch}.kpack")
    if(NOT EXISTS "${_kpack}")
        message(FATAL_ERROR "expected kpack missing: ${_kpack}")
    endif()

    # No manifest is emitted; a .manifest.json anywhere in the shard is a leak.
    file(GLOB_RECURSE _manifest "${_dir}/*.manifest.json")
    if(_manifest)
        message(FATAL_ERROR "manifest leaked into shipped tree: ${_manifest}")
    endif()

    # Every KDP's inline UKDs must be kpack-form; no hsaco/hip form may survive.
    file(GLOB _kdps "${_dir}/*.kdp.json")
    if(NOT _kdps)
        message(FATAL_ERROR "no KDP descriptors shipped for ${_arch}: ${_dir}")
    endif()
    foreach(_kdp IN LISTS _kdps)
        file(READ "${_kdp}" _contents)
        if(NOT _contents MATCHES "\"kind\"[ \t]*:[ \t]*\"kpack\"")
            message(FATAL_ERROR "KDP ${_kdp} has no kpack-form UKD (kind flip failed)")
        endif()
        if(_contents MATCHES "\"kind\"[ \t]*:[ \t]*\"hsaco\"")
            message(FATAL_ERROR "hsaco-form UKD leaked into shipped tree: ${_kdp}")
        endif()
        if(_contents MATCHES "\"kind\"[ \t]*:[ \t]*\"hip\"")
            message(FATAL_ERROR "hip-form UKD leaked into shipped tree: ${_kdp}")
        endif()
    endforeach()

    # The generic chain must ship: one file of each descriptor type, matched by
    # the filename type suffix rather than a fixed base name so any name works.
    foreach(_type ued umd udd kmd uhd)
        file(GLOB _generic "${_dir}/*.${_type}.json")
        if(NOT _generic)
            message(FATAL_ERROR "expected ${_type} descriptor missing under ${_dir}")
        endif()
    endforeach()

    file(GLOB _leaked_co "${_dir}/*.co" "${_dir}/kpack/*.co")
    if(_leaked_co)
        message(FATAL_ERROR "loose .co leaked into shipped tree: ${_leaked_co}")
    endif()
endforeach()

# An arch that prunes to nothing must leave no shard folder in the shipped tree.
if(DEFINED EMPTY_ARCH AND EMPTY_ARCH)
    if(IS_DIRECTORY "${TREE}/${EMPTY_ARCH}")
        message(FATAL_ERROR "empty arch leaked a shard folder: ${TREE}/${EMPTY_ARCH}")
    endif()
endif()

message(STATUS "AssertInstallTree: OK for arches: ${ARCHES}; empty: ${EMPTY_ARCH}")
