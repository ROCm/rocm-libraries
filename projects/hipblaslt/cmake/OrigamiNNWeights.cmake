# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Colocate origami TWREC YAML manifests and origami_nn_index next to per-arch Tensile
# libraries so ProblemPredictionLibrary can resolve models at deserialize.
#
# Weight YAML files are not tracked in git; use the external fetch script (post-PR) or
# set ORIGAMI_NN_WEIGHTS_DIR when building.

function(origami_nn_colocate_weights ARCH OUTPUT_ROOT)
    set(_target "origami-nn-colocate-${ARCH}")
    set(_origami_nn_root "${CMAKE_SOURCE_DIR}/../../shared/origami/data/nn")
    set(_tw_src "${_origami_nn_root}/tilewright/${ARCH}")
    set(_dst "${OUTPUT_ROOT}/library/${ARCH}")

    if(NOT EXISTS "${_tw_src}/origami_nn_index")
        message(STATUS "origami tilewright weights not found for ${ARCH} (skip colocate): ${_tw_src}")
        return()
    endif()

    if(NOT TARGET ${_target})
        add_custom_target(${_target})
    endif()

    add_custom_command(
        TARGET ${_target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_dst}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_tw_src}/origami_nn_index"
            "${_dst}/origami_nn_index"
        COMMENT "Colocating origami_nn_index for ${ARCH}"
    )

    file(GLOB _tw_manifests "${_tw_src}/*.tilewright.yaml")
    file(GLOB _tw_sidecars "${_tw_src}/*.tilewright.wts.yaml")
    foreach(_file IN LISTS _tw_manifests _tw_sidecars)
        add_custom_command(
            TARGET ${_target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_file}" "${_dst}/"
            COMMENT "Colocating origami tilewright weight ${ARCH}: ${_file}"
        )
    endforeach()

    set(ORIGAMI_NN_COLOCATE_TARGETS ${ORIGAMI_NN_COLOCATE_TARGETS} ${_target}
        PARENT_SCOPE)

    message(STATUS "origami tilewright weights will be colocated to ${_dst}")
endfunction()
