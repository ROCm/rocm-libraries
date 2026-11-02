# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Colocate origami NN model files and origami_nn_index next to per-arch Tensile
# libraries so ProblemPredictionLibrary can resolve models at deserialize.
#
# TWREC (tilewright) weights are not tracked in git; use the external fetch script (post-PR) or
# set ORIGAMI_NN_WEIGHTS_DIR when building. The boost models (data/nn/boost/<arch>/*.orr) are
# tracked. The library directory is installed as a whole (hipBLASLt's runtime component), so the
# colocated files are installed next to the kernel library too.
#
# The library directory gets one origami_nn_index: the index of each backend present,
# concatenated. tilewright (TWREC YAML) needs ORIGAMI_ENABLE_NN; boost (Origami Boost,
# ORIBOOST .orr) is colocated in every build.

function(origami_nn_colocate_weights ARCH OUTPUT_ROOT)
    set(_target "origami-nn-colocate-${ARCH}")
    set(_origami_nn_root "${CMAKE_SOURCE_DIR}/../../shared/origami/data/nn")
    set(_tw_src "${_origami_nn_root}/tilewright/${ARCH}")
    set(_boost_src "${_origami_nn_root}/boost/${ARCH}")
    set(_dst "${OUTPUT_ROOT}/library/${ARCH}")

    set(_indexes "")
    if(ORIGAMI_ENABLE_NN AND EXISTS "${_tw_src}/origami_nn_index")
        list(APPEND _indexes "${_tw_src}/origami_nn_index")
    endif()
    if(EXISTS "${_boost_src}/origami_nn_index")
        list(APPEND _indexes "${_boost_src}/origami_nn_index")
    endif()
    if(NOT _indexes)
        message(STATUS "origami NN weights not found for ${ARCH} (skip colocate): ${_origami_nn_root}")
        return()
    endif()

    set(_index "${CMAKE_CURRENT_BINARY_DIR}/origami_nn_index_${ARCH}")
    set(_index_text "")
    foreach(_file IN LISTS _indexes)
        file(READ "${_file}" _text)
        string(APPEND _index_text "${_text}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_file}")
    endforeach()
    file(CONFIGURE OUTPUT "${_index}" CONTENT "${_index_text}" @ONLY)

    if(NOT TARGET ${_target})
        add_custom_target(${_target} ALL)
        if(TARGET tensilelite-device-libraries)
            add_dependencies(${_target} tensilelite-device-libraries)
        endif()
    endif()

    add_custom_command(
        TARGET ${_target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_dst}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_index}"
            "${_dst}/origami_nn_index"
        COMMENT "Colocating origami_nn_index for ${ARCH}"
    )

    set(_weights "")
    if(ORIGAMI_ENABLE_NN)
        file(GLOB _tw_manifests "${_tw_src}/*.tilewright.yaml")
        file(GLOB _tw_sidecars "${_tw_src}/*.tilewright.wts.yaml")
        list(APPEND _weights ${_tw_manifests} ${_tw_sidecars})
    endif()
    file(GLOB _boost_models "${_boost_src}/*.orr")
    list(APPEND _weights ${_boost_models})
    foreach(_file IN LISTS _weights)
        add_custom_command(
            TARGET ${_target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_file}" "${_dst}/"
            COMMENT "Colocating origami NN weight ${ARCH}: ${_file}"
        )
    endforeach()

    set(ORIGAMI_NN_COLOCATE_TARGETS ${ORIGAMI_NN_COLOCATE_TARGETS} ${_target}
        PARENT_SCOPE)

    message(STATUS "origami NN weights will be colocated to ${_dst}")
endfunction()
