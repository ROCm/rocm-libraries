# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Colocate the origami::nn model files (shared/origami/data/nn/<backend>/<arch>/) next to the
# per-arch Tensile library so ProblemPredictionLibrary can resolve models at deserialize. The
# library directory gets one origami_nn_index: the index of each backend present, concatenated.
# The library directory is installed as a whole (hipBLASLt's runtime component), so the
# colocated files are installed next to the kernel library too.

function(origami_nn_colocate_weights ARCH OUTPUT_ROOT)
    set(_target "origami-nn-colocate-${ARCH}")
    set(_origami_nn_root "${CMAKE_SOURCE_DIR}/../../shared/origami/data/nn")
    set(_dst "${OUTPUT_ROOT}/library/${ARCH}")

    file(GLOB _indexes CONFIGURE_DEPENDS "${_origami_nn_root}/*/${ARCH}/origami_nn_index")
    list(SORT _indexes)
    if(NOT _indexes)
        message(STATUS "origami NN models not found for ${ARCH} (skip colocate): ${_origami_nn_root}")
        return()
    endif()

    set(_index "${CMAKE_CURRENT_BINARY_DIR}/origami_nn_index_${ARCH}")
    set(_index_text "")
    set(_models "")
    foreach(_file IN LISTS _indexes)
        file(READ "${_file}" _text)
        string(APPEND _index_text "${_text}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_file}")
        get_filename_component(_src "${_file}" DIRECTORY)
        file(GLOB _files CONFIGURE_DEPENDS "${_src}/*")
        foreach(_model IN LISTS _files)
            get_filename_component(_name "${_model}" NAME)
            if(NOT IS_DIRECTORY "${_model}" AND NOT _name STREQUAL "origami_nn_index"
               AND NOT _name MATCHES "^\\.")
                list(APPEND _models "${_model}")
            endif()
        endforeach()
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

    foreach(_file IN LISTS _models)
        add_custom_command(
            TARGET ${_target}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_file}" "${_dst}/"
            COMMENT "Colocating origami NN model ${ARCH}: ${_file}"
        )
    endforeach()

    set(ORIGAMI_NN_COLOCATE_TARGETS ${ORIGAMI_NN_COLOCATE_TARGETS} ${_target}
        PARENT_SCOPE)

    message(STATUS "origami NN models will be colocated to ${_dst}")
endfunction()
