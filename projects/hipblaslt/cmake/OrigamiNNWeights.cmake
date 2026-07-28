# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Colocate origami TWREC + ESREC YAML manifests and origami_nn_index next to
# per-arch Tensile libraries so ProblemPredictionLibrary can resolve models at
# deserialize.

set(_ORIGAMI_NN_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(origami_nn_colocate_weights ARCH OUTPUT_ROOT)
    set(_target "origami-nn-colocate-${ARCH}")
    set(_origami_nn_root "${CMAKE_SOURCE_DIR}/../../shared/origami/data/nn")
    set(_tw_src "${_origami_nn_root}/tilewright/${ARCH}")
    set(_dst "${OUTPUT_ROOT}/library/${ARCH}")

    set(_es_src_dirs "${_origami_nn_root}/embedding_similarity/${ARCH}")
    if(ARCH STREQUAL "gfx950")
        list(APPEND _es_src_dirs "${_origami_nn_root}/embedding_similarity/gfx950_id75a3")
    endif()

    set(_index_fragments "")
    set(_has_weights FALSE)

    if(EXISTS "${_tw_src}/origami_nn_index")
        list(APPEND _index_fragments "${_tw_src}/origami_nn_index")
        set(_has_weights TRUE)
    endif()

    foreach(_es_dir IN LISTS _es_src_dirs)
        if(EXISTS "${_es_dir}/origami_nn_index")
            list(APPEND _index_fragments "${_es_dir}/origami_nn_index")
            set(_has_weights TRUE)
        endif()
    endforeach()

    if(NOT _has_weights)
        message(STATUS "origami NN weights not found for ${ARCH} (skip colocate): ${_tw_src}")
        return()
    endif()

    if(NOT TARGET ${_target})
        add_custom_target(${_target})
    endif()

    add_custom_command(
        TARGET ${_target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_dst}"
        COMMAND ${CMAKE_COMMAND}
            -DOUTPUT="${_dst}/origami_nn_index"
            -DFRAGMENTS="${_index_fragments}"
            -P "${_ORIGAMI_NN_CMAKE_DIR}/OrigamiNNMergeIndex.cmake"
        COMMENT "Colocating merged origami_nn_index for ${ARCH}"
    )

    if(EXISTS "${_tw_src}/origami_nn_index")
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
    endif()

    foreach(_es_dir IN LISTS _es_src_dirs)
        if(NOT IS_DIRECTORY "${_es_dir}")
            continue()
        endif()
        file(GLOB _es_manifests "${_es_dir}/*.embedding.yaml")
        file(GLOB _es_sidecars "${_es_dir}/*.embedding.wts.yaml")
        foreach(_file IN LISTS _es_manifests _es_sidecars)
            add_custom_command(
                TARGET ${_target}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_file}" "${_dst}/"
                COMMENT "Colocating origami ESREC weight ${ARCH}: ${_file}"
            )
        endforeach()
    endforeach()

    set(ORIGAMI_NN_COLOCATE_TARGETS ${ORIGAMI_NN_COLOCATE_TARGETS} ${_target}
        PARENT_SCOPE)

    message(STATUS "origami NN weights (tilewright + ESREC) will be colocated to ${_dst}")
endfunction()
