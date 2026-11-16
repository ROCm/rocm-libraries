# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Colocate origami TWREC YAML manifests and origami_nn_index next to per-arch Tensile
# libraries so ProblemPredictionLibrary can resolve models at deserialize.
#
# Weights are downloaded into shared/origami/data/nn/tilewright/<arch>/ (gfx950 today)
# and are not tracked in git. The copy runs at build time, so files fetched after
# configure are still picked up.

# Resolved at include time. Inside the function CMAKE_CURRENT_LIST_DIR names the caller's
# directory, not this file's.
set(_ORIGAMI_NN_COLOCATE_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/OrigamiNNColocate.cmake")

function(origami_nn_colocate_weights ARCH SRC OUTPUT_ROOT DEPENDENT)
    set(_target "origami-nn-colocate-${ARCH}")
    set(_dst "${OUTPUT_ROOT}/library/${ARCH}")

    add_custom_target(${_target}
        COMMAND ${CMAKE_COMMAND}
            "-DSRC=${SRC}"
            "-DDST=${_dst}"
            -P "${_ORIGAMI_NN_COLOCATE_SCRIPT}"
        COMMENT "Colocating origami tilewright weights for ${ARCH} from ${SRC}"
        VERBATIM
    )
    if(TARGET "${DEPENDENT}")
        add_dependencies("${DEPENDENT}" ${_target})
    endif()
endfunction()
