# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Removes each <stem>.tilewright.blocklist of LIBRARY_DIR that WEIGHTS_DIR does
# not ship. TensileLite reads the blocklist next to the tilewright_index of every
# library it ranks with a model, so a copy an earlier build left there would keep
# leaving out kernels that the current model was trained with.
#
# Invoked as cmake -DWEIGHTS_DIR=<dir> -DLIBRARY_DIR=<dir> -P <this file>.

if(NOT IS_DIRECTORY "${WEIGHTS_DIR}" OR NOT IS_DIRECTORY "${LIBRARY_DIR}")
    message(FATAL_ERROR "WEIGHTS_DIR and LIBRARY_DIR must name directories")
endif()

file(GLOB _copies LIST_DIRECTORIES false "${LIBRARY_DIR}/*.tilewright.blocklist")
foreach(_copy IN LISTS _copies)
    cmake_path(GET _copy FILENAME _name)
    if(NOT EXISTS "${WEIGHTS_DIR}/${_name}")
        file(REMOVE "${_copy}")
    endif()
endforeach()
