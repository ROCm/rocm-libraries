# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Installs the tilewright models of the weights directory WEIGHTS_DIR into the
# Tensile library directory LIBRARY_DIR: its models, blocklists and guards, and
# its tilewright_index. TensileLite looks a model, and the blocklist and guards
# named after it, up in the directory the library logic was loaded from, so a
# subtree's library directory holds the whole weights directory of the subtree.
#
# Removes each <model>.tilewright.blocklist and <model>.tilewright.guards of
# LIBRARY_DIR that WEIGHTS_DIR does not ship. TensileLite reads the
# blocklist, and for a model trained with kernel guards the guards, next to
# the tilewright_index of every library it ranks with a model, so a copy an
# earlier build left there would keep leaving out, or keep offering, kernels
# other than the current model was trained with.
#
# Invoked as cmake -DWEIGHTS_DIR=<dir> -DLIBRARY_DIR=<dir> -P <this file>.

cmake_minimum_required(VERSION 3.25.2)

if(NOT IS_DIRECTORY "${WEIGHTS_DIR}" OR NOT IS_DIRECTORY "${LIBRARY_DIR}")
    message(FATAL_ERROR "WEIGHTS_DIR must name a weights directory and LIBRARY_DIR a directory")
endif()
if(NOT EXISTS "${WEIGHTS_DIR}/tilewright_index")
    message(FATAL_ERROR "${WEIGHTS_DIR} holds no tilewright_index")
endif()

file(READ "${WEIGHTS_DIR}/tilewright_index" _index)

file(GLOB _shipped LIST_DIRECTORIES false "${WEIGHTS_DIR}/*.tilewright.bin"
     "${WEIGHTS_DIR}/*.tilewright.blocklist" "${WEIGHTS_DIR}/*.tilewright.guards")
set(_names)
foreach(_file IN LISTS _shipped)
    cmake_path(GET _file FILENAME _name)
    list(APPEND _names "${_name}")
    file(COPY_FILE "${_file}" "${LIBRARY_DIR}/${_name}" ONLY_IF_DIFFERENT)
endforeach()

set(_installed "")
if(EXISTS "${LIBRARY_DIR}/tilewright_index")
    file(READ "${LIBRARY_DIR}/tilewright_index" _installed)
endif()
if(NOT _installed STREQUAL _index)
    file(WRITE "${LIBRARY_DIR}/tilewright_index" "${_index}")
endif()

file(GLOB _copies LIST_DIRECTORIES false "${LIBRARY_DIR}/*.tilewright.blocklist"
     "${LIBRARY_DIR}/*.tilewright.guards")
foreach(_copy IN LISTS _copies)
    cmake_path(GET _copy FILENAME _name)
    if(NOT _name IN_LIST _names)
        file(REMOVE "${_copy}")
    endif()
endforeach()
