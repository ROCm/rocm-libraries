# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Merge origami_nn_index fragments (tilewright + embedding_similarity) into OUTPUT.
# Invoked as: cmake -DOUTPUT=... -DFRAGMENTS="path1;path2" -P OrigamiNNMergeIndex.cmake

if(NOT OUTPUT)
    message(FATAL_ERROR "OrigamiNNMergeIndex.cmake requires -DOUTPUT=...")
endif()

# FRAGMENTS may arrive as a CMake list (semicolons) or a single space-separated string
# from add_custom_command quoting.
if(FRAGMENTS MATCHES ";")
    set(_frag_list ${FRAGMENTS})
else()
    separate_arguments(_frag_list UNIX_COMMAND "${FRAGMENTS}")
endif()

file(WRITE "${OUTPUT}" "# origami_nn_index: merged tilewright + embedding_similarity\n")
file(WRITE "${OUTPUT}" "# origami_nn_index: <logic_stem>  <backend>  <weights_manifest>\n")

foreach(_fragment IN LISTS _frag_list)
    if(NOT EXISTS "${_fragment}")
        continue()
    endif()
    file(READ "${_fragment}" _content)
    string(REPLACE "\r\n" "\n" _content "${_content}")
    string(REPLACE "\r" "\n" _content "${_content}")
    string(REPLACE "\n" ";" _lines "${_content}")
    foreach(_line IN LISTS _lines)
        string(STRIP "${_line}" _line)
        if(_line STREQUAL "")
            continue()
        endif()
        if(_line MATCHES "^#")
            continue()
        endif()
        file(APPEND "${OUTPUT}" "${_line}\n")
    endforeach()
endforeach()
