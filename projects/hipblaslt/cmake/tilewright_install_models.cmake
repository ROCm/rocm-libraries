# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Installs the tilewright models of the weights directories WEIGHTS_DIRS
# into the Tensile library directory LIBRARY_DIR: their models, blocklists and
# guards, and a tilewright_index that holds the index of each directory in
# turn. TensileLite looks a model, and the blocklist and guards of a library
# stem, up in the directory the library logic was loaded from, so the
# directories whose libraries load from LIBRARY_DIR share it: a library stem
# that the indexes of two of them list, or a file that two of them ship, is an
# error.
#
# Removes each <stem>.tilewright.blocklist and <stem>.tilewright.guards of
# LIBRARY_DIR that none of the directories ships. TensileLite reads the
# blocklist, and for a model trained with kernel guards the guards, next to
# the tilewright_index of every library it ranks with a model, so a copy an
# earlier build left there would keep leaving out, or keep offering, kernels
# other than the current model was trained with.
#
# Invoked as cmake -DWEIGHTS_DIRS=<dir>[|<dir>...] -DLIBRARY_DIR=<dir> -P <this file>.
# A | separates the directories: a ; would split the argument of the custom
# command that runs this script.

cmake_minimum_required(VERSION 3.25.2)

if(NOT WEIGHTS_DIRS OR NOT IS_DIRECTORY "${LIBRARY_DIR}")
    message(FATAL_ERROR "WEIGHTS_DIRS must name weights directories and LIBRARY_DIR a directory")
endif()
string(REPLACE "|" ";" _dirs "${WEIGHTS_DIRS}")

set(_index "")
set(_stems)
set(_stem_dirs)
set(_files)
set(_names)
set(_name_dirs)
foreach(_dir IN LISTS _dirs)
    if(NOT EXISTS "${_dir}/tilewright_index")
        message(FATAL_ERROR "${_dir} holds no tilewright_index")
    endif()
    file(READ "${_dir}/tilewright_index" _text)
    if(NOT _index STREQUAL "" AND NOT _index MATCHES "\n$")
        string(APPEND _index "\n")
    endif()
    string(APPEND _index "${_text}")

    # A row is a line with two blank-separated fields before any #, the first
    # of them the library stem. The fields are matched by the characters of
    # library stems and file names, which hold no ; or brackets, so that no
    # match can split or join the elements of the list MATCHALL returns.
    string(REGEX MATCHALL "\n[ \t\r]*[A-Za-z0-9_.+-]+[ \t\r]+[A-Za-z0-9_.+-]" _rows "\n${_text}")
    foreach(_row IN LISTS _rows)
        string(REGEX REPLACE "^\n[ \t\r]*([A-Za-z0-9_.+-]+).*$" "\\1" _stem "${_row}")
        list(FIND _stems "${_stem}" _at)
        if(_at EQUAL -1)
            list(APPEND _stems "${_stem}")
            list(APPEND _stem_dirs "${_dir}")
            continue()
        endif()
        list(GET _stem_dirs ${_at} _other)
        if(NOT _other STREQUAL _dir)
            message(FATAL_ERROR "${_stem} is listed in ${_other}/tilewright_index and in "
                                "${_dir}/tilewright_index, whose models both install into "
                                "${LIBRARY_DIR}: list each library stem in one of them")
        endif()
    endforeach()

    file(GLOB _shipped LIST_DIRECTORIES false "${_dir}/*.tilewright.bin"
         "${_dir}/*.tilewright.blocklist" "${_dir}/*.tilewright.guards")
    foreach(_file IN LISTS _shipped)
        cmake_path(GET _file FILENAME _name)
        list(FIND _names "${_name}" _at)
        if(NOT _at EQUAL -1)
            list(GET _name_dirs ${_at} _other)
            message(FATAL_ERROR "${_other} and ${_dir} both ship ${_name}, and their models both "
                                "install into ${LIBRARY_DIR}")
        endif()
        list(APPEND _names "${_name}")
        list(APPEND _name_dirs "${_dir}")
        list(APPEND _files "${_file}")
    endforeach()
endforeach()

foreach(_file IN LISTS _files)
    cmake_path(GET _file FILENAME _name)
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
