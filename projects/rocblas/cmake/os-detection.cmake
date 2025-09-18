# ########################################################################
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

function (get_os_id OS_ID)
  set(_os_id "unknown")
  if (EXISTS "/etc/os-release")
    read_key("ID" _os_id)
  endif()
  if (_os_id STREQUAL "opensuse-leap")
    set(_os_id "sles")
  endif()
  set(${OS_ID} ${_os_id} PARENT_SCOPE)
  set(${OS_ID}_${_os_id} TRUE PARENT_SCOPE)
endfunction()

function (read_key KEYVALUE OUTPUT)
  #finds the line with the keyvalue
  file (STRINGS /etc/os-release _keyvalue_line REGEX "^${KEYVALUE}=")

  #remove keyvalue=
  string (REGEX REPLACE "^${KEYVALUE}=\"?(.*)" "\\1" _output "${_keyvalue_line}")

  #remove trailing quote
  string (REGEX REPLACE "\"$" "" _output "${_output}")
  set(${OUTPUT} ${_output} PARENT_SCOPE)
endfunction ()
