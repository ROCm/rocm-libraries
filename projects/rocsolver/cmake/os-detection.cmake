# ########################################################################
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
# ########################################################################

function(get_os_id OS_ID)
  set(_os_id "unknown")
  if(EXISTS "/etc/os-release")
    read_key("ID" _os_id)
  endif()
  if(_os_id STREQUAL "opensuse-leap")
    set(_os_id "sles")
  endif()
  set(${OS_ID} ${_os_id} PARENT_SCOPE)
  set(${OS_ID}_${_os_id} TRUE PARENT_SCOPE)
endfunction()

function(read_key KEYVALUE OUTPUT)
  # Finds the line with the keyvalue
  file(STRINGS /etc/os-release _keyvalue_line REGEX "^${KEYVALUE}=")

  # Remove keyvalue=
  string(REGEX REPLACE "^${KEYVALUE}=\"?(.*)" "\\1" _output "${_keyvalue_line}")

  # Remove trailing quote
  string(REGEX REPLACE "\"$" "" _output "${_output}")
  set(${OUTPUT} ${_output} PARENT_SCOPE)
endfunction()
