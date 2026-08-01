# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

# check_config_header_flags(): fail the CMake configure step if a public header
# uses a <PREFIX>* feature macro that is not declared (#cmakedefine) in the
# config template (*-config.h.in). This guarantees every consumer-facing flag is
# baked into the installed header, so a feature cannot silently disappear for
# users because someone forgot the #cmakedefine. Macros that are purely internal
# (never seen by consumers) are passed in EXCLUDE.
#
# check_config_header_flags(PREFIX <P> TEMPLATE <file.in> HEADERS <dir>... [EXCLUDE <macro>...])
function(check_config_header_flags)
  set(oneValueArgs PREFIX TEMPLATE)
  set(multiValueArgs HEADERS EXCLUDE)
  cmake_parse_arguments(CFF "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT CFF_PREFIX OR NOT CFF_TEMPLATE OR NOT CFF_HEADERS)
    message(FATAL_ERROR "check_config_header_flags: PREFIX, TEMPLATE and HEADERS are required")
  endif()
  if(NOT EXISTS "${CFF_TEMPLATE}")
    message(FATAL_ERROR "check_config_header_flags: config template not found: ${CFF_TEMPLATE}")
  endif()

  # --- Registered flags: "#cmakedefine <PREFIX...>" lines in the template. ---
  file(STRINGS "${CFF_TEMPLATE}" _tmpl_lines REGEX "#cmakedefine")
  set(_registered "")
  foreach(_line IN LISTS _tmpl_lines)
    string(REGEX MATCHALL "${CFF_PREFIX}[A-Z0-9_]+" _names "${_line}")
    list(APPEND _registered ${_names})
  endforeach()
  if(_registered)
    list(REMOVE_DUPLICATES _registered)
  endif()

  # --- Used flags: any "<PREFIX>*" token in the installed public headers. ---
  set(_header_files "")
  foreach(_dir IN LISTS CFF_HEADERS)
    file(GLOB_RECURSE _hs CONFIGURE_DEPENDS "${_dir}/*.h")
    list(APPEND _header_files ${_hs})
  endforeach()

  set(_used "")
  foreach(_hf IN LISTS _header_files)
    file(STRINGS "${_hf}" _hits REGEX "${CFF_PREFIX}[A-Z0-9_]+")
    foreach(_hit IN LISTS _hits)
      string(REGEX MATCHALL "${CFF_PREFIX}[A-Z0-9_]+" _names "${_hit}")
      list(APPEND _used ${_names})
    endforeach()
  endforeach()
  if(_used)
    list(REMOVE_DUPLICATES _used)
  endif()

  # --- A public-header flag that is neither registered nor excluded is a bug. ---
  set(_missing "")
  foreach(_flag IN LISTS _used)
    if(NOT _flag IN_LIST _registered AND NOT _flag IN_LIST CFF_EXCLUDE)
      list(APPEND _missing "${_flag}")
    endif()
  endforeach()

  if(_missing)
    string(REPLACE ";" "\n  - " _missing_str "${_missing}")
    message(FATAL_ERROR
      "Feature-flag guardrail failed.\n"
      "The following macro(s) gate the public headers but are not registered "
      "in\n  ${CFF_TEMPLATE}\n"
      "  - ${_missing_str}\n\n"
      "Fix one of:\n"
      "  * add a matching '#cmakedefine <FLAG>' to the config template so the "
      "flag is baked into the installed header and reaches consumers, or\n"
      "  * add <FLAG> to the EXCLUDE list of check_config_header_flags() if it "
      "is intentionally build-only (never seen by consumers).")
  endif()

  # --- Informational: registered flags not referenced by any public header. ---
  foreach(_flag IN LISTS _registered)
    if(NOT _flag IN_LIST _used)
      message(STATUS "Feature flag ${_flag} is registered in the config header "
                     "but not used in any public header (ok if reserved for clients/tests).")
    endif()
  endforeach()

  list(LENGTH _registered _n)
  message(STATUS "Feature-flag guardrail: ${_n} registered flag(s) checked against public headers under ${CFF_HEADERS}.")
endfunction()
