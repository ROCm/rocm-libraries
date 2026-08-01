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

# Set <out_var> to the de-duplicated list of "<prefix>NAME" macros found in the
# files listed after the prefix argument.
function(_collect_feature_macros out_var prefix)
  set(macros "")
  foreach(file IN LISTS ARGN)
    file(STRINGS "${file}" lines REGEX "${prefix}[A-Z0-9_]+")
    foreach(line IN LISTS lines)
      string(REGEX MATCHALL "${prefix}[A-Z0-9_]+" tokens "${line}")
      list(APPEND macros ${tokens})
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES macros)
  set(${out_var} "${macros}" PARENT_SCOPE)
endfunction()

function(check_config_header_flags)
  cmake_parse_arguments(ARG "" "PREFIX;TEMPLATE" "HEADERS;EXCLUDE" ${ARGN})
  if(NOT ARG_PREFIX OR NOT ARG_TEMPLATE OR NOT ARG_HEADERS)
    message(FATAL_ERROR "check_config_header_flags: PREFIX, TEMPLATE and HEADERS are required")
  endif()

  # Flags declared in the config template (its "#cmakedefine" lines).
  _collect_feature_macros(registered "${ARG_PREFIX}" "${ARG_TEMPLATE}")

  # Flags actually referenced by the installed public headers.
  set(public_headers "")
  foreach(dir IN LISTS ARG_HEADERS)
    file(GLOB_RECURSE found CONFIGURE_DEPENDS "${dir}/*.h")
    list(APPEND public_headers ${found})
  endforeach()
  _collect_feature_macros(used "${ARG_PREFIX}" ${public_headers})

  # A flag used by a public header but neither declared nor excluded is a bug.
  set(missing "")
  foreach(flag IN LISTS used)
    if(NOT flag IN_LIST registered AND NOT flag IN_LIST ARG_EXCLUDE)
      list(APPEND missing "${flag}")
    endif()
  endforeach()

  if(missing)
    string(REPLACE ";" "\n  - " missing_list "${missing}")
    message(FATAL_ERROR
      "Feature-flag guardrail: these macros gate a public header but are missing "
      "from ${ARG_TEMPLATE}:\n  - ${missing_list}\n"
      "Add '#cmakedefine <FLAG>' there, or pass <FLAG> in EXCLUDE if it is build-only.")
  endif()

  list(LENGTH registered count)
  message(STATUS "Feature-flag guardrail: ${count} flag(s) registered, all public headers OK.")
endfunction()
