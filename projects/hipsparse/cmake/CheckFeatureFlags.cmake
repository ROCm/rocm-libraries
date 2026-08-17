# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
# SPDX-License-Identifier: MIT
# ########################################################################

# check_config_header_flags(): fail the CMake configure step when a public header
# is out of sync with the config template (*-config.h.in). It runs two checks:
#
#   1. Every <PREFIX>* macro used in a public header must be declared
#      (#cmakedefine) in the template. Otherwise the flag never gets baked into
#      the installed header and the guarded feature silently vanishes for
#      consumers.
#   2. Every public header that uses a (non-excluded) <PREFIX>* macro must
#      #include the generated config header itself, so the macro is defined no
#      matter the include order (no reliance on transitive includes).
#
# Macros that are purely internal (never seen by consumers) go in EXCLUDE.
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

  # Name of the generated header, e.g. "hipsparse-config.h" (template minus .in).
  get_filename_component(config_header "${ARG_TEMPLATE}" NAME)
  string(REGEX REPLACE "\\.in$" "" config_header "${config_header}")

  # Flags declared in the config template (its "#cmakedefine" lines).
  _collect_feature_macros(registered "${ARG_PREFIX}" "${ARG_TEMPLATE}")

  # All installed public headers.
  set(public_headers "")
  foreach(dir IN LISTS ARG_HEADERS)
    file(GLOB_RECURSE found CONFIGURE_DEPENDS "${dir}/*.h")
    list(APPEND public_headers ${found})
  endforeach()

  set(unregistered "")
  set(missing_include "")
  foreach(header IN LISTS public_headers)
    # Consumer-facing flags this header uses (build-only EXCLUDE flags ignored).
    _collect_feature_macros(flags "${ARG_PREFIX}" "${header}")
    set(relevant "")
    foreach(flag IN LISTS flags)
      if(NOT flag IN_LIST ARG_EXCLUDE)
        list(APPEND relevant "${flag}")
      endif()
    endforeach()
    if(NOT relevant)
      continue()
    endif()

    # Check 1: every flag the header uses must be declared in the template.
    foreach(flag IN LISTS relevant)
      if(NOT flag IN_LIST registered)
        list(APPEND unregistered "${flag}")
      endif()
    endforeach()

    # Check 2: the header must include the config header itself.
    file(STRINGS "${header}" include_line REGEX "#[ \t]*include.*${config_header}")
    if(NOT include_line)
      list(APPEND missing_include "${header}")
    endif()
  endforeach()

  if(unregistered)
    list(REMOVE_DUPLICATES unregistered)
    string(REPLACE ";" "\n  - " unregistered_list "${unregistered}")
    message(FATAL_ERROR
      "Feature-flag guardrail: these macros gate a public header but are missing "
      "from ${ARG_TEMPLATE}:\n  - ${unregistered_list}\n"
      "Add '#cmakedefine <FLAG>' there, or pass <FLAG> in EXCLUDE if it is build-only.")
  endif()

  if(missing_include)
    string(REPLACE ";" "\n  - " missing_include_list "${missing_include}")
    message(FATAL_ERROR
      "Feature-flag guardrail: these public headers use a ${ARG_PREFIX}* macro but "
      "do not #include \"${config_header}\":\n  - ${missing_include_list}\n"
      "Include the config header directly so the flag is defined regardless of include order.")
  endif()

  list(LENGTH registered count)
  message(STATUS "Feature-flag guardrail: ${count} flag(s) registered, all public headers include the config header.")
endfunction()
