include_guard(GLOBAL)

# Stores the content of '_variable' in 'OLD_${_variable}' and sets '_variable'
# to '_value'. Also adds the variable name to 'OVERRIDDEN_CACHE_VARIABLES' to
# easily restore them.
#
# Example:
#   # Set 'FOO' temporarily to '123'.
#   override_variable(FOO 123)
#   # Use 'FOO'.
#   do_thing(FOO)
#   # Set 'FOO' to its original value.
#   restore_variables()
macro(override_variable _variable _value)
  list(APPEND OLD_${_variable} ${${_variable}})
  set(${_variable} ${_value})
endmacro()

# Restores the content of '_variable'. If original was unset,
# then unsets '_variable'.
macro(restore_variable _variable)
  list(POP_BACK OLD_${_variable} _value)
  set(${_variable} ${_value})
endmacro()

# Filters out unused C++ flags from a variable into a out variable. This is
# used to ignore warnings when building depedencies.
function(filter_cxx_flags_for_deps _flags_variable _out_variable)
  separate_arguments(_cxx_flags_list NATIVE_COMMAND ${${_flags_variable}})
  list(
    REMOVE_ITEM
    _cxx_flags_list /WX -Werror -Werror=pendantic -pedantic-errors
  )
  if(MSVC)
    # Remove MSVC warning flags
    list(FILTER _cxx_flags_list EXCLUDE REGEX "/[Ww]([0-4]?)(all)?")
    list(APPEND _cxx_flags_list /w)
  else()
    # Remove GCC/LLVM flags
    list(FILTER _cxx_flags_list EXCLUDE REGEX "-W(all|extra|everything)")
    list(APPEND _cxx_flags_list -w)
  endif()
  list(JOIN _cxx_flags_list " " ${_out_variable})
  set(${_out_variable} ${${_out_variable}} PARENT_SCOPE)
endfunction()

# Finds the Git executable that supports sparse checkout. Exposes 'GIT_PATH'
# and 'GIT_VERSION'.
function(find_git)
  set(REQUIRED_MIN_GIT_VERSION "2.25")
  message(STATUS "Looking for Git (>=${REQUIRED_MIN_GIT_VERSION}) ...")
  if (NOT GIT_PATH)
    find_program(system_git git PATHS /usr/bin NO_DEFAULT_PATH)
    if(NOT (${system_git} STREQUAL "system_git-NOTFOUND"))
      # We found a system installed Git.
      set(
        GIT_PATH ${system_git}
        CACHE INTERNAL "Path to the git executable"
      )
    else()
      # We found Git made available through CMake.
      find_package(Git QUIET)
      if(GIT_FOUND)
        set(
          GIT_PATH ${GIT_EXECUTABLE}
          CACHE INTERNAL "Path to the git executable"
        )
      else()
        # Everything failed, but we need Git!
        message(
          FATAL_ERROR
          "Could not locate Git to restore spare checkout of monorepo."
        )
      endif()
    endif()

    if(NOT DEFINED GIT_VERSION)
      execute_process(
        COMMAND ${GIT_PATH} "--version" OUTPUT_VARIABLE git_version_output
      )
      string(
        REGEX MATCH "([0-9]+\.[0-9]+\.[0-9]+)" GIT_VERSION ${git_version_output}
      )
      set(GIT_VERSION ${GIT_VERSION} PARENT_SCOPE)
      if (GIT_VERSION VERSION_LESS REQUIRED_MIN_GIT_VERSION)
        message(STATUS "Found git at: ${GIT_PATH}")
        message(
          FATAL_ERROR
          "Git version ${GIT_VERSION} is too old. "
          "Please upgrade to git >= 2.25"
        )
      endif()

      message(STATUS "Found git (${GIT_VERSION}) at: ${GIT_PATH}")
    endif()
  endif()
endfunction()
