# Compile support sources once for executables with the same compile environment.
# Call after the executable's mode definitions and dependencies have been added.
function(rocwmma_share_test_sources TEST_TARGET GROUP)
  # The assembly workflow addresses sources through each executable's Make rules.
  if(ROCWMMA_BUILD_ASSEMBLY)
    return()
  endif()

  get_target_property(target_sources ${TEST_TARGET} SOURCES)
  set(shared_sources)
  set(shared_includes)
  foreach(source IN LISTS ARGN)
    if(source IN_LIST target_sources)
      list(APPEND shared_sources "${source}")
      get_filename_component(source_dir "${source}" DIRECTORY)
      list(APPEND shared_includes "${source_dir}")
      list(REMOVE_ITEM target_sources "${source}")
    endif()
  endforeach()
  if(NOT shared_sources)
    return()
  endif()
  list(APPEND shared_includes "${PROJECT_SOURCE_DIR}/test")
  list(REMOVE_DUPLICATES shared_includes)

  # Do not mix mode-dependent definitions (including rocBLAS and AMD SMI),
  # instrumentation, or compiler launchers between support variants.
  set(signature "${shared_sources};${shared_includes}")
  set(properties COMPILE_DEFINITIONS COMPILE_OPTIONS COMPILE_FLAGS COMPILE_FEATURES
                 CXX_STANDARD CXX_STANDARD_REQUIRED CXX_EXTENSIONS
                 CXX_COMPILER_LAUNCHER POSITION_INDEPENDENT_CODE LINK_LIBRARIES)
  foreach(property IN LISTS properties)
    get_target_property(value ${TEST_TARGET} ${property})
    set(saved_${property} "${value}")
    string(APPEND signature "|${property}=${value}")
  endforeach()
  foreach(property COMPILE_DEFINITIONS COMPILE_OPTIONS INCLUDE_DIRECTORIES)
    get_directory_property(value ${property})
    string(APPEND signature "|directory_${property}=${value}")
  endforeach()
  # Keep target names stable when CI checks out the same sources elsewhere.
  string(REPLACE "${PROJECT_SOURCE_DIR}" "<source>" signature "${signature}")
  string(REPLACE "${PROJECT_BINARY_DIR}" "<build>" signature "${signature}")
  string(SHA256 key "${signature}")
  string(SUBSTRING "${key}" 0 16 key)
  set(support_target "rocwmma_${GROUP}_${key}")
  if(NOT TARGET ${support_target})
    add_library(${support_target} OBJECT ${shared_sources})
    target_include_directories(${support_target} PRIVATE ${shared_includes})
    foreach(property IN LISTS properties)
      if(NOT "${saved_${property}}" STREQUAL "value-NOTFOUND")
        set_property(TARGET ${support_target} PROPERTY ${property} "${saved_${property}}")
      endif()
    endforeach()
  endif()
  set_property(TARGET ${TEST_TARGET} PROPERTY SOURCES "${target_sources}")
  # Object expressions preserve all support definitions, without exporting a
  # private build target as a dependency of the installed executables.
  target_sources(${TEST_TARGET} PRIVATE $<TARGET_OBJECTS:${support_target}>)
endfunction()
