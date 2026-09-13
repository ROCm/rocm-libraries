# Reject the GCC-LTO/lld combination that drops version-script symbols.
function(rocm_interfaces_assert_lto_linker_supported)
  set(_lto OFF)
  if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(_lto ON)
  endif()
  foreach(_flags IN ITEMS
      "${CMAKE_CXX_FLAGS}"
      "${CMAKE_EXE_LINKER_FLAGS}"
      "${CMAKE_SHARED_LINKER_FLAGS}"
      "${CMAKE_MODULE_LINKER_FLAGS}")
    if(_flags MATCHES "-flto")
      set(_lto ON)
    endif()
  endforeach()

  set(_lld OFF)
  if(DEFINED CMAKE_LINKER_TYPE AND CMAKE_LINKER_TYPE STREQUAL "LLD")
    set(_lld ON)
  endif()
  foreach(_flags IN ITEMS
      "${CMAKE_EXE_LINKER_FLAGS}"
      "${CMAKE_SHARED_LINKER_FLAGS}"
      "${CMAKE_MODULE_LINKER_FLAGS}")
    if(_flags MATCHES "-fuse-ld=lld" OR _flags MATCHES "--ld-path=[^ \t]*lld")
      set(_lld ON)
    endif()
  endforeach()
  if(CMAKE_LINKER MATCHES "lld")
    set(_lld ON)
  endif()

  if(_lto AND _lld AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "ROCm interfaces: link-time optimization with the LLVM linker (lld) requires a "
      "Clang-family C++ compiler (amdclang++/clang++), but the active compiler is "
      "'${CMAKE_CXX_COMPILER_ID}'. lld carries no GCC LTO plugin and cannot resolve "
      "version-script symbol assignments from GCC LTO IR, which silently breaks the "
      "versioned-symbol ABI scheme (RES-03). Build with amdclang++ and ld.lld, or pair "
      "GCC with GNU ld (bfd).")
  endif()
endfunction()
