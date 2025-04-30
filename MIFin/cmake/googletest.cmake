include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

if(GOOGLETEST_DIR)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override")
endif()

message(STATUS "Fetching GoogleTest")

list(APPEND GTEST_CMAKE_CXX_FLAGS 
     -Wno-undef
     -Wno-reserved-identifier
     -Wno-global-constructors
     -Wno-missing-noreturn
     -Wno-disabled-macro-expansion
     -Wno-used-but-marked-unused
     -Wno-switch-enum
     -Wno-zero-as-null-pointer-constant
     -Wno-unused-member-function
     -Wno-comma
     -Wno-old-style-cast
     -Wno-deprecated
     -Wno-float-equal
)
message(STATUS "Suppressing googltest warnings with flags: ${GTEST_CMAKE_CXX_FLAGS}")

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        f8d7d77c06936315286eb55f8de22cd23c188571
)

# Suppress ROCMChecks WARNING on GoogleTests
macro(rocm_check_toolchain_var var access value list_file)
  if(NOT ROCM_DISABLE_CHECKS)
    _rocm_check_toolchain_var("${var}" "${access}" "${value}" "${list_file}")
  endif()
endmacro()

# Will be necessary for windows build
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(ROCM_DISABLE_CHECKS TRUE)
FetchContent_MakeAvailable(googletest)
set(ROCM_DISABLE_CHECKS FALSE)

target_compile_options(gtest PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
target_compile_options(gtest_main PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
target_compile_options(gmock PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
target_compile_options(gmock_main PRIVATE ${GTEST_CMAKE_CXX_FLAGS})
