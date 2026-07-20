# Provide GoogleTest via pinned FetchContent, with a system-package escape hatch.
option(RPP_TEST_USE_SYSTEM_GTEST "Use an installed GoogleTest instead of FetchContent" OFF)

if(RPP_TEST_USE_SYSTEM_GTEST)
    find_package(GTest REQUIRED)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
