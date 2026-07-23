# Build GTest from source as a fallback when no installed GTest is found.
#
# Callers must set GTEST_TARGET to the CMake target that needs gtest.
# Set GTEST_LINK_MAIN to ON to also link gtest_main (for targets that
# do not provide their own main).

include( ExternalProject )

set( _gtest_prefix ${CMAKE_BINARY_DIR}/googletest )
set( _gtest_lib
  ${_gtest_prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX} )
set( _gtest_main_lib
  ${_gtest_prefix}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX} )

if( NOT TARGET googletest )
  ExternalProject_Add( googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${_gtest_prefix}
    BUILD_BYPRODUCTS ${_gtest_lib} ${_gtest_main_lib}
    DOWNLOAD_NO_PROGRESS YES
  )
endif()

target_include_directories( ${GTEST_TARGET} PRIVATE ${_gtest_prefix}/include )
target_link_libraries( ${GTEST_TARGET} PRIVATE ${_gtest_lib} )
if( GTEST_LINK_MAIN )
  target_link_libraries( ${GTEST_TARGET} PRIVATE ${_gtest_main_lib} )
endif()
add_dependencies( ${GTEST_TARGET} googletest )
