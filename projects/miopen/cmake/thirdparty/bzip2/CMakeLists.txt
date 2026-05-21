# Slim wrapper that builds bzip2-1.0.8 from sources downloaded by FetchContent
# and exposes the BZip2::BZip2 imported target so the existing
# find_package(BZip2 REQUIRED) call in projects/MIOpen/CMakeLists.txt is
# satisfied transparently in MIOPEN_STANDALONE_BUILD mode.
#
# Adapted from D:/develop/src/TheRock/third-party/sysdeps/common/bzip2/CMakeLists.txt;
# stripped of TheRock's therock_* orchestration and rocm_sysdeps install paths.
#
# Caller (cmake/ThirdParty.cmake) sets BZIP2_UPSTREAM_SOURCE_DIR before
# add_subdirectory().

if(NOT DEFINED BZIP2_UPSTREAM_SOURCE_DIR)
    message(FATAL_ERROR "BZIP2_UPSTREAM_SOURCE_DIR must be set before include/add_subdirectory of the bzip2 wrapper")
endif()

# bzip2 sources are C; the parent MIOpen project() declares CXX only.
enable_language(C)

set(_bz2_srcs
    "${BZIP2_UPSTREAM_SOURCE_DIR}/blocksort.c"
    "${BZIP2_UPSTREAM_SOURCE_DIR}/huffman.c"
    "${BZIP2_UPSTREAM_SOURCE_DIR}/crctable.c"
    "${BZIP2_UPSTREAM_SOURCE_DIR}/randtable.c"
    "${BZIP2_UPSTREAM_SOURCE_DIR}/compress.c"
    "${BZIP2_UPSTREAM_SOURCE_DIR}/decompress.c"
    "${BZIP2_UPSTREAM_SOURCE_DIR}/bzlib.c"
)

# Always static: bzip2 is linked privately into MIOpen and exported via
# miopen-targets; an accidental shared bzip2 from BUILD_SHARED_LIBS=ON would
# install alongside MIOpen's own shared lib for no benefit.
add_library(miopen_bz2 STATIC ${_bz2_srcs})

set_target_properties(miopen_bz2 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME bz2
)

target_include_directories(miopen_bz2
    PUBLIC
        $<BUILD_INTERFACE:${BZIP2_UPSTREAM_SOURCE_DIR}>
)

# Provide the canonical BZip2::BZip2 alias that find_package(BZip2) would create.
add_library(BZip2::BZip2 ALIAS miopen_bz2)

# MIOpen privately links to BZip2::BZip2, so CMake requires miopen_bz2 to live
# in the same export set as MIOpen for install(EXPORT miopen-targets) to validate.
include(GNUInstallDirs)
install(TARGETS miopen_bz2
    EXPORT miopen-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Set the legacy variables that some consumers still read (FindBZip2 sets
# these when it succeeds; FetchContent's find_package redirection does not).
set(BZIP2_FOUND TRUE PARENT_SCOPE)
set(BZIP2_INCLUDE_DIR "${BZIP2_UPSTREAM_SOURCE_DIR}" PARENT_SCOPE)
set(BZIP2_INCLUDE_DIRS "${BZIP2_UPSTREAM_SOURCE_DIR}" PARENT_SCOPE)
set(BZIP2_LIBRARIES BZip2::BZip2 PARENT_SCOPE)
set(BZIP2_VERSION_STRING "1.0.8" PARENT_SCOPE)
