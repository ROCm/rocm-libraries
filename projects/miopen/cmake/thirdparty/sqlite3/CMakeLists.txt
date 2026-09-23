# Slim wrapper that builds the SQLite3 amalgamation downloaded by FetchContent
# and exposes the SQLite::SQLite3 imported target so the existing
# find_package(SQLite3 REQUIRED) call in projects/MIOpen/CMakeLists.txt is
# satisfied transparently in MIOPEN_STANDALONE_BUILD mode.
#
# Adapted from D:/develop/src/TheRock/third-party/sysdeps/common/sqlite3/CMakeLists.txt;
# stripped of TheRock's therock_* orchestration, version scripts, and rocm_sysdeps
# install paths.
#
# Caller (cmake/ThirdParty.cmake) sets SQLITE3_UPSTREAM_SOURCE_DIR before
# add_subdirectory().

if(NOT DEFINED SQLITE3_UPSTREAM_SOURCE_DIR)
    message(FATAL_ERROR "SQLITE3_UPSTREAM_SOURCE_DIR must be set before include/add_subdirectory of the sqlite3 wrapper")
endif()

# sqlite3.c is C; the parent MIOpen project() declares CXX only.
enable_language(C)

find_package(Threads REQUIRED)

if(WIN32)
    add_library(miopen_sqlite3 STATIC "${SQLITE3_UPSTREAM_SOURCE_DIR}/sqlite3.c")
else()
    add_library(miopen_sqlite3 "${SQLITE3_UPSTREAM_SOURCE_DIR}/sqlite3.c")
endif()

set_target_properties(miopen_sqlite3 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME sqlite3
)

target_include_directories(miopen_sqlite3
    PUBLIC
        $<BUILD_INTERFACE:${SQLITE3_UPSTREAM_SOURCE_DIR}>
)

target_link_libraries(miopen_sqlite3 PRIVATE
    ${CMAKE_DL_LIBS}
    Threads::Threads
)

# Provide the canonical SQLite::SQLite3 alias that FindSQLite3 would create.
add_library(SQLite::SQLite3 ALIAS miopen_sqlite3)

# MIOpen privately links to SQLite::SQLite3, so CMake requires miopen_sqlite3 to
# live in the same export set as MIOpen for install(EXPORT miopen-targets) to validate.
include(GNUInstallDirs)
install(TARGETS miopen_sqlite3
    EXPORT miopen-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 3510300 → 3.51.3
set(SQLite3_FOUND TRUE PARENT_SCOPE)
set(SQLite3_INCLUDE_DIR "${SQLITE3_UPSTREAM_SOURCE_DIR}" PARENT_SCOPE)
set(SQLite3_INCLUDE_DIRS "${SQLITE3_UPSTREAM_SOURCE_DIR}" PARENT_SCOPE)
set(SQLite3_LIBRARIES SQLite::SQLite3 PARENT_SCOPE)
set(SQLite3_VERSION "3.51.3" PARENT_SCOPE)
