cmake_minimum_required(VERSION 3.16)
project(composable_kernel_host)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
configure_file(${CK_ROOT}/include/ck/config.h.in ${CK_ROOT}/include/ck/config.h)

find_package(ROCM)
include(ROCMInstallTargets)
include(ROCMTest)

rocm_setup_version(VERSION 1.0)

list(APPEND CMAKE_MODULE_PATH ${CK_ROOT}/cmake)
include(Embed)
file(GLOB_RECURSE KERNEL_FILES CONFIGURE_DEPENDS
    ${CK_ROOT}/include/ck/*.hpp)
# printouts fot debug purposes
# message(STATUS "KERNEL_FILES: ${KERNEL_FILES}")
# message(STATUS "RELATIVE: ${CK_ROOT}/include")
add_embed_library(ck_headers ${KERNEL_FILES} RELATIVE ${CK_ROOT}/include)

add_compile_options(-std=c++17)

file(GLOB SOURCES CONFIGURE_DEPENDS src/*.cpp)
# TODO: Use object library
add_library(ck_host STATIC ${SOURCES})
target_link_libraries(ck_host PRIVATE ck_headers)

set_target_properties(ck_host PROPERTIES 
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON)

# target_include_directories(ck_host PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# )

add_executable(ck-template-driver driver/main.cpp)
target_link_libraries(ck-template-driver ck_host)

rocm_install_targets(
    TARGETS ck_host ck_headers
    EXPORT ck_host_targets
    INCLUDE include
)
rocm_export_targets(
    EXPORT ck_host_targets
    NAMESPACE composable_kernel::
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()

