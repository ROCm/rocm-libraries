cmake_minimum_required(VERSION 3.25)

project(rocm-library-interfaces VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ROCM_INTERFACES_BUILD_TOOLS "Build Clang-based API extraction tools" ON)
option(ROCM_INTERFACES_BUILD_EXAMPLES "Build recording providers and shadow loaders" ON)
option(ROCM_INTERFACES_BUILD_REAL_ROCBLAS_PROVIDER
  "Build the noncanonical provider backed by a canonical rocBLAS DSO" ON)
option(ROCM_INTERFACES_ENABLE_GPU_TESTS
  "Enable numerical differential tests that require a usable AMD GPU" OFF)
option(ROCM_INTERFACES_CHECK_API_DRIFT
  "Register the API-snapshot drift check as a ctest gate" ON)
set(ROCM_INTERFACES_ROCBLAS_BACKEND_SONAME "librocblas.so.5" CACHE STRING
  "Default canonical rocBLAS DSO loaded by the real providers")
set(ROCM_INTERFACES_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH
    "Path to the rocm-libraries source tree")
find_package(hip CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)

# Public headers remain owned by their projects during the spike. Stage them in
# the build tree so protocol headers consume the exact public enum/type
# declarations instead of creating parallel definitions.
set(ROCM_INTERFACES_STAGED_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/staged/include")
file(COPY "${ROCM_INTERFACES_SOURCE_ROOT}/projects/rocblas/library/include/"
  DESTINATION "${ROCM_INTERFACES_STAGED_INCLUDE}/rocblas")
file(MAKE_DIRECTORY "${ROCM_INTERFACES_STAGED_INCLUDE}/rocblas/internal")
file(WRITE "${ROCM_INTERFACES_STAGED_INCLUDE}/rocblas/internal/rocblas-export.h"
  "#ifndef ROCBLAS_EXPORT_H\n#define ROCBLAS_EXPORT_H\n#if defined(_WIN32)\n#define ROCBLAS_EXPORT __declspec(dllexport)\n#define ROCBLAS_NO_EXPORT\n#else\n#define ROCBLAS_EXPORT __attribute__((visibility(\"default\")))\n#define ROCBLAS_NO_EXPORT __attribute__((visibility(\"hidden\")))\n#endif\n#endif\n")
file(WRITE "${ROCM_INTERFACES_STAGED_INCLUDE}/rocblas/internal/rocblas-version.h"
  "#ifndef ROCBLAS_VERSION_H\n#define ROCBLAS_VERSION_H\n#define ROCBLAS_VERSION_MAJOR 0\n#define ROCBLAS_VERSION_MINOR 0\n#define ROCBLAS_VERSION_PATCH 0\n#endif\n")
file(COPY "${ROCM_INTERFACES_SOURCE_ROOT}/projects/rocrand/library/include/rocrand"
  DESTINATION "${ROCM_INTERFACES_STAGED_INCLUDE}")
file(COPY "${ROCM_INTERFACES_SOURCE_ROOT}/projects/hiprand/library/include/hiprand"
  DESTINATION "${ROCM_INTERFACES_STAGED_INCLUDE}")
file(COPY "${ROCM_INTERFACES_SOURCE_ROOT}/projects/rocsolver/library/include/rocsolver"
  DESTINATION "${ROCM_INTERFACES_STAGED_INCLUDE}")
file(COPY "${ROCM_INTERFACES_SOURCE_ROOT}/projects/hipblaslt/library/include/hipblaslt"
  DESTINATION "${ROCM_INTERFACES_STAGED_INCLUDE}")
file(COPY "${ROCM_INTERFACES_SOURCE_ROOT}/projects/hipblas/library/include/"
  DESTINATION "${ROCM_INTERFACES_STAGED_INCLUDE}/hipblas")
file(COPY "${ROCM_INTERFACES_SOURCE_ROOT}/projects/hipsolver/library/include/"
  DESTINATION "${ROCM_INTERFACES_STAGED_INCLUDE}/hipsolver")
file(COPY "${ROCM_INTERFACES_SOURCE_ROOT}/projects/hipblas-common/library/include/hipblas-common"
  DESTINATION "${ROCM_INTERFACES_STAGED_INCLUDE}")

foreach(export_header IN ITEMS
    hipblas/hipblas-export.h
    hipsolver/internal/hipsolver-export.h
    hipblaslt/hipblaslt-export.h
    rocsolver/rocsolver-export.h)
  get_filename_component(export_dir "${ROCM_INTERFACES_STAGED_INCLUDE}/${export_header}" DIRECTORY)
  file(MAKE_DIRECTORY "${export_dir}")
  string(TOUPPER "${export_header}" export_guard)
  string(REGEX REPLACE "[^A-Z0-9]" "_" export_guard "${export_guard}")
  if(export_header MATCHES "^hipblas/")
    set(export_macro HIPBLAS_EXPORT)
  elseif(export_header MATCHES "^hipsolver/")
    set(export_macro HIPSOLVER_EXPORT)
  elseif(export_header MATCHES "^hipblaslt/")
    set(export_macro HIPBLASLT_EXPORT)
  else()
    set(export_macro ROCSOLVER_EXPORT)
  endif()
  file(WRITE "${ROCM_INTERFACES_STAGED_INCLUDE}/${export_header}"
    "#ifndef ${export_guard}\n#define ${export_guard}\n#define ${export_macro}\n#endif\n")
endforeach()

set(hipblas_VERSION_MAJOR 0)
set(hipblas_VERSION_MINOR 0)
set(hipblas_VERSION_PATCH 0)
set(hipblas_VERSION_TWEAK 0)
set(hipsolver_VERSION_MAJOR 0)
set(hipsolver_VERSION_MINOR 0)
set(hipsolver_VERSION_PATCH 0)
set(hipsolver_VERSION_TWEAK 0)
set(hipblaslt_VERSION_MAJOR 0)
set(hipblaslt_VERSION_MINOR 0)
set(hipblaslt_VERSION_PATCH 0)
set(hipblaslt_VERSION_TWEAK 0)
set(rocsolver_VERSION_MAJOR 0)
set(rocsolver_VERSION_MINOR 0)
set(rocsolver_VERSION_PATCH 0)
set(rocsolver_VERSION_TWEAK 0)
set(hiprand_VERSION_NUMBER 0)
set(rocrand_VERSION 0)
configure_file(
  "${ROCM_INTERFACES_SOURCE_ROOT}/projects/rocrand/library/include/rocrand/rocrand_version.h.in"
  "${ROCM_INTERFACES_STAGED_INCLUDE}/rocrand/rocrand_version.h" @ONLY)
configure_file(
  "${ROCM_INTERFACES_SOURCE_ROOT}/projects/hiprand/library/include/hiprand/hiprand_version.h.in"
  "${ROCM_INTERFACES_STAGED_INCLUDE}/hiprand/hiprand_version.h" @ONLY)
configure_file(
  "${ROCM_INTERFACES_SOURCE_ROOT}/projects/hipblas/library/include/hipblas-version.h.in"
  "${ROCM_INTERFACES_STAGED_INCLUDE}/hipblas/hipblas-version.h" @ONLY)
configure_file(
  "${ROCM_INTERFACES_SOURCE_ROOT}/projects/hipsolver/library/include/internal/hipsolver-version.h.in"
  "${ROCM_INTERFACES_STAGED_INCLUDE}/hipsolver/internal/hipsolver-version.h" @ONLY)
configure_file(
  "${ROCM_INTERFACES_SOURCE_ROOT}/projects/hipblaslt/library/include/hipblaslt-version.h.in"
  "${ROCM_INTERFACES_STAGED_INCLUDE}/hipblaslt/hipblaslt-version.h" @ONLY)
configure_file(
  "${ROCM_INTERFACES_SOURCE_ROOT}/projects/rocsolver/library/include/rocsolver/rocsolver-version.h.in"
  "${ROCM_INTERFACES_STAGED_INCLUDE}/rocsolver/rocsolver-version.h" @ONLY)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocm_interfaces_lto_linker_guard.cmake")
rocm_interfaces_assert_lto_linker_supported()

set(ROCM_INTERFACES_SANITIZE "" CACHE STRING
    "Build the interfaces tree with a sanitizer (one of: thread, address)")
set_property(CACHE ROCM_INTERFACES_SANITIZE PROPERTY STRINGS "" thread address)
if(ROCM_INTERFACES_SANITIZE STREQUAL "thread")
  add_compile_options(-fsanitize=thread -g -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
elseif(ROCM_INTERFACES_SANITIZE STREQUAL "address")
  add_compile_options(-fsanitize=address -g -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
elseif(NOT ROCM_INTERFACES_SANITIZE STREQUAL "")
  message(FATAL_ERROR
    "ROCM_INTERFACES_SANITIZE='${ROCM_INTERFACES_SANITIZE}' is not supported "
    "(expected empty, 'thread', or 'address').")
endif()

add_library(rocm_interfaces_runtime STATIC
  runtime/src/module.cpp
  runtime/src/provider_registry.cpp
)
add_library(ROCmInterfaces::runtime ALIAS rocm_interfaces_runtime)
set_target_properties(rocm_interfaces_runtime PROPERTIES
  EXPORT_NAME runtime
  POSITION_INDEPENDENT_CODE ON
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(rocm_interfaces_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/protocols/include>
    $<BUILD_INTERFACE:${ROCM_INTERFACES_STAGED_INCLUDE}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(rocm_interfaces_runtime PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(rocm_interfaces_runtime PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(rocm_interfaces_runtime PUBLIC hip::host)

add_library(rocm_interfaces_loader STATIC
  loader/src/blas_loader.cpp
  loader/src/rand_loader.cpp
  loader/src/solver_loader.cpp
)
add_library(ROCmInterfaces::loader ALIAS rocm_interfaces_loader)
set_target_properties(rocm_interfaces_loader PROPERTIES
  EXPORT_NAME loader
  POSITION_INDEPENDENT_CODE ON
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(rocm_interfaces_loader
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/loader/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/protocols/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime/include>
    $<BUILD_INTERFACE:${ROCM_INTERFACES_STAGED_INCLUDE}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(rocm_interfaces_loader PUBLIC rocm_interfaces_runtime)
target_compile_definitions(rocm_interfaces_loader PUBLIC __HIP_PLATFORM_AMD__=1)

add_library(rocblas_narrow_loader_shadow SHARED loader/src/rocblas_shim.cpp)
target_include_directories(rocblas_narrow_loader_shadow PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/loader/include
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
  ${CMAKE_CURRENT_SOURCE_DIR}/protocols/include
  ${ROCM_INTERFACES_STAGED_INCLUDE}
)
target_link_libraries(rocblas_narrow_loader_shadow PRIVATE rocm_interfaces_loader hip::host)
target_compile_definitions(rocblas_narrow_loader_shadow PRIVATE __HIP_PLATFORM_AMD__=1)
set_target_properties(rocblas_narrow_loader_shadow PROPERTIES
  OUTPUT_NAME rocblas-loader-narrow
  VERSION 5.0.0
  SOVERSION 5
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
if(UNIX AND NOT APPLE)
  target_link_options(rocblas_narrow_loader_shadow PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/loader/rocblas_loader.map")
endif()

set(ROCM_INTERFACES_ROCBLAS_GENERATED_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/generated/rocblas")
set(ROCM_INTERFACES_ROCBLAS_GENERATED
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge_generated.h"
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge_facade.cpp"
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge_provider.cpp"
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_forwarding_provider.cpp"
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge_link_test.cpp"
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge.map"
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge.exports")
add_custom_command(
  OUTPUT ${ROCM_INTERFACES_ROCBLAS_GENERATED}
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rocblas_bridge.py"
    --snapshot "${CMAKE_CURRENT_SOURCE_DIR}/api/snapshots/rocblas.json"
    --output-dir "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rocblas_bridge.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/api/snapshots/rocblas.json"
  COMMENT "Generating the complete typed rocBLAS compatibility bridge")
add_custom_target(rocm_interfaces_rocblas_bridge_generated
  DEPENDS ${ROCM_INTERFACES_ROCBLAS_GENERATED}
  COMMENT "Ensuring generated rocBLAS compatibility sources are current")

set(ROCM_INTERFACES_ROCBLAS_CATEGORIZATION
  "${CMAKE_CURRENT_BINARY_DIR}/generated/rocblas-categorization.json")
add_custom_command(
  OUTPUT "${ROCM_INTERFACES_ROCBLAS_CATEGORIZATION}"
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/categorize_rocblas_api.py"
    --snapshot "${CMAKE_CURRENT_SOURCE_DIR}/api/snapshots/rocblas.json"
    --output "${ROCM_INTERFACES_ROCBLAS_CATEGORIZATION}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/categorize_rocblas_api.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/api/snapshots/rocblas.json"
  COMMENT "Categorizing every rocBLAS public callable")
add_custom_target(rocm-interfaces-check-rocblas-categorization
  COMMAND "${CMAKE_COMMAND}" -E compare_files
    "${CMAKE_CURRENT_SOURCE_DIR}/api/rocblas-categorization.json"
    "${ROCM_INTERFACES_ROCBLAS_CATEGORIZATION}"
  DEPENDS "${ROCM_INTERFACES_ROCBLAS_CATEGORIZATION}"
  COMMENT "Checking the exhaustive rocBLAS categorization ledger")

set(ROCM_INTERFACES_ROCBLAS_NARROW_V2_CPP
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_narrow_v2_facade.cpp")
add_custom_command(
  OUTPUT "${ROCM_INTERFACES_ROCBLAS_NARROW_V2_CPP}"
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rocblas_narrow_v2.py"
    --snapshot "${CMAKE_CURRENT_SOURCE_DIR}/api/snapshots/rocblas.json"
    --categorization "${CMAKE_CURRENT_SOURCE_DIR}/api/rocblas-categorization.json"
    --output "${ROCM_INTERFACES_ROCBLAS_NARROW_V2_CPP}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_rocblas_narrow_v2.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/api/snapshots/rocblas.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/api/rocblas-categorization.json"
  COMMENT "Generating all rocBLAS adapters over the ten-call narrow v2 protocol")

add_library(rocblas_narrow_v2_loader_shadow SHARED
  loader/src/rocblas_narrow_v2_runtime.cpp
  "${ROCM_INTERFACES_ROCBLAS_NARROW_V2_CPP}")
add_dependencies(rocblas_narrow_v2_loader_shadow rocm_interfaces_rocblas_bridge_generated)
target_include_directories(rocblas_narrow_v2_loader_shadow
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/loader/src
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/protocols/include
  PUBLIC
    $<BUILD_INTERFACE:${ROCM_INTERFACES_STAGED_INCLUDE}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(rocblas_narrow_v2_loader_shadow PRIVATE rocm_interfaces_runtime hip::host)
target_compile_definitions(rocblas_narrow_v2_loader_shadow PRIVATE __HIP_PLATFORM_AMD__=1)
target_compile_definitions(rocblas_narrow_v2_loader_shadow PRIVATE
  "ROCM_INTERFACES_DEFAULT_ROCBLAS_NARROW_V2_MANIFEST=\"${CMAKE_INSTALL_FULL_LIBDIR}/rocm/interfaces/providers/rocblas-narrow-v2-system.json\"")
set_target_properties(rocblas_narrow_v2_loader_shadow PROPERTIES
  OUTPUT_NAME rocblas-loader-narrow-v2 VERSION 5.0.0 SOVERSION 5
  C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(UNIX AND NOT APPLE)
  target_link_options(rocblas_narrow_v2_loader_shadow PRIVATE
    "LINKER:--version-script=${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge.map")
endif()

add_library(rocblas_loader_shadow SHARED
  loader/src/rocblas_bridge_runtime.cpp
  "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge_facade.cpp")
add_dependencies(rocblas_loader_shadow rocm_interfaces_rocblas_bridge_generated)
target_include_directories(rocblas_loader_shadow
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/loader/src
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/protocols/include
    ${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}
  PUBLIC
    $<BUILD_INTERFACE:${ROCM_INTERFACES_STAGED_INCLUDE}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(rocblas_loader_shadow PRIVATE rocm_interfaces_runtime PUBLIC hip::host)
target_compile_definitions(rocblas_loader_shadow PUBLIC __HIP_PLATFORM_AMD__=1)
target_compile_definitions(rocblas_loader_shadow PRIVATE
  "ROCM_INTERFACES_DEFAULT_ROCBLAS_MANIFEST=\"${CMAKE_INSTALL_FULL_LIBDIR}/rocm/interfaces/providers/rocblas-system.json\"")
set_target_properties(rocblas_loader_shadow PROPERTIES
  OUTPUT_NAME rocblas-loader
  VERSION 5.0.0
  SOVERSION 5
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(UNIX AND NOT APPLE)
  target_link_options(rocblas_loader_shadow PRIVATE
    "LINKER:--version-script=${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge.map")
endif()

if(ROCM_INTERFACES_BUILD_EXAMPLES)
  add_subdirectory(providers/recording)
endif()

if(ROCM_INTERFACES_BUILD_REAL_ROCBLAS_PROVIDER)
  if(UNIX AND NOT APPLE)
    add_subdirectory(providers/rocblas)
  else()
    message(FATAL_ERROR
      "ROCM_INTERFACES_BUILD_REAL_ROCBLAS_PROVIDER currently requires Linux/ELF")
  endif()
endif()

if(ROCM_INTERFACES_BUILD_TOOLS)
  add_subdirectory(tools/api-extract)
  if(TARGET rocm-api-extract)
    find_program(ROCM_INTERFACES_CLANG_EXECUTABLE NAMES clang
      HINTS "${LLVM_DIR}/../../../bin" "${LLVM_TOOLS_BINARY_DIR}"
      REQUIRED)
    execute_process(
      COMMAND "${ROCM_INTERFACES_CLANG_EXECUTABLE}" --print-resource-dir
      OUTPUT_VARIABLE ROCM_INTERFACES_CLANG_RESOURCE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
    get_target_property(ROCM_INTERFACES_HIP_INCLUDE_DIRS hip::amdhip64
      INTERFACE_INCLUDE_DIRECTORIES)
    add_custom_target(rocm-interfaces-api-snapshots
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_api_snapshots.py"
        --extractor $<TARGET_FILE:rocm-api-extract>
        --profiles "${CMAKE_CURRENT_SOURCE_DIR}/api/profiles.json"
        --include-root "${ROCM_INTERFACES_STAGED_INCLUDE}"
        --resource-dir "${ROCM_INTERFACES_CLANG_RESOURCE_DIR}"
        --extra-include "${ROCM_INTERFACES_HIP_INCLUDE_DIRS}"
        --output-dir "${CMAKE_CURRENT_BINARY_DIR}/api-snapshots"
      DEPENDS rocm-api-extract
      COMMENT "Extracting public math-library API snapshots with Clang")
    add_custom_target(rocm-interfaces-check-api-snapshots
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_api_snapshots.py"
        --expected "${CMAKE_CURRENT_SOURCE_DIR}/api/snapshots"
        --actual "${CMAKE_CURRENT_BINARY_DIR}/api-snapshots"
      DEPENDS
        rocm-interfaces-api-snapshots
        rocm-interfaces-check-rocblas-categorization
      COMMENT "Checking checked-in API snapshots and rocBLAS categorization for drift")
  endif()
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

install(TARGETS rocm_interfaces_runtime rocm_interfaces_loader
  EXPORT ROCmInterfacesTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rocm-interfaces
)
install(TARGETS rocblas_loader_shadow
  EXPORT ROCmInterfacesTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rocm-interfaces
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT rocm-interfaces
)
set_target_properties(rocblas_loader_shadow PROPERTIES EXPORT_NAME rocblas_loader)
install(TARGETS rocblas_narrow_v2_loader_shadow
  EXPORT ROCmInterfacesTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT rocm-interfaces
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT rocm-interfaces)
set_target_properties(rocblas_narrow_v2_loader_shadow PROPERTIES
  EXPORT_NAME rocblas_narrow_v2_loader)
install(DIRECTORY protocols/include/ runtime/include/ loader/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT rocm-interfaces
  FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY "${ROCM_INTERFACES_STAGED_INCLUDE}/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT rocm-interfaces
  FILES_MATCHING REGEX ".*\\.(h|hpp)$")
install(FILES "${ROCM_INTERFACES_ROCBLAS_GENERATED_DIR}/rocblas_bridge_generated.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm/interfaces/experimental
  COMPONENT rocm-interfaces)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCmInterfacesShadowConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ROCmInterfacesShadowConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ROCmInterfacesShadow")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ROCmInterfacesShadowConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(EXPORT ROCmInterfacesTargets
  FILE ROCmInterfacesTargets.cmake
  NAMESPACE ROCmInterfacesShadow::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ROCmInterfacesShadow
  COMPONENT rocm-interfaces
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ROCmInterfacesShadowConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ROCmInterfacesShadowConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ROCmInterfacesShadow
  COMPONENT rocm-interfaces)
