# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# The HipKittens backend: HIPBLASLT_JIT can compile HipKittens kernel templates
# at run time. Included by the HIPBLASLT_ENABLE_JIT block of
# library/src/amd_detail/CMakeLists.txt. The HipKittens headers the templates
# include are run-time data: they are staged next to the built library and
# installed with the runtime under hipblaslt/hipkittens/<commit>. Offline
# builds set FETCHCONTENT_SOURCE_DIR_HIPKITTENS to an unpacked archive of the
# pinned commit.
option(HIPBLASLT_JIT_ENABLE_HIPKITTENS "Build the JIT HipKittens backend (developer builds)." OFF)
if(NOT HIPBLASLT_JIT_ENABLE_HIPKITTENS)
    return()
endif()
if(NOT TARGET _rocisa)
    message(FATAL_ERROR "HIPBLASLT_JIT_ENABLE_HIPKITTENS requires the local _rocisa target")
endif()

set(_hk_gfx950 OFF)
foreach(_hk_target IN LISTS GPU_TARGETS)
    if(_hk_target MATCHES "^gfx950(:|$)")
        set(_hk_gfx950 ON)
    endif()
endforeach()
if(NOT _hk_gfx950)
    message(WARNING "HIPBLASLT_JIT_ENABLE_HIPKITTENS: HipKittens has no kernel for "
                    "GPU_TARGETS (${GPU_TARGETS}); building without the HipKittens backend")
    return()
endif()

set(_hk_commit be1c91841b817c9690aa3af22ea771f8e3624bf4)
set(_hk_archive_sha256 036c8d9e4af25b1f25335567c4bcf4cb5b7f02fac6aa383a028c361aaff18b08)
set(_hk_abi 1)
string(SUBSTRING "${_hk_commit}" 0 12 _hk_short)
set(_hk_relative "hipblaslt/hipkittens/${_hk_short}")

include(FetchContent)
FetchContent_Declare(hipkittens
    URL https://github.com/HazyResearch/HipKittens/archive/${_hk_commit}.tar.gz
    URL_HASH SHA256=${_hk_archive_sha256}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR do-not-configure)
FetchContent_MakeAvailable(hipkittens)

# The headers the gfx950 kernels include.
file(GLOB_RECURSE _hk_files RELATIVE "${hipkittens_SOURCE_DIR}"
    "${hipkittens_SOURCE_DIR}/include/cdna4/*")
list(APPEND _hk_files include/kittens.cuh include/pyutils/util.cuh)
list(SORT _hk_files)

# Stages the headers and a manifest.json listing each file's size and SHA256.
set(_hk_stage "${CMAKE_CURRENT_BINARY_DIR}/hipkittens/${_hk_short}")
set(_hk_entries)
foreach(_hk_file IN LISTS _hk_files)
    set(_hk_source "${hipkittens_SOURCE_DIR}/${_hk_file}")
    file(SHA256 "${_hk_source}" _hk_hash)
    file(SIZE "${_hk_source}" _hk_size)
    get_filename_component(_hk_dir "${_hk_file}" DIRECTORY)
    file(COPY "${_hk_source}" DESTINATION "${_hk_stage}/${_hk_dir}")
    list(APPEND _hk_entries
        "    {\"path\": \"${_hk_file}\", \"size\": ${_hk_size}, \"sha256\": \"${_hk_hash}\"}")
endforeach()
list(JOIN _hk_entries ",\n" _hk_entries)
file(CONFIGURE OUTPUT "${_hk_stage}/manifest.json" @ONLY CONTENT
"{
  \"commit\": \"${_hk_commit}\",
  \"archive_sha256\": \"${_hk_archive_sha256}\",
  \"abi\": ${_hk_abi},
  \"files\": [
${_hk_entries}
  ]
}
")

add_custom_target(hipblaslt-hipkittens-headers
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${_hk_stage}" "$<TARGET_FILE_DIR:hipblaslt>/${_hk_relative}"
    VERBATIM)
add_dependencies(hipblaslt hipblaslt-hipkittens-headers)

if(WIN32)
    set(_hk_install "${CMAKE_INSTALL_BINDIR}/${_hk_relative}")
else()
    set(_hk_install "${CMAKE_INSTALL_LIBDIR}/${_hk_relative}")
endif()
rocm_install(DIRECTORY "${_hk_stage}/" DESTINATION "${_hk_install}" COMPONENT runtime)
rocm_install(FILES "${hipkittens_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}/third-party/hipkittens"
    COMPONENT runtime)

# The compiled-in resources: header manifest, kernel templates and their entries.
set(_hk_source "${PROJECT_SOURCE_DIR}/library/src/amd_detail")
set(_hk_variant_dir "${_hk_source}/hipkittens")
set(_hk_variants "${_hk_variant_dir}/gemm_bf16_tn_256x256x64_gfx950.yaml")
set(_hk_resources "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-jit-hipkittens-resources.cpp")
if(WIN32)
    set(_hk_pythonpath "$<TARGET_FILE_DIR:_rocisa>/..;${PROJECT_SOURCE_DIR}/tensilelite")
else()
    set(_hk_pythonpath "$<TARGET_FILE_DIR:_rocisa>/..:${PROJECT_SOURCE_DIR}/tensilelite")
endif()
add_custom_command(
    OUTPUT "${_hk_resources}"
    COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_hk_pythonpath}"
            "${Python_EXECUTABLE}" "${_hk_variant_dir}/make_entries.py"
            --compiler "${CMAKE_CXX_COMPILER}"
            --headers "${_hk_stage}/manifest.json"
            --output "${_hk_resources}"
            ${_hk_variants}
    DEPENDS "${_hk_variant_dir}/make_entries.py" "${_hk_variant_dir}/gemm_bf16_tn_256x256x64_gfx950.hip"
            ${_hk_variants} "${_hk_stage}/manifest.json" _rocisa
    COMMENT "Generating the JIT HipKittens entries"
    VERBATIM)
# hipblaslt is defined in another directory, which gets no rule for the output.
add_custom_target(hipblaslt-hipkittens-resources DEPENDS "${_hk_resources}")
add_dependencies(hipblaslt hipblaslt-hipkittens-resources)

target_sources(hipblaslt PRIVATE
    "${_hk_source}/hipblaslt-jit-hipkittens.cpp"
    "${_hk_resources}")
set_source_files_properties("${_hk_resources}" TARGET_DIRECTORY hipblaslt
    PROPERTIES INCLUDE_DIRECTORIES "${_hk_source}")
target_compile_definitions(hipblaslt PRIVATE
    HIPBLASLT_JIT_HIPKITTENS
    HIPBLASLT_JIT_HIPKITTENS_DIR="${_hk_relative}"
    HIPBLASLT_JIT_HIPKITTENS_FALLBACK="${CMAKE_INSTALL_FULL_LIBDIR}/${_hk_relative}")

if(HIPBLASLT_BUILD_TESTING)
    # "host" needs no gfx950 device; "gpu" and "library" run the kernels on one.
    add_executable(hipblaslt-jit-hipkittens-test
        "${PROJECT_SOURCE_DIR}/clients/tests/jit/hipkittens_backend_test.cpp"
        "${_hk_source}/hipblaslt-jit-code-object.cpp")
    target_include_directories(hipblaslt-jit-hipkittens-test PRIVATE
        "${_hk_source}" "${_hk_source}/rocblaslt/include" "${_hk_source}/rocblaslt/src/include")
    target_link_libraries(hipblaslt-jit-hipkittens-test PRIVATE
        roc::hipblaslt roc::tensilelite-host amd_comgr hip::device)
    target_compile_features(hipblaslt-jit-hipkittens-test PRIVATE cxx_std_17)
    set_target_properties(hipblaslt-jit-hipkittens-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
endif()
