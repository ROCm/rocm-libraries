###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 ###############################################################################

# The rocKE kernel pack of the rocKE backend (HIPTENSOR_ENABLE_ROCKE_BACKEND=ON).
#
# At build time, library/src/backends/rocke/pack/make_rocke_pack.py imports rocKE from
# HIPTENSOR_ROCKE_SOURCE_DIR and compiles the backend's kernels for every target of
# HIPTENSOR_ROCKE_PACK_TARGETS into <build>/share/hiptensor/rocke/<target>/rocke.htpk. The packs are
# installed to share/hiptensor/rocke. With HIPTENSOR_ROCKE_PACK_DIR set, prebuilt packs from that
# directory are installed instead and no Python runs. The library never runs Python.
# The kernel selection models of HIPTENSOR_ROCKE_SELECT_MODEL_DIR are written into the packs.

set(HIPTENSOR_ROCKE_SUPPORTED_TARGETS gfx90a gfx942 gfx950)

# Targets of GPU_TARGETS with a rocKE pack: their base names (target features dropped), in
# HIPTENSOR_ROCKE_SUPPORTED_TARGETS.
function(hiptensor_rocke_pack_targets GPU_TARGET_LIST RESULT_VAR)
    set(targets)
    foreach(target IN LISTS GPU_TARGET_LIST)
        string(REGEX REPLACE ":.*$" "" base "${target}")
        list(FIND HIPTENSOR_ROCKE_SUPPORTED_TARGETS "${base}" index)
        if(NOT index EQUAL -1)
            list(APPEND targets "${base}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES targets)
    set(${RESULT_VAR} "${targets}" PARENT_SCOPE)
endfunction()

# Adds the target `hiptensor_rocke_pack` that builds (or copies) the packs, installs them, and
# defines on LIBRARY_TARGET where the runtime looks for them.
function(hiptensor_add_rocke_pack LIBRARY_TARGET)
    set(pack_root "${PROJECT_BINARY_DIR}/share/hiptensor/rocke")
    hiptensor_rocke_pack_targets("${GPU_TARGETS}" pack_targets)
    if(DEFINED HIPTENSOR_ROCKE_PACK_TARGETS AND NOT HIPTENSOR_ROCKE_PACK_TARGETS STREQUAL "")
        set(pack_targets ${HIPTENSOR_ROCKE_PACK_TARGETS})
    endif()
    if(NOT pack_targets)
        message(WARNING "HIPTENSOR_ENABLE_ROCKE_BACKEND: no target of GPU_TARGETS has a rocKE pack "
                        "(${HIPTENSOR_ROCKE_SUPPORTED_TARGETS}); the rocKE backend declines every "
                        "problem")
    endif()

    set(outputs)
    foreach(target IN LISTS pack_targets)
        list(APPEND outputs "${pack_root}/${target}/rocke.htpk")
    endforeach()
    # The JIT's recipes (one file for every target; the packs record its SHA-256)
    set(recipe_args)
    if(HIPTENSOR_ENABLE_ROCKE_JIT AND pack_targets)
        list(APPEND outputs "${pack_root}/rocke.recipes")
        set(recipe_args --jit-recipes)
    endif()

    if(HIPTENSOR_ROCKE_PACK_DIR)
        # Prebuilt packs: copied into the build tree, no Python.
        if(recipe_args)
            add_custom_command(
                OUTPUT "${pack_root}/rocke.recipes"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${pack_root}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${HIPTENSOR_ROCKE_PACK_DIR}/rocke.recipes" "${pack_root}/"
                DEPENDS "${HIPTENSOR_ROCKE_PACK_DIR}/rocke.recipes"
                COMMENT "Copying the prebuilt rocKE JIT recipes"
                VERBATIM)
        endif()
        foreach(target IN LISTS pack_targets)
            add_custom_command(
                OUTPUT "${pack_root}/${target}/rocke.htpk"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${pack_root}/${target}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${HIPTENSOR_ROCKE_PACK_DIR}/${target}/rocke.htpk"
                        "${HIPTENSOR_ROCKE_PACK_DIR}/${target}/rocke.index.json"
                        "${pack_root}/${target}/"
                DEPENDS "${HIPTENSOR_ROCKE_PACK_DIR}/${target}/rocke.htpk"
                COMMENT "Copying the prebuilt rocKE pack for ${target}"
                VERBATIM)
        endforeach()
        message(STATUS "rocKE packs: prebuilt from ${HIPTENSOR_ROCKE_PACK_DIR} (${pack_targets})")
    elseif(pack_targets)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        if(NOT EXISTS "${HIPTENSOR_ROCKE_SOURCE_DIR}/platform/python/rocke")
            message(FATAL_ERROR "HIPTENSOR_ROCKE_SOURCE_DIR=${HIPTENSOR_ROCKE_SOURCE_DIR} is not a "
                                "rocKE source tree (no platform/python/rocke)")
        endif()
        set(builder "${PROJECT_SOURCE_DIR}/library/src/backends/rocke/pack/make_rocke_pack.py")
        file(GLOB_RECURSE rocke_kernel_sources
             "${HIPTENSOR_ROCKE_SOURCE_DIR}/library/kernels/common/hiptensor_aot/*.py")
        set(arch_args)
        foreach(target IN LISTS pack_targets)
            list(APPEND arch_args --arch ${target})
        endforeach()
        set(jobs_args)
        if(HIPTENSOR_ROCKE_PACK_JOBS)
            set(jobs_args --jobs ${HIPTENSOR_ROCKE_PACK_JOBS})
        endif()
        # The kernel selection models (<dir>/<target>/<cell>.model) go into the packs' route
        # tables; a target without models selects by rocKE's static rules.
        file(GLOB_RECURSE select_models "${HIPTENSOR_ROCKE_SELECT_MODEL_DIR}/*.model")
        add_custom_command(
            OUTPUT ${outputs}
            COMMAND ${Python3_EXECUTABLE} "${builder}"
                    --rocke-src "${HIPTENSOR_ROCKE_SOURCE_DIR}"
                    ${arch_args}
                    --out "${pack_root}"
                    --cache-dir "${PROJECT_BINARY_DIR}/rocke_pack_cache"
                    --select-models "${HIPTENSOR_ROCKE_SELECT_MODEL_DIR}"
                    ${recipe_args}
                    ${jobs_args}
            DEPENDS "${builder}" ${rocke_kernel_sources} ${select_models}
            COMMENT "Building the rocKE kernel packs (${pack_targets})"
            VERBATIM)
        message(STATUS "rocKE packs: built from ${HIPTENSOR_ROCKE_SOURCE_DIR} (${pack_targets}) "
                       "with ${Python3_EXECUTABLE}")
    endif()

    add_custom_target(hiptensor_rocke_pack ALL DEPENDS ${outputs})
    add_dependencies(${LIBRARY_TARGET} hiptensor_rocke_pack)

    include(GNUInstallDirs)
    set(install_dir "${CMAKE_INSTALL_DATADIR}/hiptensor/rocke")
    file(RELATIVE_PATH relative "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
         "${CMAKE_INSTALL_PREFIX}/${install_dir}")
    target_compile_definitions(hiptensor_core PRIVATE
        HIPTENSOR_ROCKE_PACK_RELATIVE_DIR="${relative}"
        HIPTENSOR_ROCKE_PACK_INSTALL_DIR="${CMAKE_INSTALL_PREFIX}/${install_dir}"
        HIPTENSOR_ROCKE_PACK_BUILD_DIR="${pack_root}")
    foreach(target IN LISTS pack_targets)
        install(FILES "${pack_root}/${target}/rocke.htpk" "${pack_root}/${target}/rocke.index.json"
                DESTINATION "${install_dir}/${target}")
    endforeach()
    if(recipe_args)
        install(FILES "${pack_root}/rocke.recipes" DESTINATION "${install_dir}")
    endif()
endfunction()

# The JIT of the rocKE backend (HIPTENSOR_ENABLE_ROCKE_JIT=ON): the rocKE engine's recipe VM and
# LLVM lowering (rocke_core: platform/cpp/{core,portable_ir,support} of HIPTENSOR_ROCKE_SOURCE_DIR,
# a static library linked into LIBRARY_TARGET and hidden by its symbol map) and the COMGR library,
# which becomes a run-time dependency of hipTensor.
function(hiptensor_add_rocke_jit LIBRARY_TARGET CORE_TARGET)
    set(engine "${HIPTENSOR_ROCKE_SOURCE_DIR}/platform")
    if(NOT EXISTS "${engine}/cpp/portable_ir/recipe_vm.cpp")
        message(FATAL_ERROR "HIPTENSOR_ENABLE_ROCKE_JIT: ${engine} has no rocKE recipe VM "
                            "(cpp/portable_ir/recipe_vm.cpp)")
    endif()
    file(GLOB_RECURSE engine_sources CONFIGURE_DEPENDS
         "${engine}/cpp/core/*.cpp"
         "${engine}/cpp/portable_ir/*.cpp"
         "${engine}/cpp/support/*.cpp")

    # rocKE's build stamp (cpp/core/rocke_build_id.cpp includes the generated header)
    set(generated "${PROJECT_BINARY_DIR}/rocke_core_generated")
    set(build_id_header "${generated}/rocke_build_id_values.h")
    add_custom_target(
        hiptensor_rocke_build_id
        COMMAND "${CMAKE_COMMAND}"
                "-DROCKE_SOURCE_ROOT=${engine}"
                "-DROCKE_BUILD_ID_HEADER=${build_id_header}"
                "-DROCKE_ENGINE_VERSION=1.0.0+hiptensor"
                -P "${engine}/cmake/generate_rocke_build_id.cmake"
        BYPRODUCTS "${build_id_header}"
        COMMENT "Stamping the rocKE engine build id"
        VERBATIM)

    add_library(hiptensor_rocke_core STATIC ${engine_sources})
    add_dependencies(hiptensor_rocke_core hiptensor_rocke_build_id)
    set_target_properties(hiptensor_rocke_core PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    target_include_directories(hiptensor_rocke_core
        PUBLIC "${engine}/cpp/include"
        PRIVATE "${generated}")
    target_link_libraries(hiptensor_rocke_core PUBLIC ${CMAKE_DL_LIBS} Threads::Threads)
    # the engine's own warnings are not hipTensor's to fix
    target_compile_options(hiptensor_rocke_core PRIVATE -w)

    find_package(amd_comgr REQUIRED CONFIG
                 PATHS ${ROCM_PATH} /opt/rocm
                 PATH_SUFFIXES lib/cmake/amd_comgr)
    message(STATUS "rocKE JIT: engine ${engine}, COMGR ${amd_comgr_VERSION}")

    target_include_directories(${CORE_TARGET} PRIVATE "${engine}/cpp/include")
    target_link_libraries(${CORE_TARGET} PRIVATE amd_comgr)
    target_compile_definitions(${CORE_TARGET} PRIVATE HIPTENSOR_ENABLE_ROCKE_JIT)
    target_link_libraries(${LIBRARY_TARGET} PRIVATE hiptensor_rocke_core amd_comgr)
endfunction()
