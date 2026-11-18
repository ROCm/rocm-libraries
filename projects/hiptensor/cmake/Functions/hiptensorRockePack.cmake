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
# At build time, the host tool hiptensor_rocke_pack_builder (library/src/backends/rocke/pack/
# rocke_pack_builder.cpp, linked with the C++ engine of HIPTENSOR_ROCKE_SOURCE_DIR and the COMGR
# library) builds the backend's kernels with rocKE's C++ kernel builders and compiles them for
# every target of HIPTENSOR_ROCKE_PACK_TARGETS into <build>/share/hiptensor/rocke/<target>/
# rocke.htpk. The packs are installed to share/hiptensor/rocke. With HIPTENSOR_ROCKE_PACK_DIR set,
# prebuilt packs from that directory are installed instead and no pack is built. The kernel
# selection models of HIPTENSOR_ROCKE_SELECT_MODEL_DIR are written into the packs. Neither the
# pack build nor the library runs Python. (rocKE's tests check that the packs equal the ones
# rocKE's Python engine writes for the same sources; that check is rocKE's, not this build's.)
#
# The JIT (HIPTENSOR_ENABLE_ROCKE_JIT) builds its kernels with rocKE's C++ kernel builders linked
# into the library. Its contraction route (rocKE's tensor GETT planner) also reads the GEMM
# selection models the pack build writes next to the packs of the targets that planner serves
# (<target>/gemm_models: rocKE's model text files and an index of their SHA-256 digests).

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

# rocKE's C++ engine (rocke_core: platform/cpp of HIPTENSOR_ROCKE_SOURCE_DIR without the Python
# bindings, i.e. the kernel builders and the LLVM lowering), the static library
# hiptensor_rocke_core, and the COMGR library (imported target amd_comgr). The pack builder and the
# JIT link them; the engine is built once for both.
function(hiptensor_add_rocke_engine)
    if(TARGET hiptensor_rocke_core)
        return()
    endif()
    set(engine "${HIPTENSOR_ROCKE_SOURCE_DIR}/platform")
    # The C++ kernel builders, rocKE's tuned and AOT GETT route tables and its reduction route
    # table: data files of rocKE's Python package tree (python/rocke/dispatch/families), which the
    # engine compiles in. The rocKE tree must still carry those files. The only Python step is
    # rocKE's own arch-domain table generator below (standard library only, no rocke package).
    set(tuned_json "${engine}/python/rocke/dispatch/families/tensor_gett_tuned.json")
    set(aot_json "${engine}/python/rocke/dispatch/families/tensor_gett_aot.json")
    set(rreduce_json "${engine}/python/rocke/dispatch/families/hiptensor_rreduce.json")
    foreach(required "${engine}/cpp/include/rocke/instance_hiptensor_aot.h"
                     "${engine}/cpp/instances/common/hiptensor_aot_jit_ew.cpp"
                     "${tuned_json}"
                     "${aot_json}"
                     "${rreduce_json}")
        if(NOT EXISTS "${required}")
            message(FATAL_ERROR "HIPTENSOR_ROCKE_SOURCE_DIR=${HIPTENSOR_ROCKE_SOURCE_DIR} is not a "
                                "rocKE source tree with the C++ kernel builders and the GETT "
                                "route tables: ${required} is missing")
        endif()
    endforeach()
    file(GLOB_RECURSE engine_sources CONFIGURE_DEPENDS "${engine}/cpp/*.cpp")
    list(FILTER engine_sources EXCLUDE REGEX "/cpp/bindings/")

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

    # rocKE's tuned GETT route table (tuned_json above), compiled into the engine
    # (rocke_tensor_gett_tuned_json)
    set(tuned_source "${generated}/tensor_gett_tuned_table.cpp")
    add_custom_command(
        OUTPUT "${tuned_source}"
        COMMAND "${CMAKE_COMMAND}" "-DIN=${tuned_json}" "-DOUT=${tuned_source}"
                "-DSYMBOL=rocke_tensor_gett_tuned_json"
                -P "${engine}/cmake/rocke_embed_json.cmake"
        DEPENDS "${tuned_json}" "${engine}/cmake/rocke_embed_json.cmake"
        COMMENT "Embedding rocKE's tuned GETT route table"
        VERBATIM)

    # rocKE's GETT AOT route table (aot_json above: the product routes of fixed problems, frozen
    # for ahead-of-time kernel packs), compiled into the engine (rocke_tensor_gett_aot_json)
    set(aot_source "${generated}/tensor_gett_aot_table.cpp")
    add_custom_command(
        OUTPUT "${aot_source}"
        COMMAND "${CMAKE_COMMAND}" "-DIN=${aot_json}" "-DOUT=${aot_source}"
                "-DSYMBOL=rocke_tensor_gett_aot_json"
                -P "${engine}/cmake/rocke_embed_json.cmake"
        DEPENDS "${aot_json}" "${engine}/cmake/rocke_embed_json.cmake"
        COMMENT "Embedding rocKE's GETT AOT route table"
        VERBATIM)

    # rocKE's route table of the tuned reduction classes (rreduce_json above: the palette of the
    # pack and the tuned reduction problems), compiled into the engine
    # (rocke_htaot_rreduce_table_json)
    set(rreduce_source "${generated}/hiptensor_rreduce_table.cpp")
    add_custom_command(
        OUTPUT "${rreduce_source}"
        COMMAND "${CMAKE_COMMAND}" "-DIN=${rreduce_json}" "-DOUT=${rreduce_source}"
                "-DSYMBOL=rocke_htaot_rreduce_table_json"
                -P "${engine}/cmake/rocke_embed_json.cmake"
        DEPENDS "${rreduce_json}" "${engine}/cmake/rocke_embed_json.cmake"
        COMMENT "Embedding rocKE's reduction route table"
        VERBATIM)

    # rocKE's intrinsic arch-domain table (cpp/core/arch/domain.cpp includes
    # rocke/arch_domain_table.inc): generated, as rocKE's own build does, from the committed columns
    # python/rocke/core/arch/data/intrinsic_arch_domain.*.json by tools/gen_arch_domain_cpp.py, which
    # loads only the standard-library reader python/rocke/core/arch/domain.py (no rocke package)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(arch_domain_inc "${generated}/rocke/arch_domain_table.inc")
    file(GLOB arch_domain_columns CONFIGURE_DEPENDS
         "${engine}/python/rocke/core/arch/data/intrinsic_arch_domain.*.json")
    add_custom_command(
        OUTPUT "${arch_domain_inc}"
        COMMAND "${Python3_EXECUTABLE}" -I "${engine}/tools/gen_arch_domain_cpp.py"
                --out "${arch_domain_inc}"
        DEPENDS "${engine}/tools/gen_arch_domain_cpp.py"
                "${engine}/python/rocke/core/arch/domain.py"
                "${engine}/python/rocke/core/arch/data/arch_domain_waivers.json"
                ${arch_domain_columns}
        COMMENT "Generating rocKE's C++ arch-domain table"
        VERBATIM)

    add_library(hiptensor_rocke_core STATIC ${engine_sources} "${tuned_source}" "${aot_source}"
                "${rreduce_source}" "${arch_domain_inc}")
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
    message(STATUS "rocKE engine: ${engine}, COMGR ${amd_comgr_VERSION}")
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

    set(pack_outputs)
    foreach(target IN LISTS pack_targets)
        list(APPEND pack_outputs "${pack_root}/${target}/rocke.htpk"
                                 "${pack_root}/${target}/rocke.index.json")
    endforeach()

    if(HIPTENSOR_ROCKE_PACK_DIR)
        # Prebuilt packs: copied into the build tree (with the GEMM selection models of the
        # contraction JIT where the prebuilt tree has them)
        foreach(target IN LISTS pack_targets)
            if(IS_DIRECTORY "${HIPTENSOR_ROCKE_PACK_DIR}/${target}/gemm_models")
                file(GLOB prebuilt_models "${HIPTENSOR_ROCKE_PACK_DIR}/${target}/gemm_models/*")
                add_custom_command(
                    OUTPUT "${pack_root}/${target}/gemm_models/index.txt"
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                            "${HIPTENSOR_ROCKE_PACK_DIR}/${target}/gemm_models"
                            "${pack_root}/${target}/gemm_models"
                    DEPENDS ${prebuilt_models}
                    COMMENT "Copying the prebuilt GEMM selection models for ${target}"
                    VERBATIM)
                list(APPEND pack_outputs "${pack_root}/${target}/gemm_models/index.txt")
            endif()
            add_custom_command(
                OUTPUT "${pack_root}/${target}/rocke.htpk" "${pack_root}/${target}/rocke.index.json"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${pack_root}/${target}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${HIPTENSOR_ROCKE_PACK_DIR}/${target}/rocke.htpk"
                        "${HIPTENSOR_ROCKE_PACK_DIR}/${target}/rocke.index.json"
                        "${pack_root}/${target}/"
                DEPENDS "${HIPTENSOR_ROCKE_PACK_DIR}/${target}/rocke.htpk"
                        "${HIPTENSOR_ROCKE_PACK_DIR}/${target}/rocke.index.json"
                COMMENT "Copying the prebuilt rocKE pack for ${target}"
                VERBATIM)
        endforeach()
        message(STATUS "rocKE packs: prebuilt from ${HIPTENSOR_ROCKE_PACK_DIR} (${pack_targets})")
    elseif(pack_targets)
        hiptensor_add_rocke_engine()
        set(backend_dir "${PROJECT_SOURCE_DIR}/library/src/backends/rocke")
        add_executable(hiptensor_rocke_pack_builder
            "${backend_dir}/pack/rocke_pack_builder.cpp"
            "${backend_dir}/pack/rocke_pack_format.cpp"
            "${backend_dir}/rocke_comgr.cpp")
        set_target_properties(hiptensor_rocke_pack_builder PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF)
        target_link_libraries(hiptensor_rocke_pack_builder PRIVATE hiptensor_rocke_core amd_comgr)
        # a host tool: the device-linker option of library/src does not apply (clang warns that it
        # is unused)
        get_target_property(tool_options hiptensor_rocke_pack_builder COMPILE_OPTIONS)
        if(tool_options)
            list(REMOVE_ITEM tool_options "SHELL: -Xoffload-linker -w")
            set_target_properties(hiptensor_rocke_pack_builder PROPERTIES
                COMPILE_OPTIONS "${tool_options}")
        endif()

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
        # With the JIT: the GEMM selection models of rocKE's tensor GETT planner (data files of
        # rocKE's Python package tree, only read) go next to each pack the planner serves
        # (<target>/gemm_models, the contraction JIT's input; see rocke_gett_jit.hpp)
        set(gemm_model_args)
        set(gemm_models)
        if(HIPTENSOR_ENABLE_ROCKE_JIT)
            set(gemm_model_dir
                "${HIPTENSOR_ROCKE_SOURCE_DIR}/platform/python/rocke/dispatch/gemm/selector_models")
            if(NOT IS_DIRECTORY "${gemm_model_dir}")
                message(FATAL_ERROR "HIPTENSOR_ENABLE_ROCKE_JIT: the rocKE tree has no GEMM "
                                    "selection models (${gemm_model_dir})")
            endif()
            file(GLOB gemm_models "${gemm_model_dir}/*")
            set(gemm_model_args --gemm-models "${gemm_model_dir}")
        endif()
        add_custom_command(
            OUTPUT ${pack_outputs}
            COMMAND hiptensor_rocke_pack_builder
                    ${arch_args}
                    --out "${pack_root}"
                    --cache-dir "${PROJECT_BINARY_DIR}/rocke_pack_cache"
                    --select-models "${HIPTENSOR_ROCKE_SELECT_MODEL_DIR}"
                    ${gemm_model_args}
                    ${jobs_args}
            DEPENDS hiptensor_rocke_pack_builder ${select_models} ${gemm_models}
            COMMENT "Building the rocKE kernel packs (${pack_targets})"
            VERBATIM)
        message(STATUS "rocKE packs: built from ${HIPTENSOR_ROCKE_SOURCE_DIR} (${pack_targets})")
    endif()

    add_custom_target(hiptensor_rocke_pack ALL DEPENDS ${pack_outputs})
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
        # the contraction JIT's GEMM selection models (targets of rocKE's tensor GETT planner)
        install(DIRECTORY "${pack_root}/${target}/gemm_models"
                DESTINATION "${install_dir}/${target}"
                OPTIONAL)
    endforeach()
endfunction()

# The JIT of the rocKE backend (HIPTENSOR_ENABLE_ROCKE_JIT=ON): the rocKE engine's kernel builders
# and LLVM lowering (hiptensor_rocke_core, a static library linked into LIBRARY_TARGET and hidden
# by its symbol map) and the COMGR library, which becomes a run-time dependency of hipTensor.
function(hiptensor_add_rocke_jit LIBRARY_TARGET CORE_TARGET)
    hiptensor_add_rocke_engine()
    target_include_directories(${CORE_TARGET} PRIVATE "${HIPTENSOR_ROCKE_SOURCE_DIR}/platform/cpp/include")
    target_link_libraries(${CORE_TARGET} PRIVATE amd_comgr)
    target_compile_definitions(${CORE_TARGET} PRIVATE HIPTENSOR_ENABLE_ROCKE_JIT)
    target_link_libraries(${LIBRARY_TARGET} PRIVATE hiptensor_rocke_core amd_comgr)
endfunction()
