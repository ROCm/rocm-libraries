# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The JIT tuning knowledge: one file per GPU target with tuned logic, extracted
# from the logic files at build time and installed next to the Tensile library.
# Included by the top-level CMakeLists.txt when HIPBLASLT_ENABLE_JIT is on.
set(_knowledge_logic "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full")
set(_knowledge_files)
foreach(_target IN LISTS GPU_TARGETS)
    string(REGEX REPLACE ":.*" "" _arch "${_target}")
    if(_arch STREQUAL "gfx942")
        set(_logic_dir "${_knowledge_logic}/aquavanjaram")
    elseif(_arch STREQUAL "gfx950" OR _arch STREQUAL "gfx1250")
        set(_logic_dir "${_knowledge_logic}/${_arch}")
    else()
        continue()
    endif()
    set(_output "${PROJECT_BINARY_DIR}/Tensile/library/${_arch}/hipblaslt-jit-knowledge-${_arch}.dat.zlib")
    if(_output IN_LIST _knowledge_files)
        continue()
    endif()
    file(GLOB_RECURSE _logic_files CONFIGURE_DEPENDS "${_logic_dir}/*.yaml")
    add_custom_command(
        OUTPUT "${_output}"
        COMMAND ${HIPBLASLT_PYTHON_COMMAND} -m Tensile.JitKnowledge
            "${_logic_dir}" "${_output}" --architecture ${_arch}
        DEPENDS ${_logic_files} "${PROJECT_SOURCE_DIR}/tensilelite/Tensile/JitKnowledge.py"
            ${HIPBLASLT_PYTHON_DEPS}
        COMMENT "Extracting JIT tuning knowledge for ${_arch}"
        VERBATIM
        USES_TERMINAL
    )
    list(APPEND _knowledge_files "${_output}")
    # A device library installs the whole Tensile/library directory instead.
    if(NOT (HIPBLASLT_ENABLE_DEVICE OR HIPBLASLT_ENABLE_ROCROLLER))
        rocm_install(
            FILES "${_output}"
            DESTINATION "${HIPBLASLT_TENSILE_LIBRARY_DIR}/library/${_arch}"
            COMPONENT runtime
        )
    endif()
endforeach()
add_custom_target(hipblaslt_jit_knowledge ALL DEPENDS ${_knowledge_files})
