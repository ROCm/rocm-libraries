# Computes the intersection of TARGETS and CMAKE_HIP_ARCHITECTURES.
#
# The result takes its elements from CMAKE_HIP_ARCHITECTURES, so target features
# survive, and is deduplicated because CMake populates CMAKE_HIP_ARCHITECTURES
# from rocm_agent_enumerator, which emits one entry per installed device.
function(hipconv_intersect_gpus targets out)
    set(served "")
    foreach(gpu IN LISTS CMAKE_HIP_ARCHITECTURES)
        string(REGEX REPLACE ":.*" "" logical "${gpu}")
        if("${logical}" IN_LIST targets)
            list(APPEND served "${gpu}")
        endif()
    endforeach()
    if(served)
        list(REMOVE_DUPLICATES served)
    endif()
    set(${out} "${served}" PARENT_SCOPE)
endfunction()

# Build one architecture's kernel library.
#
# TARGETS is the set of GPUs this architecture serves. Every architecture is built
# whatever GPUs the build targets, NPI excepted; only its device code varies with
# them. See docs/multi-arch-convergence.md.
#
# Options go ahead of TARGETS, which has to be last. publish_to_miopen.sh's
# embargo guard reads an embargoed arch's GPU names straight out of its call
# site, taking everything from TARGETS to the closing paren, and refuses to
# publish on a token that is not a gfx name.
function(hipconv_add_arch_lib name)
    cmake_parse_arguments(ARG "NPI" "" "TARGETS" ${ARGN})
    if(NOT ARG_TARGETS)
        message(FATAL_ERROR "hipconv_add_arch_lib(${name}): TARGETS is required")
    endif()

    # NPI is dropped when the build serves none of its GPUs, because its registry
    # entry must not exist. Every other architecture is built regardless.
    hipconv_intersect_gpus("${ARG_TARGETS}" served_gpus)
    if(NOT served_gpus AND ARG_NPI)
        return()
    endif()

    set(HIPCONV_ARCH_NAME "${name}")
    string(TOUPPER "${name}" HIPCONV_ARCH_NAME_UPPER)

    configure_file(
        ${HIPCONV_ROOT}/src/algorithm_registry.hpp.in
        algorithm_registry.hpp
        @ONLY
    )
    set(sources
        "direct_backend.cpp"
        "grouped_backend.cpp"
        "depthwise_backend.cpp"
    )
    # One .cpp per kernel header, globbed here.
    #
    # hipconv_autoshard generates the per-config shard .cpp files for direct,
    # direct_l1 and depthwise_1d/2d_toeplitz into the build tree and adds them
    # with target_sources, so the glob sees only their host helper .cpp.
    file(GLOB variant_sources CONFIGURE_DEPENDS
        "grouped/*.cpp"
        "depthwise/*.cpp"
        "direct/*.cpp"
        "direct/direct_l1/*.cpp"
        "direct/direct/*.cpp"
        "direct/direct_wgrad/*.cpp"
    )
    list(APPEND sources "${variant_sources}")
    add_library(hipconv_arch_${name} OBJECT ${sources})
    set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
    target_include_directories(hipconv_arch_${name} PRIVATE
        "${HIPCONV_ROOT}/src"
        "${HIPCONV_ROOT}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(hipconv_arch_${name} PRIVATE hip::device)
    # Position-independent so the objects can link into libMIOpen.so
    set_target_properties(hipconv_arch_${name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        HIPCONV_ARCH_NAME ${name}
        HIPCONV_ARCH_TARGETS "${ARG_TARGETS}"
    )
    # Emit device code only for the GPUs this architecture serves.
    #
    # Which sources compile never varies, so the host code is the same in every
    # build and only the offload bundles differ, as RFC0008 sharding requires. An
    # architecture the build serves no GPU for still needs a non-empty
    # HIP_ARCHITECTURES; it takes the first GPU in the build and its bundles go
    # unused. See docs/multi-arch-convergence.md.
    set(offload_gpus "${served_gpus}")
    if(NOT offload_gpus AND CMAKE_HIP_ARCHITECTURES)
        list(GET CMAKE_HIP_ARCHITECTURES 0 offload_gpus)
    endif()
    if(offload_gpus)
        set_target_properties(hipconv_arch_${name} PROPERTIES
            HIP_ARCHITECTURES "${offload_gpus}"
        )
    endif()
    # The shipped kernels' disassembly is what hot_loop_check.py reads, so the collection
    # has to run after this library rather than only after the tests. It already did by
    # accident, since every test links this; saying so keeps a library-only build honest.
    if(TARGET collect-asm)
        add_dependencies(collect-asm hipconv_arch_${name})
    endif()
endfunction()

function(hipconv_autoshard)
    cmake_parse_arguments(ARG
        ""
        "ARCH;NAMESPACE;KERNEL_CLASS;CONFIG_TABLE;KERNEL;NUM_SHARDS;EXTRA_HIP_FLAGS"
        ""
        ${ARGN})
    set(target hipconv_arch_${ARG_ARCH})
    # An NPI architecture the build serves no GPU for has no target to shard into.
    if(NOT TARGET ${target})
        return()
    endif()
    set(autoshard_target hipconv_arch_${ARG_ARCH}_${ARG_NAMESPACE}_autoshard)
    set(autoshard_cpp ${ARG_NAMESPACE}_autoshard.cpp)
    set(autogen_target ${autoshard_target}_generate)
    set(autogen_dir ${CMAKE_CURRENT_BINARY_DIR}/autoshard)
    set(prefix ${autogen_dir}/${ARG_NAMESPACE}_)

    file(MAKE_DIRECTORY ${autogen_dir})

    if(NOT TARGET libautoshard)
        # Set the library type to STATIC explicitly.
        #
        # Without a type this follows BUILD_SHARED_LIBS, which MIOpen sets. A Windows DLL
        # exports no symbols without dllexport, so the generators fail to link against
        # make_autoshard.
        add_library(libautoshard STATIC ${HIPCONV_ROOT}/cmake/libautoshard.cpp)
        target_include_directories(libautoshard PUBLIC ${HIPCONV_ROOT}/cmake/)
        target_compile_options(libautoshard PRIVATE ${HIPCONV_HOST_WARNING_FLAGS})
    endif()

    configure_file(
        ${HIPCONV_ROOT}/cmake/autoshard.cpp.in
        ${autoshard_cpp}
        @ONLY
    )
    add_executable(${autoshard_target} ${autoshard_cpp})
    target_link_libraries(${autoshard_target} libautoshard)
    target_include_directories(${autoshard_target} PRIVATE
        "${HIPCONV_ROOT}/src"
        "${HIPCONV_ROOT}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(${autoshard_target} PRIVATE ${HIPCONV_HOST_WARNING_FLAGS})

    set(generated_files "${prefix}kernel_table.cpp")
    math(EXPR num_shards_1 "${ARG_NUM_SHARDS}-1")
    foreach(i RANGE ${num_shards_1})
        list(APPEND generated_files "${prefix}shard${i}.cpp")
    endforeach()

    add_custom_command(OUTPUT ${generated_files}
                      COMMAND ${autoshard_target} ${autogen_dir}
                      DEPENDS ${autoshard_target})

    target_sources(${target} PRIVATE ${generated_files})
    set_source_files_properties(${generated_files} PROPERTIES LANGUAGE HIP)
    if(ARG_EXTRA_HIP_FLAGS)
        set_source_files_properties(${generated_files} PROPERTIES
            COMPILE_OPTIONS "${ARG_EXTRA_HIP_FLAGS}")
    endif()

endfunction()

# Emit the arch registry from the declared TARGETS, never from what the build
# compiles for. Whether a row's GPU has device code here is a runtime question;
# see the probe in registry.cpp.
function(hipconv_make_arch_registry targets out_file_name)
    set(HIPCONV_REG_DECLS "")
    set(HIPCONV_REG_ROWS "")
    foreach(target IN LISTS targets)
        get_target_property(name ${target} HIPCONV_ARCH_NAME)
        get_target_property(gpus ${target} HIPCONV_ARCH_TARGETS)
        string(APPEND HIPCONV_REG_DECLS
            "#include \"arch/${name}/algorithm_registry.hpp\"\n"
        )
        foreach(gpu IN LISTS gpus)
            string(APPEND HIPCONV_REG_ROWS
                "{ \"${gpu}\", algos_${name}, },\n"
            )
        endforeach()
    endforeach()
    configure_file(
        ${HIPCONV_ROOT}/src/arch_registry.cpp.in
        ${out_file_name}
        @ONLY
    )
endfunction()
