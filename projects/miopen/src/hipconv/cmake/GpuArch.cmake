function(hipconv_get_logical gpu out)
    string(REGEX REPLACE ":.*" "" logical "${gpu}")
    if(NOT logical)
        message(FATAL_ERROR "hipconv: ${gpu} empty after stripping feature flags")
    endif()
    set(${out} "${logical}" PARENT_SCOPE)
endfunction()

function(hipconv_make_logical_to_gpu hip_architectures out)
    set(stripped "")
    foreach(arch IN LISTS hip_architectures)
        hipconv_get_logical("${arch}" arch_base)
        list(APPEND stripped "${arch_base}")
    endforeach()
    set("${out}_KEYS" "${stripped}" PARENT_SCOPE)
    set("${out}_VALUES" "${hip_architectures}" PARENT_SCOPE)
    set(${out} "${out}_KEYS;${out}_VALUES" PARENT_SCOPE)
endfunction()

function(hipconv_get_matching_gpu logical_to_gpu logicals out)
    list(GET logical_to_gpu 0 keys)
    list(GET logical_to_gpu 1 values)
    set(gpus "")
    foreach(key value IN ZIP_LISTS ${keys} ${values})
        foreach(logical IN LISTS logicals)
            if(logical STREQUAL key)
                list(APPEND gpus "${value}")
                break()
            endif()
        endforeach()
    endforeach()
    set(${out} "${gpus}" PARENT_SCOPE)
endfunction()

function(hipconv_add_arch_lib name gpus)
    set(HIPCONV_ARCH_NAME "${name}")
    configure_file(
        ${HIPCONV_ROOT}/src/algorithm_registry.hpp.in
        algorithm_registry.hpp
        @ONLY
    )
    set(sources
        "direct_backend.cpp"
        "grouped_backend.cpp"
    )
    file(GLOB variant_sources CONFIGURE_DEPENDS
        "grouped/grouped_*.cpp"
        "direct/*.cpp"
        "direct/direct_l1/*.cpp"
        "direct/direct/*.cpp"
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
        HIP_ARCHITECTURES "${gpus}"
        POSITION_INDEPENDENT_CODE ON
        HIPCONV_ARCH_NAME ${name}
    )
endfunction()

function(hipconv_make_arch_registry targets out_file_name)
    set(HIPCONV_REG_DECLS "")
    set(HIPCONV_REG_ROWS "")
    foreach(target IN LISTS targets)
        get_target_property(name ${target} HIPCONV_ARCH_NAME)
        get_target_property(gpus ${target} HIP_ARCHITECTURES)
        string(APPEND HIPCONV_REG_DECLS
            "#include \"arch/${name}/algorithm_registry.hpp\"\n"
        )
        foreach(gpu IN LISTS gpus)
            hipconv_get_logical("${gpu}" logical)
            string(APPEND HIPCONV_REG_ROWS
                "{ \"${logical}\", algos_${name}, },\n"
            )
        endforeach()
    endforeach()
    configure_file(
        ${HIPCONV_ROOT}/src/arch_registry.cpp.in
        ${out_file_name}
        @ONLY
    )
endfunction()
