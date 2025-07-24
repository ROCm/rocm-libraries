# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_library(isa-spec-manager OBJECT)
target_sources(isa-spec-manager
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/isa_spec_manager/isa_xml_reader.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/isa_spec_manager/tinyxml2.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/isa_spec_manager/amdisa_utility.cpp"
)
add_library(rocroller::isa-spec-manager ALIAS isa-spec-manager)
target_include_directories(isa-spec-manager
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/isa_spec_manager"
)
add_executable(GPUArchitectureGenerator
    "${CMAKE_CURRENT_SOURCE_DIR}/source/GPUArchitectureGenerator.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/GPUArchitecture.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/GPUArchitectureLibrary.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/Assemblers/Assembler.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/Costs/Cost.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/DataTypes_F6_Utils.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/DataTypes_F8_Utils.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/DataTypes_BF16_Utils.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/Error.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/Logging.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/Scheduler.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/Utilities/Settings.cpp"
    "${PROJECT_SOURCE_DIR}/lib/source/Timer.cpp"
)

if(ROCROLLER_ENABLE_LLVM)
    target_compile_definitions(GPUArchitectureGenerator PRIVATE ROCROLLER_USE_LLVM)
    target_link_libraries(GPUArchitectureGenerator PRIVATE ${llvm_yaml_libs})
    target_include_directories(GPUArchitectureGenerator PUBLIC "${LLVM_INCLUDE_DIRS}")
else()
    target_compile_definitions(GPUArchitectureGenerator PRIVATE ROCROLLER_USE_YAML_CPP)
    target_link_libraries(GPUArchitectureGenerator PRIVATE yaml-cpp::yaml-cpp)
endif()

target_include_directories(GPUArchitectureGenerator
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/lib/include" # necessary?
)
target_compile_features(GPUArchitectureGenerator PRIVATE cxx_std_20)
target_link_libraries(GPUArchitectureGenerator
    PRIVATE
        ${msgpack_libs}
        ${llvm_libs}
        spdlog::spdlog_header_only
        hip::host
        fmt::fmt-header-only
        rocroller::isa-spec-manager
)
target_compile_definitions(GPUArchitectureGenerator
    PRIVATE
       ROCROLLER_USE_HIP
       __HIP_PLATFORM_AMD__
)

if(ROCROLLER_ENABLE_TIMERS)
    target_compile_definitions(GPUArchitectureGenerator PRIVATE ROCROLLER_ENABLE_TIMERS)
    target_compile_options(GPUArchitectureGenerator PRIVATE -pg -fno-omit-frame-pointer)
endif()

if(NOT ROCROLLER_ENABLE_PREGENERATED_ARCH_DEF)
    set(amd_gpu_mr_isa_zip "${ROCROLLER_MRISAS_DIR}/amd_gpu_mr_isa.zip")
    file(DOWNLOAD https://gpuopen.com/download/machine-readable-isa/latest ${amd_gpu_mr_isa_zip})
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/amd-mrisa.stamp"
        DEPENDS  ${amd_gpu_mr_isa_zip}
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${amd_gpu_mr_isa_zip}
        COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/amd-mrisa.stamp"
        WORKING_DIRECTORY "${ROCROLLER_MRISAS_DIR}"
    )
    add_custom_target(amd-mrisa
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/amd-mrisa.stamp"
    )
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml"
        COMMAND
            $<TARGET_FILE:GPUArchitectureGenerator>
            "${CMAKE_CURRENT_BINARY_DIR}/split_yamls/GPUArchitecture_def.yaml"
            ${CMAKE_CXX_COMPILER}
            -Y
            --split
            --xml_dir
            ${ROCROLLER_MRISAS_DIR}
        COMMAND
            $<TARGET_FILE:GPUArchitectureGenerator>
            "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml"
            ${CMAKE_CXX_COMPILER}
            -Y
            --xml_dir ${ROCROLLER_MRISAS_DIR}
        DEPENDS GPUArchitectureGenerator amd-mrisa
    )
else()
    file(GLOB YAML_FILES "${CMAKE_CURRENT_SOURCE_DIR}/pregenerated/*.yaml")
    set(YAML_ARGS "")
    foreach(YAML_FILE ${YAML_FILES})
        list(APPEND YAML_ARGS --yaml "${YAML_FILE}")
    endforeach()

    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml"
        COMMAND
            $<TARGET_FILE:GPUArchitectureGenerator>
            "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml"
            ${CMAKE_CXX_COMPILER}
            -Y ${YAML_ARGS}
        DEPENDS GPUArchitectureGenerator "${YAML_FILES}"
    )
endif()

add_custom_command(
    DEPENDS GPUArchitectureGenerator "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml"
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.msgpack"
    COMMAND
        $<TARGET_FILE:GPUArchitectureGenerator>
        "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.msgpack"
        ${CMAKE_CXX_COMPILER}
        --yaml "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml"
)

if(ROCROLLER_EMBED_ARCH_DEF)
    cmrc_add_resource_library(
        GPUArchitecture_def_library
        ALIAS rocroller::GPUArchitecture_def
        TYPE OBJECT
        NAMESPACE rocRoller
        WHENCE "${CMAKE_CURRENT_BINARY_DIR}"
        PREFIX resources
        "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.msgpack"
    )
    set_target_properties(
        GPUArchitecture_def_library PROPERTIES POSITION_INDEPENDENT_CODE ON
    )
endif()

add_custom_target(
    GPUArchitecture_def ALL
    DEPENDS
        GPUArchitecture_def_library
        "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml"
        "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.msgpack"
)
