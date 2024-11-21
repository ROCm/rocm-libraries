# ArchitectureGenerator

FetchContent_Declare(
  isa_spec_manager
  GIT_REPOSITORY https://github.com/GPUOpen-Tools/isa_spec_manager.git
  GIT_TAG        1c6fcc8072c7a94ceedb734cf06fd8a08487c4b3 # release 1.0.0 + cstdint fix
)

FetchContent_MakeAvailable(isa_spec_manager)

FetchContent_Declare(
  mrisa_xml
  URL      https://gpuopen.com/download/machine-readable-isa/AMD_GPU_MR_ISA_XML-2024_08_22.zip
)

FetchContent_MakeAvailable(mrisa_xml)

set(isa_spec_manager_SOURCES
  ${isa_spec_manager_SOURCE_DIR}/source/common/isa_xml_reader.cpp
  ${isa_spec_manager_SOURCE_DIR}/source/third_party/tinyxml2/tinyxml2.cpp
  ${isa_spec_manager_SOURCE_DIR}/source/common/amdisa_utility.cpp
)

set(isa_spec_manager_INCLUDES
  ${isa_spec_manager_SOURCE_DIR}/include
  ${isa_spec_manager_SOURCE_DIR}/source/common
  ${isa_spec_manager_SOURCE_DIR}/source/third_party/tinyxml2
)

add_executable(
  GPUArchitectureGenerator
  source/GPUArchitectureGenerator.cpp

  ${CMAKE_SOURCE_DIR}/lib/source/GPUArchitecture.cpp
  ${CMAKE_SOURCE_DIR}/lib/source/GPUArchitectureLibrary.cpp

  ${CMAKE_SOURCE_DIR}/lib/source/Assemblers/Assembler.cpp
  ${CMAKE_SOURCE_DIR}/lib/source/Costs/Cost.cpp
  ${CMAKE_SOURCE_DIR}/lib/source/DataTypes_F8_Utils.cpp
  ${CMAKE_SOURCE_DIR}/lib/source/DataTypes_BF16_Utils.cpp
  ${CMAKE_SOURCE_DIR}/lib/source/Error.cpp
  ${CMAKE_SOURCE_DIR}/lib/source/Logging.cpp
  ${CMAKE_SOURCE_DIR}/lib/source/Scheduler.cpp
  ${ROCROLLER_DEFAULT_ARCHITECTURE_CPP_FILE}
  ${isa_spec_manager_SOURCES}
)

if (DEFINED INTERNAL_MRISAS AND INTERNAL_MRISAS)
  foreach(MRISA_FILE IN LISTS INTERNAL_MRISAS)
    if(EXISTS ${MRISA_FILE})
      get_filename_component(mrisa_file_name "${MRISA_FILE}" NAME)
      add_custom_target(copy_${mrisa_file_name}
        COMMAND ${CMAKE_COMMAND} -E copy "${MRISA_FILE}" "${mrisa_xml_SOURCE_DIR}"
        BYPRODUCTS "${mrisa_xml_SOURCE_DIR}/${mrisa_file_name}"
        DEPENDS ${MRISA_FILE}
      )
      list(APPEND INTERNAL_MRISA_FILES "${mrisa_xml_SOURCE_DIR}/${mrisa_file_name}")
      list(APPEND INTERNAL_MRISA_TARGETS copy_${mrisa_file_name})
    else()
      message(WARNING "MRISA file ${MRISA_FILE} could not be found.")
    endif()
  endforeach(MRISA_FILE IN LISTS INTERNAL_MRISAS)
endif()

add_dependencies(GPUArchitectureGenerator spdlog)
target_compile_features(GPUArchitectureGenerator PUBLIC cxx_std_20)
target_compile_options(GPUArchitectureGenerator PUBLIC  ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(GPUArchitectureGenerator PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  GPUArchitectureGenerator PUBLIC
  ${llvm_libs}
  msgpackc-cxx
)
target_include_directories(GPUArchitectureGenerator PUBLIC
                           ${CMAKE_SOURCE_DIR}/lib/include
                           include
                           ${CMAKE_SOURCE_DIR}/lib/include/rocRoller
                           )

target_include_directories(GPUArchitectureGenerator SYSTEM PUBLIC
                           ${libdivide_SOURCE_DIR}/
                           ${LLVM_INCLUDE_DIRS}
                           ${LLD_INCLUDE_DIRS}
                           ${isa_spec_manager_INCLUDES}
                           ${spdlog_INCLUDES}
                           )

if(YAML_BACKEND STREQUAL "YAML_CPP")
  target_link_libraries(GPUArchitectureGenerator PUBLIC yaml-cpp)
endif()

if(ROCROLLER_USE_HIP)
  target_link_libraries(GPUArchitectureGenerator PUBLIC hip::host)
  target_compile_definitions(GPUArchitectureGenerator PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)
endif()

file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/source/rocRoller)

add_custom_command(
  OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/source/rocRoller/GPUArchitecture_def.yaml
  COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/GPUArchitectureGenerator ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/source/rocRoller/GPUArchitecture_def.yaml -Y --xml_dir ${mrisa_xml_SOURCE_DIR}
  DEPENDS GPUArchitectureGenerator ${INTERNAL_MRISA_TARGETS}
)

add_custom_command(
  OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/source/rocRoller/GPUArchitecture_def.msgpack
  COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/GPUArchitectureGenerator ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/source/rocRoller/GPUArchitecture_def.msgpack --xml_dir ${mrisa_xml_SOURCE_DIR}
  DEPENDS GPUArchitectureGenerator ${INTERNAL_MRISA_TARGETS}
)

if(INTERNAL_MRISA_FILES)
  set(delete_internal_command COMMAND ${CMAKE_COMMAND} -E remove ${INTERNAL_MRISA_FILES})
endif()

ADD_CUSTOM_TARGET(
  GPUArchitecture_def
  ALL
  ${delete_internal_command}
  DEPENDS
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/source/rocRoller/GPUArchitecture_def.yaml
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/source/rocRoller/GPUArchitecture_def.msgpack
)
