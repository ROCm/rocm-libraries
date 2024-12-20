# ArchitectureGenerator

# Not visible to library users, so not necessary to use system packages
rocroller_add_dependency(isa_spec_manager HASH 1c6fcc8072c7a94ceedb734cf06fd8a08487c4b3)
rocroller_add_dependency(mrisa_xml VERSION 2024_08_22)

add_executable(
  GPUArchitectureGenerator
  ${CMAKE_CURRENT_SOURCE_DIR}/source/GPUArchitectureGenerator.cpp

  ${PROJECT_SOURCE_DIR}/lib/source/GPUArchitecture.cpp
  ${PROJECT_SOURCE_DIR}/lib/source/GPUArchitectureLibrary.cpp

  ${PROJECT_SOURCE_DIR}/lib/source/Assemblers/Assembler.cpp
  ${PROJECT_SOURCE_DIR}/lib/source/Costs/Cost.cpp
  ${PROJECT_SOURCE_DIR}/lib/source/DataTypes_F8_Utils.cpp
  ${PROJECT_SOURCE_DIR}/lib/source/DataTypes_BF16_Utils.cpp
  ${PROJECT_SOURCE_DIR}/lib/source/Error.cpp
  ${PROJECT_SOURCE_DIR}/lib/source/Logging.cpp
  ${PROJECT_SOURCE_DIR}/lib/source/Scheduler.cpp
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

target_compile_options(GPUArchitectureGenerator PUBLIC  ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(GPUArchitectureGenerator PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  GPUArchitectureGenerator PUBLIC
  ${llvm_libs}
  msgpackc-cxx
  isa_spec_manager
  spdlog::spdlog_header_only
)
target_include_directories(GPUArchitectureGenerator PRIVATE
                           include
                           ${PROJECT_SOURCE_DIR}/lib/include
                           )

target_include_directories(GPUArchitectureGenerator SYSTEM PRIVATE
                           ${LLVM_INCLUDE_DIRS}
                           )

target_compile_definitions(GPUArchitectureGenerator PRIVATE ROCROLLER_NO_EMBED_ARCH_DEF)

if(YAML_BACKEND STREQUAL "YAML_CPP")
  target_link_libraries(GPUArchitectureGenerator PUBLIC yaml-cpp)
endif()

if(ROCROLLER_USE_HIP)
  target_link_libraries(GPUArchitectureGenerator PUBLIC hip::host)
  target_compile_definitions(GPUArchitectureGenerator PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)
endif()

set(ARCH_FILE_DIR ${PROJECT_BINARY_DIR}/share/rocRoller)

file(MAKE_DIRECTORY ${ARCH_FILE_DIR})

add_custom_command(
  OUTPUT ${ARCH_FILE_DIR}/GPUArchitecture_def.yaml
  COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/GPUArchitectureGenerator ${ARCH_FILE_DIR}/GPUArchitecture_def.yaml -Y --xml_dir ${mrisa_xml_SOURCE_DIR}
  DEPENDS GPUArchitectureGenerator ${INTERNAL_MRISA_TARGETS}
)

add_custom_command(
  OUTPUT ${ARCH_FILE_DIR}/GPUArchitecture_def.msgpack
  COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/GPUArchitectureGenerator ${ARCH_FILE_DIR}/GPUArchitecture_def.msgpack --xml_dir ${mrisa_xml_SOURCE_DIR}
  DEPENDS GPUArchitectureGenerator ${INTERNAL_MRISA_TARGETS}
)

if(INTERNAL_MRISA_FILES)
  set(delete_internal_command COMMAND ${CMAKE_COMMAND} -E remove ${INTERNAL_MRISA_FILES})
endif()

if(ROCROLLER_EMBED_ARCH_DEF)
# Not exposed to users of the library, so not necessary to use system package
  rocroller_add_dependency(cmrc VERSION 2.0.1)

  cmrc_add_resource_library(
    GPUArchitecture_def_library

    ALIAS rocRoller::GPUArchitecture_def
    NAMESPACE rocRoller

    WHENCE ${ARCH_FILE_DIR}
    PREFIX resources

    ${ARCH_FILE_DIR}/GPUArchitecture_def.msgpack
  )

  set_target_properties(
    GPUArchitecture_def_library
    PROPERTIES
      POSITION_INDEPENDENT_CODE True)

  set(resource_library GPUArchitecture_def_library)
endif()

ADD_CUSTOM_TARGET(
  GPUArchitecture_def
  ALL
  ${delete_internal_command}
  DEPENDS
  ${ARCH_FILE_DIR}/GPUArchitecture_def.yaml
  ${ARCH_FILE_DIR}/GPUArchitecture_def.msgpack
  ${resource_library}
)
