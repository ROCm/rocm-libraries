

cmake_minimum_required(VERSION 3.16)

project(rocRollerClient VERSION 1.0)

option(BUILD_CLIENT "Build client program" ON)

if(BUILD_CLIENT)
  message("Building client program")

  list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
  find_package(hip)

  add_executable(gemm
    gemm.cpp

    ../test/unit/Utilities.cpp
  )

  target_compile_options(gemm PUBLIC ${EXTRA_COMPILE_OPTIONS})
  target_include_directories(
    gemm
    PUBLIC
    lib/include
    ${HIP_ROOT_DIR}/include
  )
  target_link_options(gemm PUBLIC ${EXTRA_LINK_OPTIONS})
  target_link_libraries(
    gemm
    PUBLIC
    rocroller
    msgpackc-cxx
    gtest_main
    gmock
    openblas
  )

  target_link_libraries(gemm PUBLIC hip::host)
  target_compile_definitions(gemm PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)

  if(ROCROLLER_USE_LLVM)
    target_compile_definitions(gemm PUBLIC ROCROLLER_USE_LLVM)
    target_include_directories(
      gemm
      PUBLIC
      ${HIP_ROOT_DIR}/include
      ${LLVM_INCLUDE_DIRS}
      ${LLD_INCLUDE_DIRS}
    )
    target_link_libraries(
      gemm
      PUBLIC
      ${llvm_libs}
      lldELF
    )
  endif()
  if(ROCROLLER_USE_YAML_CPP)
    target_compile_definitions(gemm PUBLIC ROCROLLER_USE_YAML_CPP)
    target_link_libraries(gemm PUBLIC yaml-cpp)
    set_property(TARGET yaml-cpp PROPERTY POSITION_INDEPENDENT_CODE ON)
  endif()


  add_custom_target(all_clients)
  add_dependencies(all_clients gemm)
endif()
