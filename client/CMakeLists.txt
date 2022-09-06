

cmake_minimum_required(VERSION 3.16)

project(rocRollerClient VERSION 1.0)

option(BUILD_CLIENT "Build client program" ON)

if(BUILD_CLIENT)
  message("Building client program")

  if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
  endif()
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP)

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
    amdhip64
  )
  target_link_directories(gemm PUBLIC ${HIP_ROOT_DIR}/lib)
  target_compile_definitions(gemm PUBLIC __HIP_PLATFORM_AMD__)

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
endif()
