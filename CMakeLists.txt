cmake_minimum_required(VERSION 3.10)

project(GpuLib)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld -stdlib=libc++")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld")

list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
find_package(hip REQUIRED)

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

# Used for automatic generation of LLVM IR (by dissasembling bitcode generated during compilation from the "-save-temps" option)
find_program(LLVM_DIS llvm-dis REQUIRED HINTS /opt/rocm/llvm/bin)

add_subdirectory(src)
if (CMAKE_BUILD_TYPE MATCHES Debug)
  add_subdirectory(test)
endif()
