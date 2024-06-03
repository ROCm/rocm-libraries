cmake_minimum_required(VERSION 3.10)

# can't figure out how to get clang-tidy to use hipcc/recognize HIP code
# set(CMAKE_CXX_CLANG_TIDY clang-tidy "-checks=                                                                          \
#             readability-*,-readability-identifier-length,-readability-magic-numbers,-readability-isolate-declaration,  \
#             modernize-*,-modernize-use-trailing-return-type,-modernize-avoid-c-arrays,                                 \
#             bugprone-*,-bugprone-easily-swappable-parameters,                                                          \
#             concurrency-*")

# Turn static library generation ON
option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

if (WIN32)
  # For windows, AR is MS Librarian and that is picked by Visual Studio's command prompt.
  find_program(libpath NAMES lib.exe)
  set (CMAKE_AR ${libpath})

  # For windows, We need to tell cmake how to create static library.
  set (CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /out:<TARGET> <LINK_FLAGS> <OBJECTS>")
endif()

# Generate static lib libGpuThreads.a
add_library(GpuThreads STATIC gpu/thread.cxx)
target_include_directories(GpuThreads PUBLIC ${CMAKE_SOURCE_DIR}/inc)

target_compile_options(GpuThreads PRIVATE "-Werror" "-Wall" "-Wextra" "-Wshadow" "-Wnull-dereference" "-Wold-style-cast" "-Wzero-as-null-pointer-constant" "-Wfloat-equal" "-Wdouble-promotion")
target_compile_options(GpuThreads PRIVATE "-save-temps=obj" "-O2" "-fgpu-rdc")
target_link_libraries(GpuThreads PRIVATE "-fgpu-rdc")

# needed to get intellisense to stop complaining about the co-ordinate built-ins not being defined
target_compile_definitions(GpuThreads PUBLIC __HIP__)

#TODO: define an install target, and a GpuThreadsConfig.cmake so find_package can be used


# add_custom_command(TARGET GpuThreads POST_BUILD
#     COMMAND ${LLVM_DIS} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/GpuThreads.dir/gpu/thread-host-x86_64-unknown-linux-gnu.bc
# )
# add_custom_command(TARGET GpuThreads POST_BUILD
#     COMMAND ${LLVM_DIS} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/GpuThreads.dir/gpu/thread-hip-amdgcn-amd-amdhsa-gfx1100.bc
# )
