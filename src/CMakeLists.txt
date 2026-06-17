cmake_minimum_required(VERSION 3.21)

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

# Generate static lib libhipthreads.a
add_library(hipthreads STATIC hip/thread.cxx)
set_source_files_properties(hip/thread.cxx PROPERTIES LANGUAGE HIP)
target_include_directories(hipthreads PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/inc>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipthreads>
)

if(NOT DISABLE_WERROR)
  target_compile_options(hipthreads PRIVATE "-Werror")
endif()
target_compile_options(hipthreads PRIVATE "-Wall" "-Wextra" "-Wshadow" "-Wnull-dereference" "-Wold-style-cast" "-Wzero-as-null-pointer-constant" "-Wfloat-equal" "-Wdouble-promotion")
target_compile_options(hipthreads PRIVATE "-save-temps=obj" "-O2" "-fgpu-rdc")
target_link_libraries(hipthreads PRIVATE "-fgpu-rdc")

# needed to get intellisense to stop complaining about the co-ordinate built-ins not being defined
target_compile_definitions(hipthreads PUBLIC __HIP__)

# Optional override for the number of scheduler vcores launched per WGP/multiprocessor.
# Default lives in src/hip/thread.cxx; only forward a -D when the user sets a value.
set(HIPTHREADS_VCORES_PER_WGP "" CACHE STRING
  "Scheduler vcores launched per WGP/multiprocessor (default defined in src/hip/thread.cxx)")
if(HIPTHREADS_VCORES_PER_WGP)
  target_compile_definitions(hipthreads PRIVATE HIPTHREADS_VCORES_PER_WGP=${HIPTHREADS_VCORES_PER_WGP})
endif()


target_link_libraries(hipthreads PUBLIC libhipcxx::libhipcxx)


# add_custom_command(TARGET hipthreads POST_BUILD
#     COMMAND ${LLVM_DIS} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/hipthreads.dir/hip/thread-host-x86_64-unknown-linux-gnu.bc
# )
# add_custom_command(TARGET hipthreads POST_BUILD
#     COMMAND ${LLVM_DIS} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/hipthreads.dir/hip/thread-hip-amdgcn-amd-amdhsa-gfx1100.bc
# )
