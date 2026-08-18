# ============================================================================
# Usage Instructions:
# ============================================================================
# 1a. Build (Windows, from x64 Native Tools Command Prompt for VS 2022):
#    cmake -B build -G Ninja -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_C_COMPILER="clang" -DHIP_PLATFORM=amd -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_BUILD_TYPE=Release .
#    cmake --build build
#
# 1b. Build (Linux):
#    cmake -B build && cmake --build build
#
# 2. Download model and tokenizer:
#    - Get LLaMA 3 models from: https://huggingface.co/meta-llama/
#    - Export with: python export.py llama3.2_3b_instruct_fp32.bin --meta-llama ../llama3.2-3b-instruct/
#
# 3. Run:
#    ./build/bin/llama3 ~/models/llama3.2_3b_instruct_fp32.bin -z ~/models/tokenizer.bin -s 42 -i "My car" -n 100
#    ./build/bin/llama3 ~/models/llama3.2_3b_instruct_fp32.bin -z ~/models/tokenizer.bin -m chat
#
# Options:
#    -t <float>  temperature in [0,inf], default 1.0
#    -p <float>  p value in top-p sampling in [0,1], default 0.9
#    -s <int>    random seed, default time(NULL)
#    -n <int>    number of steps to run for, default 4096
#    -i <string> input prompt
#    -z <string> path to tokenizer
#    -m <string> mode: generate|chat, default: generate
#    -y <string> system prompt in chat mode
# ============================================================================

cmake_minimum_required(VERSION 3.21)

project(Llama3 LANGUAGES CXX HIP)

if(WIN32)
  set(CMAKE_CXX_USING_LINKER_DEFAULT "-fuse-ld=lld")
  set(CMAKE_CXX_USING_LINKER_LLD "-fuse-ld=lld")
  set(CMAKE_HIP_USING_LINKER_DEFAULT "-fuse-ld=lld")
  set(CMAKE_HIP_USING_LINKER_LLD "-fuse-ld=lld")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

# Treat llama3.cxx as HIP source code
set_source_files_properties(llama3.cxx PROPERTIES LANGUAGE HIP)

add_executable(llama3 llama3.cxx)

if(WIN32)
  target_sources(llama3 PRIVATE win.c)
  set_source_files_properties(win.c PROPERTIES LANGUAGE CXX)
  target_compile_definitions(llama3 PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

target_compile_options(llama3 PUBLIC "-fgpu-rdc" "-O3" "-march=native")
target_link_options(llama3 PUBLIC "-fgpu-rdc" "--hip-link")

if(WIN32)
  # TheRock embeds /DEFAULTLIB:c++ in amdhip64.lib but doesn't ship libc++ on Windows,
  # and clang's HIP driver auto-injects -lc++. Suppress both; MSVC's stdlib pulls in
  # automatically via #pragma comment(lib, "msvcprt") from MSVC headers.
  target_link_options(llama3 PRIVATE "-nostdlib++" "-Xlinker" "/NODEFAULTLIB:c++.lib")
endif()

find_package(hipthreads REQUIRED)
find_package(rocprim REQUIRED CONFIG)
find_package(rocthrust REQUIRED CONFIG)

target_link_libraries(llama3 PRIVATE hipthreads::hipthreads)
target_link_libraries(llama3 PRIVATE roc::rocthrust)

# needed to get intellisense to stop complaining about the co-ordinate built-ins not being defined
target_compile_definitions(llama3 PUBLIC __HIP__)
