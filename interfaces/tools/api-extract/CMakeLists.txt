find_package(LLVM CONFIG QUIET)
find_package(Clang CONFIG QUIET)

if(NOT LLVM_FOUND OR NOT Clang_FOUND)
  message(WARNING "LLVM/Clang packages not found; rocm-api-extract will not be built")
  return()
endif()

add_executable(rocm-api-extract api_extract.cpp)
target_include_directories(rocm-api-extract SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(rocm-api-extract PRIVATE -fno-rtti)
endif()
if(TARGET clang-cpp)
  target_link_libraries(rocm-api-extract PRIVATE clang-cpp LLVM)
else()
  target_link_libraries(rocm-api-extract PRIVATE
    clangTooling clangFrontend clangASTMatchers clangAST clangBasic LLVM)
endif()
