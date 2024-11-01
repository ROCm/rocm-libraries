project(rocsparse_mtx2csr)
message("CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR}")
add_executable(rocsparse_mtx2csr ../../deps/convert.cpp)