cmake_minimum_required(VERSION 3.18)
project(pybind11_rtld_local_spike CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

add_library(pyplugin SHARED pyplugin.cpp)
target_link_libraries(pyplugin PRIVATE pybind11::embed)
set_target_properties(pyplugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    PREFIX "lib"
)

add_executable(host_main host_main.cpp)
target_link_libraries(host_main PRIVATE ${CMAKE_DL_LIBS})
# Export host's dynamic symbols so libpython (loaded transitively) can resolve.
set_target_properties(host_main PROPERTIES ENABLE_EXPORTS ON)
