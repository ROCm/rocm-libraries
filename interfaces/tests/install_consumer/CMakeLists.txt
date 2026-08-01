cmake_minimum_required(VERSION 3.25)
project(rocm-interfaces-install-consumer LANGUAGES CXX)

find_package(ROCmInterfacesShadow CONFIG REQUIRED)
add_executable(rocm_interfaces_install_consumer main.cpp)
target_link_libraries(rocm_interfaces_install_consumer PRIVATE
  ROCmInterfacesShadow::loader)
