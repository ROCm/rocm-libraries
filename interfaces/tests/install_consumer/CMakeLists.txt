cmake_minimum_required(VERSION 3.25)
project(rocm-interfaces-install-consumer LANGUAGES CXX)

find_package(ROCmInterfacesShadow CONFIG REQUIRED)
if(NOT TARGET ROCmInterfacesShadow::rocblas_narrow_v2_loader)
  message(FATAL_ERROR "installed narrow-v2 loader target is missing")
endif()
add_executable(rocm_interfaces_install_consumer main.cpp)
target_link_libraries(rocm_interfaces_install_consumer PRIVATE
  ROCmInterfacesShadow::loader
  ROCmInterfacesShadow::rocblas_loader)
