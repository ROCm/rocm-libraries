include_guard(DIRECTORY)

if(CMAKE_CXX_FLAGS MATCHES "(^| )--offload-compress( |$)")
  message(WARNING
    "'--offload-compress' is set before including HIP. This may"
    "cause CMake errors when including CXX depedencies."
  )
endif()

find_package(hip REQUIRED)
