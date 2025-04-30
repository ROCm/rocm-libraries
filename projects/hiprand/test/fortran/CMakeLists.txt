# Fortran Wrapper Tests

file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/fruit/*.f90)
set(FRUIT_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/fruit_helpers.f90)
set(FRUIT_HELPER_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*_hiprand.f90)
set(FORTRAN_HIPRAND_TEST ${tmp})

set(test_hiprand "test_hiprand_fortran_wrapper")

# Build unit test (hiprand)
add_executable(${test_hiprand} "${FRUIT_SRCS}" "${FRUIT_HELPER_SRCS}" "${FORTRAN_HIPRAND_TEST}")
target_include_directories(${test_hiprand}
    PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/fortran>"
)
target_link_libraries(${test_hiprand} PRIVATE hiprand_fortran hip::host)
if(HIP_COMPILER STREQUAL "nvcc")
    if(CMAKE_VERSION VERSION_LESS 3.17)
        find_package(CUDA REQUIRED)
    else()
        find_package(CUDAToolkit)
        set(CUDA_LIBRARIES CUDA::cudart)
    endif()
    target_link_libraries(${test_hiprand} PRIVATE ${CUDA_LIBRARIES})
endif()
set_target_properties(${test_hiprand}
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)
rocm_install(TARGETS ${test_hiprand} COMPONENT tests)

add_relative_test(${test_hiprand} ${test_hiprand})
