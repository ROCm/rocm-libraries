# Fortran Wrapper
message(WARNING "rocRAND's Fortran API is deprecated. hipfort should be used instead.")
if(HIP_COMPILER STREQUAL "nvcc")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_nvcc_m.f90"
        "${PROJECT_BINARY_DIR}/src/fortran/hip_m.f90"
        COPYONLY
    )
else()
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_rocm_m.f90"
        "${PROJECT_BINARY_DIR}/src/fortran/hip_m.f90"
        COPYONLY
    )
endif()

set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/library/fortran)

# rocRAND Fortran wrapper
set(FORTRAN_ROCRAND_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/rocrand_m.f90
    ${PROJECT_BINARY_DIR}/src/fortran/hip_m.f90
)
add_library(rocrand_fortran STATIC "${FORTRAN_ROCRAND_SRCS}")
set_target_properties(rocrand_fortran PROPERTIES LINKER_LANGUAGE Fortran)
if(HIP_COMPILER STREQUAL "nvcc")
    target_link_libraries(rocrand_fortran
        PRIVATE
            rocrand
            ${CUDA_LIBRARIES}
    )
else()
    target_link_libraries(rocrand_fortran
        PRIVATE
            rocrand
            "-L${HIP_ROOT_DIR}/lib -Wl,-rpath,${HIP_ROOT_DIR}/lib"
    )
endif()

# Install Fortran sources
rocm_install(
    FILES ${FORTRAN_ROCRAND_SRCS}
    DESTINATION rocrand/src/fortran
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
