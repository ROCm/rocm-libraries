# In-tree unit test. Builds against the rpp target, so it needs no installed RPP
# and no test data, Python, OpenCV or OpenMP.

add_executable(test_rpp_backend_macro test_rpp_backend_macro.cpp)

# Linking rpp supplies the public header paths (including the generated
# api/rpp_backend.h) and, for HIP, hip::host. HIP builds also inherit
# -DRPP_BACKEND_HIP=1 from add_definitions() in the parent scope; HOST builds
# inherit nothing, which is the case the header guards regressed on.
target_link_libraries(test_rpp_backend_macro PRIVATE ${PROJECT_NAME})

if("${BACKEND}" STREQUAL "HIP")
    target_compile_definitions(test_rpp_backend_macro PRIVATE EXPECTED_RPP_BACKEND_HIP=1)
else()
    target_compile_definitions(test_rpp_backend_macro PRIVATE EXPECTED_RPP_BACKEND_HIP=0)
endif()

add_test(NAME rpp_backend_macro COMMAND test_rpp_backend_macro)

# Linux executable symbols interpose the HIP calls made by the shared RPP library.
# This keeps fault injection independent of GPU access and avoids LD_PRELOAD
# interfering with sanitizer runtime initialization. Other platforms still run
# the backend macro test above.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_handle_lifecycle test_handle_lifecycle.cpp)
    target_link_libraries(test_handle_lifecycle PRIVATE ${PROJECT_NAME})
    target_link_options(test_handle_lifecycle PRIVATE "-Wl,--export-dynamic")
    set(lifecycle_cases host_default host_explicit host_mismatch host_invalid null)
    if("${BACKEND}" STREQUAL "HIP")
        list(APPEND lifecycle_cases hip_default hip_explicit hip_mismatch
             hip_device_retry hip_pinned_retry)
    endif()
    foreach(case IN LISTS lifecycle_cases)
        add_test(NAME rpp_handle_${case} COMMAND test_handle_lifecycle ${case})
    endforeach()
endif()
