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
