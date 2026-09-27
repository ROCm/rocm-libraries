function(add_compile_test testFileName)
    cmake_path(GET testFileName STEM LAST_ONLY testName)
    set(testName ${testPrefix}${testName})
    message(STATUS "testName: ${testName}")
    add_executable(${testName} ${testFileName})
    set_source_files_properties(${testFileName} PROPERTIES LANGUAGE HIP)

    target_include_directories(${testName} PUBLIC ${CMAKE_SOURCE_DIR}/inc)
    if(NOT DISABLE_WERROR)
      target_compile_options(${testName} PUBLIC "-Werror")
    endif()
    target_compile_options(${testName} PUBLIC "-Wall" "-Wextra" "-Wshadow" "-Wnull-dereference" "-Wold-style-cast" "-Wzero-as-null-pointer-constant" "-Wfloat-equal" "-Wdouble-promotion")
    target_compile_options(${testName} PUBLIC "-save-temps=obj" "-O2" "-fgpu-rdc")
    target_link_options(${testName} PUBLIC "-fgpu-rdc" "--hip-link")
    target_link_libraries(${testName} PRIVATE hipthreads)
    # needed to get intellisense to stop complaining about the co-ordinate built-ins not being defined
    target_compile_definitions(${testName} PUBLIC __HIP__)
    # add_custom_command(TARGET ${testName} POST_BUILD
    #     COMMAND ${LLVM_DIS} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${testName}.dir/${testName}-host-x86_64-unknown-linux-gnu.bc
    # )
    # add_custom_command(TARGET ${testName} POST_BUILD
    #     COMMAND ${LLVM_DIS} ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${testName}.dir/${testName}-hip-amdgcn-amd-amdhsa-gfx1034.bc
    # )
    #add_custom_target(targetName)
    #add_custom_command(TARGET targetName POST_BUILD COMMAND ${testName})
endfunction(add_compile_test)

set(testPrefix "")

# TEMP: until libhipcxx resolves this
add_compile_definitions(_LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE)

file(GLOB TEST_SOURCES "*.cxx")
foreach(testFileName ${TEST_SOURCES})
    add_compile_test(${testFileName})
endforeach(testFileName)

# For windows, Change in a way to pass lib details
if (WIN32)
  target_link_libraries(hip_thread_mutex_test PRIVATE -lhipthreads -L${CMAKE_BINARY_DIR})
  target_link_libraries(hip_thread_condvar_test PRIVATE -lhipthreads -L${CMAKE_BINARY_DIR})
endif()
