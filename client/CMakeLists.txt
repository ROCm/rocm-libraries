option(BUILD_CLIENT "Build client program" ON)

if(BUILD_CLIENT)
  message("Building client program")

  add_executable(gemm
    gemm.cpp

    ../test/unit/Utilities.cpp
  )

  target_compile_options(gemm PUBLIC ${EXTRA_COMPILE_OPTIONS})
  target_include_directories(gemm PUBLIC
                              lib/include
                            )
  target_link_options(gemm PUBLIC ${EXTRA_LINK_OPTIONS})
  target_link_libraries(
    gemm
    rocroller
    msgpackc-cxx
    gtest_main
    gmock
    openblas
  )
endif()
