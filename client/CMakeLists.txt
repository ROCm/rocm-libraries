option(BUILD_CLIENT "Build client program" ON)

if(BUILD_CLIENT)
  message("Building client program")

  add_executable(rocRollerClient 
    main.cpp
    
    ../test/unit/Utilities.cpp
  )

  target_compile_options(rocRollerClient PUBLIC ${EXTRA_COMPILE_OPTIONS})
  target_include_directories(rocRollerClient PUBLIC
                              lib/include
                            )
  target_link_options(rocRollerClient PUBLIC ${EXTRA_LINK_OPTIONS})
  target_link_libraries(
    rocRollerClient
    rocroller
    msgpackc-cxx
    gtest_main
    gmock
    openblas
  )
endif()
