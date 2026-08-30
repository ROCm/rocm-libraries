if(GPU_TARGETS MATCHES "gfx125")
   add_gtest_executable(test_tdm_basic tdm_basic.cpp)
   target_link_libraries(test_tdm_basic PRIVATE utility)

   add_test_executable(test_tdm_fmha_d192_layout tdm_fmha_d192_layout.cpp)
   target_link_libraries(test_tdm_fmha_d192_layout PRIVATE utility)
   set(TDM_FMHA_D192_GPU_TARGETS ${SUPPORTED_GPU_TARGETS})
   list(FILTER TDM_FMHA_D192_GPU_TARGETS INCLUDE REGEX "^gfx125")
   set_property(TARGET test_tdm_fmha_d192_layout
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})
   set_tests_properties(test_tdm_fmha_d192_layout PROPERTIES SKIP_RETURN_CODE 77)

   add_test_executable(test_tdm_fmha_d192_schedule tdm_fmha_d192_schedule.cpp)
   target_link_libraries(test_tdm_fmha_d192_schedule PRIVATE utility)
   set_property(TARGET test_tdm_fmha_d192_schedule
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})

   add_test_executable(test_tdm_fmha_d192_softmax tdm_fmha_d192_softmax.cpp)
   target_link_libraries(test_tdm_fmha_d192_softmax PRIVATE utility)
   set_property(TARGET test_tdm_fmha_d192_softmax
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})
   set_tests_properties(test_tdm_fmha_d192_softmax PROPERTIES SKIP_RETURN_CODE 77)

   add_test_executable(test_tdm_fmha_d192_cross_tile
      tdm_fmha_d192_cross_tile.cpp)
   target_link_libraries(test_tdm_fmha_d192_cross_tile PRIVATE utility)
   set_property(TARGET test_tdm_fmha_d192_cross_tile
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})
   set_tests_properties(test_tdm_fmha_d192_cross_tile PROPERTIES SKIP_RETURN_CODE 77)

   add_test_executable(test_tdm_fmha_d192_cross_tile_fast_exp2
      tdm_fmha_d192_cross_tile.cpp)
   target_link_libraries(test_tdm_fmha_d192_cross_tile_fast_exp2 PRIVATE utility)
   target_compile_definitions(test_tdm_fmha_d192_cross_tile_fast_exp2 PRIVATE
      CK_TILE_FMHA_FWD_FAST_EXP2=1)
   set_property(TARGET test_tdm_fmha_d192_cross_tile_fast_exp2
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})
   set_tests_properties(test_tdm_fmha_d192_cross_tile_fast_exp2 PROPERTIES SKIP_RETURN_CODE 77)

   add_test_executable(test_tdm_fmha_d192_rescale_schedule
      tdm_fmha_d192_rescale_schedule.cpp)
   target_link_libraries(test_tdm_fmha_d192_rescale_schedule PRIVATE utility)
   set_property(TARGET test_tdm_fmha_d192_rescale_schedule
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})
   set_tests_properties(test_tdm_fmha_d192_rescale_schedule PROPERTIES SKIP_RETURN_CODE 77)

   add_test_executable(test_tdm_fmha_d192_deferred_p
      tdm_fmha_d192_deferred_p.cpp)
   target_link_libraries(test_tdm_fmha_d192_deferred_p PRIVATE utility)
   set_property(TARGET test_tdm_fmha_d192_deferred_p
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})
   set_tests_properties(test_tdm_fmha_d192_deferred_p PROPERTIES SKIP_RETURN_CODE 77)
endif()
