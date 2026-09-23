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

   add_test_executable(test_tdm_fmha_v128_contract tdm_fmha_v128_contract.cpp)
   target_link_libraries(test_tdm_fmha_v128_contract PRIVATE utility)
   target_compile_definitions(test_tdm_fmha_v128_contract PRIVATE
      CK_TILE_FMHA_TDM_V128_TEST_INTEGRATION=1)
   set_property(TARGET test_tdm_fmha_v128_contract
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})

   add_test_executable(test_tdm_fmha_v128_schedule_contract tdm_fmha_v128_schedule_contract.cpp)
   target_link_libraries(test_tdm_fmha_v128_schedule_contract PRIVATE utility)
   set_property(TARGET test_tdm_fmha_v128_schedule_contract
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})

   add_test_executable(test_tdm_fmha_v128_reference tdm_fmha_v128_reference_test.cpp)
   target_link_libraries(test_tdm_fmha_v128_reference PRIVATE utility)
   set_property(TARGET test_tdm_fmha_v128_reference
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})

   add_test_executable(test_tdm_fmha_v128_lifetime tdm_fmha_v128_lifetime_test.cpp)
   target_link_libraries(test_tdm_fmha_v128_lifetime PRIVATE utility)
   target_include_directories(test_tdm_fmha_v128_lifetime PRIVATE
      ${PROJECT_SOURCE_DIR}/example/ck_tile/01_fmha)
   target_compile_definitions(test_tdm_fmha_v128_lifetime PRIVATE CK_TILE_FMHA_FWD_FAST_EXP2=1)
   set_property(TARGET test_tdm_fmha_v128_lifetime
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})

   set(TDM_FMHA_EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/example/ck_tile/01_fmha)
   add_test_executable(test_tdm_fmha_v128_layout tdm_fmha_v128_layout.cpp)
   target_link_libraries(test_tdm_fmha_v128_layout PRIVATE utility)
   target_include_directories(test_tdm_fmha_v128_layout PRIVATE ${TDM_FMHA_EXAMPLE_DIR})
   target_compile_definitions(test_tdm_fmha_v128_layout PRIVATE CK_TILE_FMHA_FWD_FAST_EXP2=1)
   set_property(TARGET test_tdm_fmha_v128_layout PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})
   set_tests_properties(test_tdm_fmha_v128_layout PROPERTIES SKIP_RETURN_CODE 77)

   add_test_executable(test_tdm_fmha_v128_layout_fp16 tdm_fmha_v128_layout.cpp)
   target_link_libraries(test_tdm_fmha_v128_layout_fp16 PRIVATE utility)
   target_include_directories(test_tdm_fmha_v128_layout_fp16 PRIVATE ${TDM_FMHA_EXAMPLE_DIR})
   target_compile_definitions(test_tdm_fmha_v128_layout_fp16 PRIVATE
      CK_TILE_FMHA_FWD_FAST_EXP2=1 CK_TILE_FMHA_TDM_V128_TEST_FP16=1)
   set_property(TARGET test_tdm_fmha_v128_layout_fp16
      PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})
   set_tests_properties(test_tdm_fmha_v128_layout_fp16 PROPERTIES SKIP_RETURN_CODE 77)

   # The example runner defaults to FP16; register an explicit BF16-only smoke.
   set_source_files_properties(tdm_fmha_v128_fwd.cpp
      ${TDM_FMHA_EXAMPLE_DIR}/example_fmha_fwd.cpp PROPERTIES LANGUAGE HIP)
   set(TDM_FMHA_V128_RUNNER_SUPPORT tdm_fmha_v128_runner_support.cpp)
   add_executable(test_tdm_fmha_v128_fwd tdm_fmha_v128_fwd.cpp
      ${TDM_FMHA_V128_RUNNER_SUPPORT}
      ${TDM_FMHA_EXAMPLE_DIR}/example_fmha_fwd.cpp)
   target_link_libraries(test_tdm_fmha_v128_fwd PRIVATE utility getopt::getopt)
   add_test(NAME test_tdm_fmha_v128_fwd COMMAND $<TARGET_FILE:test_tdm_fmha_v128_fwd>
      -prec=bf16 -mode=0 -b=1 -h=1 -h_k=1 -d=128 -d_v=128 -s=128 -s_k=128
      -iperm=0 -operm=0 -mask=0 -v=1 -warmup=0 -repeat=1)
   set_tests_properties(test_tdm_fmha_v128_fwd PROPERTIES SKIP_RETURN_CODE 77 LABELS SMOKE_TEST)
   add_dependencies(tests test_tdm_fmha_v128_fwd)
   add_dependencies(check test_tdm_fmha_v128_fwd)
   add_dependencies(smoke test_tdm_fmha_v128_fwd)
   rocm_install(TARGETS test_tdm_fmha_v128_fwd COMPONENT tests)
   target_include_directories(test_tdm_fmha_v128_fwd PRIVATE ${TDM_FMHA_EXAMPLE_DIR})
   target_compile_definitions(test_tdm_fmha_v128_fwd PRIVATE
      CK_TILE_FMHA_FWD_FAST_EXP2=1 CK_TILE_FMHA_FWD_API=1
      CK_TILE_FMHA_FWD_SPLITKV_API=0 CK_TILE_FMHA_FWD_APPENDKV_API=0
      CK_TILE_FMHA_FWD_PAGEDKV_API=0 CK_TILE_FMHA_FWD_BATCH_PREFILL_API=0)
   # Test TU only. Production generated-TU routing belongs to the dispatch slice.
   set_property(SOURCE tdm_fmha_v128_fwd.cpp APPEND PROPERTY COMPILE_OPTIONS
      -mllvm -amdgpu-expert-scheduling-mode -fgpu-flush-denormals-to-zero)
   set_property(TARGET test_tdm_fmha_v128_fwd PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})

   set_source_files_properties(tdm_fmha_v128_oracle.cpp PROPERTIES LANGUAGE HIP)
   add_executable(test_tdm_fmha_v128_oracle tdm_fmha_v128_oracle.cpp
      ${TDM_FMHA_V128_RUNNER_SUPPORT}
      ${TDM_FMHA_EXAMPLE_DIR}/example_fmha_fwd.cpp)
   target_link_libraries(test_tdm_fmha_v128_oracle PRIVATE utility getopt::getopt)
   target_include_directories(test_tdm_fmha_v128_oracle PRIVATE ${TDM_FMHA_EXAMPLE_DIR})
   target_compile_definitions(test_tdm_fmha_v128_oracle PRIVATE
      CK_TILE_FMHA_FWD_FAST_EXP2=1 CK_TILE_FMHA_FWD_API=1
      CK_TILE_FMHA_FWD_SPLITKV_API=0 CK_TILE_FMHA_FWD_APPENDKV_API=0
      CK_TILE_FMHA_FWD_PAGEDKV_API=0 CK_TILE_FMHA_FWD_BATCH_PREFILL_API=0)
   set_property(SOURCE tdm_fmha_v128_oracle.cpp APPEND PROPERTY COMPILE_OPTIONS
      -mllvm -amdgpu-expert-scheduling-mode -fgpu-flush-denormals-to-zero)
   set_property(TARGET test_tdm_fmha_v128_oracle PROPERTY HIP_ARCHITECTURES ${TDM_FMHA_D192_GPU_TARGETS})
   add_test(NAME test_tdm_fmha_v128_oracle COMMAND $<TARGET_FILE:test_tdm_fmha_v128_oracle>
      -prec=bf16 -mode=0 -b=1 -h=1 -h_k=1 -d=128 -d_v=128 -s=128 -s_k=128
      -iperm=0 -operm=0 -mask=0 -lse=1 -v=0 -init=uf -seed=20260908 -warmup=0 -repeat=1)
   set_tests_properties(test_tdm_fmha_v128_oracle PROPERTIES SKIP_RETURN_CODE 77 LABELS SMOKE_TEST)
   add_dependencies(tests test_tdm_fmha_v128_oracle)
   add_dependencies(check test_tdm_fmha_v128_oracle)
   add_dependencies(smoke test_tdm_fmha_v128_oracle)
   rocm_install(TARGETS test_tdm_fmha_v128_oracle COMPONENT tests)
endif()
