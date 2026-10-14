# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a|gfx125|gfx11|gfx12")
  add_executable(tile_example_gemm_basic gemm_basic.cpp)
  add_executable(tile_example_gemm_universal universal_gemm.cpp)
  add_executable(tile_example_gemm_weight_preshuffle gemm_weight_preshuffle.cpp)
  add_executable(tile_example_gemm_reduce gemm_splitk_two_stage_reduce.cpp)
  add_executable(tile_example_gemm_splitk_two_stage gemm_splitk_two_stage.cpp)
  if(GPU_TARGETS MATCHES "gfx125")
    add_executable(tile_example_gemm_mixed_prec gemm_mixed_prec.cpp)
    add_executable(tile_example_gemm_tdm_data_cache_prefetch gemm_tdm_data_cache_prefetch.cpp)
    add_executable(tile_example_gemm_weight_preshuffle_tdm_data_cache_prefetch gemm_weight_preshuffle_tdm_data_cache_prefetch.cpp)
    add_executable(tile_example_gemm_universal_cluster_launch universal_gemm.cpp)

    set(CK_GEMM_SWEEP_TILE_CONFIGS "128,128,64,2,4,1" CACHE STRING
        "Semicolon-separated M,N,K,MWarp,NWarp,KWarp tile presets for basic/TDM sweep binaries")
    foreach(tile_config IN LISTS CK_GEMM_SWEEP_TILE_CONFIGS)
      string(REPLACE "," ";" tile_values "${tile_config}")
      list(LENGTH tile_values tile_value_count)
      if(NOT tile_value_count EQUAL 6)
        message(FATAL_ERROR
          "Invalid CK_GEMM_SWEEP_TILE_CONFIGS entry '${tile_config}'; expected M,N,K,MWarp,NWarp,KWarp")
      endif()

      list(GET tile_values 0 tile_m)
      list(GET tile_values 1 tile_n)
      list(GET tile_values 2 tile_k)
      list(GET tile_values 3 warp_m)
      list(GET tile_values 4 warp_n)
      list(GET tile_values 5 warp_k)
      set(tile_suffix "m${tile_m}_n${tile_n}_k${tile_k}_wm${warp_m}_wn${warp_n}_wk${warp_k}")

      foreach(pipeline IN ITEMS basic tdm)
        set(target "tile_example_gemm_sweep_${pipeline}_${tile_suffix}")
        add_executable(${target} gemm_pipeline_sweep.cpp)
        target_compile_definitions(${target} PRIVATE
          CK_GEMM_SWEEP_PIPELINE_TDM=$<STREQUAL:${pipeline},tdm>
          CK_GEMM_SWEEP_M_TILE=${tile_m}
          CK_GEMM_SWEEP_N_TILE=${tile_n}
          CK_GEMM_SWEEP_K_TILE=${tile_k}
          CK_GEMM_SWEEP_M_WARP=${warp_m}
          CK_GEMM_SWEEP_N_WARP=${warp_n}
          CK_GEMM_SWEEP_K_WARP=${warp_k})
      endforeach()
    endforeach()
  endif()
  set(EXAMPLE_GEMM_COMPILE_OPTIONS)
  set(EXAMPLE_WEIGHT_PRESHUFFLE_COMPILE_OPTIONS)
  if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
  endif()
  list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -mllvm -enable-noalias-to-md-conversion=0)
  list(APPEND EXAMPLE_WEIGHT_PRESHUFFLE_COMPILE_OPTIONS -Wno-unused-local-typedef)
  list(APPEND EXAMPLE_WEIGHT_PRESHUFFLE_COMPILE_OPTIONS -Wno-gnu-line-marker)
  # list(APPEND EXAMPLE_WEIGHT_PRESHUFFLE_COMPILE_OPTIONS --save-temps)
  list(APPEND EXAMPLE_WEIGHT_PRESHUFFLE_COMPILE_OPTIONS "SHELL: -mllvm -greedy-reverse-local-assignment=1 -mllvm -enable-noalias-to-md-conversion=1")
  target_compile_options(tile_example_gemm_basic PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
  target_compile_options(tile_example_gemm_universal PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
  target_compile_options(tile_example_gemm_weight_preshuffle PRIVATE ${EXAMPLE_WEIGHT_PRESHUFFLE_COMPILE_OPTIONS})
  target_compile_options(tile_example_gemm_reduce PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
  target_compile_options(tile_example_gemm_splitk_two_stage PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
  if(GPU_TARGETS MATCHES "gfx125")
    target_compile_options(tile_example_gemm_mixed_prec PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(tile_example_gemm_tdm_data_cache_prefetch PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(tile_example_gemm_weight_preshuffle_tdm_data_cache_prefetch PRIVATE ${EXAMPLE_WEIGHT_PRESHUFFLE_COMPILE_OPTIONS})
    target_compile_options(tile_example_gemm_universal_cluster_launch PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_definitions(tile_example_gemm_universal_cluster_launch PRIVATE CLUSTER_LAUNCH_ENABLED=1)
    foreach(tile_config IN LISTS CK_GEMM_SWEEP_TILE_CONFIGS)
      string(REPLACE "," ";" tile_values "${tile_config}")
      list(GET tile_values 0 tile_m)
      list(GET tile_values 1 tile_n)
      list(GET tile_values 2 tile_k)
      list(GET tile_values 3 warp_m)
      list(GET tile_values 4 warp_n)
      list(GET tile_values 5 warp_k)
      set(tile_suffix "m${tile_m}_n${tile_n}_k${tile_k}_wm${warp_m}_wn${warp_n}_wk${warp_k}")
      foreach(pipeline IN ITEMS basic tdm)
        target_compile_options(tile_example_gemm_sweep_${pipeline}_${tile_suffix}
          PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
      endforeach()
    endforeach()
  endif()
endif()
