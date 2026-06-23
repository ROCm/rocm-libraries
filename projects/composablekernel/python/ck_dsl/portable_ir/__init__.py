# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Portable CK-DSL IR + builder-recipe tooling (Python authoring side).
#
# Layout:
#   src/      core engine + runtime binding
#               recording_builder    RecordingIRBuilder + record_kernel
#               kerneldef_to_recipe  KernelDef -> concrete recipe
#               recipe_recorder      record a parametric recipe from idiomatic authoring
#               roller / roll        infer + verify ONE parametric recipe over an axis
#               recipe_bundle        CBOR codec + bundle (ck.dsl.bundle/v1)
#               online               ctypes binding to the C backend (recipe/IR -> .ll)
#   utils/    device-free helpers
#               recipe_expand        pure-Python recipe expander + recipes_equiv oracle
#   examples/ demo kernels / authoring emitters (each runnable: --emit recipe|ll|name)
#               recipe_toy mini_attn qk_block export_mha export_gemm_cshuffle
#               recipe_multi_result
#   drivers/  runnable harnesses / benchmarks / end-to-end
#               bench_online record_coverage roll_coverage
#               verify_recording_production roll_recipe
#   tests/    unittest suites
#   portable_ir_scaling_plan.md   the scaling plan (kept next to the code)
#
# ir_export (in ck_dsl.core) serializes a KernelDef to portable IR (schema
# ck.dsl.ir/v1). The C engine (recipe VM, JSON/CBOR DOM, importer, online wrapper)
# lives in ../ck_dsl_c; the C drivers and shell runners live in
# ../ck_dsl_c/tests/portable_ir.
