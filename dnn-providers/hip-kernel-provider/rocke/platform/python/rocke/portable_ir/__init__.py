# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Portable rocKE IR + builder-recipe tooling (Python authoring side).
#
# Layout:
#   src/      core engine + runtime binding
#               recording_builder    RecordingIRBuilder + record_kernel
#               kerneldef_to_recipe  KernelDef -> concrete recipe
#               recipe_recorder      record a parametric recipe from idiomatic authoring
#               roller / roll        infer + verify ONE parametric recipe over an axis
#               recipe_bundle        CBOR codec + bundle (rocke.bundle/v1)
#               online               ctypes binding to the C backend (recipe/IR -> .ll)
#   utils/    device-free helpers
#               recipe_expand        pure-Python recipe expander + recipes_equiv oracle
#   examples/ demo kernels / authoring emitters (each runnable: --emit recipe|ll|name)
#               recipe_toy mini_attn qk_block export_mha export_gemm_cshuffle
#               recipe_multi_result
#   drivers/  runnable harnesses / benchmarks / end-to-end
#               bench_online record_coverage roll_coverage
#               verify_recording_production roll_recipe
#               parity_matrix   byte-identity vs the Python lowerer (the gate)
#               gpu_replay      record -> CBOR -> C replay -> comgr -> launch
#   tests/    unittest suites
#   portable_ir_scaling_plan.md   the scaling plan (kept next to the code)
#
# ir_export (in rocke.core) serializes a KernelDef to portable IR (schema
# rocke.ir/v1). The C++ side (recipe VM, JSON/CBOR DOM, importer, online
# wrapper) lives in platform/cpp/portable_ir/, with its ctests and the
# standalone replay CLI in platform/tests/portable_ir/. The wire schemas are
# documented in dsl_docs/architecture/portable_ir_schema.md.
