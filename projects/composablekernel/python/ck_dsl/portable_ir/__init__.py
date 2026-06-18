# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Portable CK-DSL IR + builder-recipe tooling (Python authoring side).
#
# Modules:
#   ir_export (in ck_dsl.core) : KernelDef -> portable IR (schema ck.dsl.ir/v1)
#   recipe_recorder            : record a parametric recipe from idiomatic authoring
#   kerneldef_to_recipe        : KernelDef -> concrete recipe
#   roll_recipe                : roll a concrete recipe into a parametric one
#   recipe_toy / mini_attn / qk_block : example/demo kernels
#   export_mha                 : build the unified-attention 2D kernel for demos
#
# The C engine (recipe VM, JSON DOM, importer) lives in ../ck_dsl_c; the C
# drivers and shell runners live in ../ck_dsl_c/tests/portable_ir.
