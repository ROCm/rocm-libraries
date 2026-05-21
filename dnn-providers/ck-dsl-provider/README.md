# ck-dsl-provider

hipDNN engine plugin that exposes kernels produced by the
Composable Kernel Python DSL (`ck_dsl`).

This is the M1 skeleton (milestone I-1): the plugin links and loads,
but its single engine (`CkDslConvImplicitGemmEngine`) reports nothing
applicable. JIT compilation and Python embedding land in later steps.

See `projects/composablekernel/python/ck_dsl/dsl_docs/hipdnn_provider/plan.md`
for the full plan.
