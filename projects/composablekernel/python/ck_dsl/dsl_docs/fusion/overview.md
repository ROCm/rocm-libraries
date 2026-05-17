# Fusion Subsystem Overview

`helpers/fuse.py` and the surrounding `fusion_*` modules implement a minimal-but-complete graph-level fusion pipeline. The goal is to take a Python callable (typically a `torch.fx` symbolic graph), detect a known fusion pattern (matmul → bias-add → activation; gemm + cast; gemm + residual; etc.), and lower the pattern to a single CK DSL kernel via `UniversalGemmSpec` plus a `FusedEpilogue`.

This is a real subsystem; the README's `compile_fn` and `explain_fn` entry points use it. The architecture is built to extend: add an `EpilogueOp` for a new activation, add a pattern to `_PATTERN_TABLE`, and you get a fused kernel without touching the IR or lowering.

```text
Python fn
   │
   ▼
[Graph capture]   torch.fx.symbolic_trace
   │
   ▼
[Pattern match]   _PATTERN_TABLE  (extensible)
   │
   ▼
[Lower to CK DSL spec]   FusionLowering.lower_<pattern>
   │                     produces (UniversalGemmSpec, FusedEpilogue, args_builder)
   ▼
[Autotuner.select() + launch]   from helpers.autotune
```

## Entry Points

```python
from ck_dsl.helpers import compile_fn, explain_fn, fuse_matmul_bias_relu

# explain_fn: dry-run, show what the planner found.
explain_fn(my_fn)

# compile_fn: JIT-compile to a CK DSL kernel and return the launcher.
launcher = compile_fn(my_fn, example_inputs=(A, B, bias))
launcher(A, B, bias)

# fuse_matmul_bias_relu: convenience wrapper for the canonical pattern.
out = fuse_matmul_bias_relu(A, B, bias)
```

`torch_backend.py` wires `compile_fn` into `torch.compile(backend=...)`, so you can use it as a torch Inductor-style backend on the patterns it covers.

## Epilogue Ops (`helpers/fuse.py`)

```text
EpilogueOp (base)
BiasAdd       (+ bias[n])
Cast          (dtype conversion)
Clamp         (min/max bounds)
GELU          (approx="tanh")
ReLU          (max(0, x))
ResidualAdd   (+ skip)
ResidualMul   (* skip)
Scale         (* scale_scalar | * scale[n])
SiLU          (x * sigmoid(x))
```

Each `EpilogueOp` subclass implements:

```python
def emit(self, b: IRBuilder, dtype: Type, v: Value) -> Value:
    # IR emission for the per-element operation on v
    ...
```

Composition into a chain:

```python
fused = FusedEpilogue(ops=[
    BiasAdd(param_name="bias", dtype="fp16"),
    ReLU(),
    Cast(target_dtype="fp16"),
])
```

`BiasAdd` takes `(param_name, dtype)` not `(bias_dtype,)`. The kwarg `dtype` accepts either an IR `Type` or any string that `dtype_to_ir` recognizes (e.g. `"f16"`, `"fp16"`, `"bf16"`, or `torch.float16`).

`FusedEpilogue.lower(b, acc_f32, ...)` walks `ops` and threads the value through, leveraging the dtype dispatch helpers (`ir_dtype_zero`, `ir_dtype_const`, `ir_dtype_global_load`, `dtype_to_ir`) so each op doesn't grow N copies for each precision.

## Pattern Matching

`_PATTERN_TABLE` is the registry. Each entry:

- describes the FX subgraph to match;
- emits a `(UniversalGemmSpec, FusedEpilogue, args_builder)` triple;
- raises `FusionMatchError` if matching fails on the current callable.

Builtin patterns:

- `matmul → bias-add → relu`
- `matmul → bias-add → gelu`
- `matmul → bias-add → silu`
- `matmul → cast → residual-add`
- (and a small set of canonical variants — see `_PATTERN_TABLE` in `helpers/fuse.py`)

Adding a pattern is documented in `development/extending.md` §4. The contract:

1. Subclass `EpilogueOp` for any new activation / transform.
2. Add a pattern entry that walks the FX graph and returns the triple.
3. Test with `explain_fn` first; then `compile_fn`.

## Fusion IR (`helpers/fusion_ir.py`)

Intermediate representation between FX and CK DSL. Tensor / op / region / graph dataclasses used by the legalizer and scheduler. `build_graph(fn, example_inputs)` returns a `FusionGraph`.

## Legalizer (`helpers/fusion_legalize.py`)

`FusionLegalizer.legalize(graph) -> LegalResult` rejects fusion combinations that:

- exceed LDS budget after staging;
- require unsupported dtype combinations;
- violate alignment / vector-width constraints in the epilogue;
- need atomics in a non-atomic-compatible region.

## Lowering Registry (`helpers/fusion_lowering.py`)

```text
LoweringRegistry          # dispatch table: graph.op_kind -> Lowerer
ElementwiseLowerer        # per-element ops in the epilogue
ExplainOnlyLowerer        # records intent; emits no IR (used by explain_fn)
GemmEpilogueLowerer       # matmul + fused epilogue chain
ReductionLowerer          # block-wide reductions
default_lowering_registry # the registry used by compile_fn
BuiltRegion               # the lowered output of one region
```

Each lowerer turns a region of the `FusionGraph` into an `IRBuilder` emission plan.

## Workspace Planning (`helpers/fusion_memory.py`)

```text
WorkspaceAllocation
WorkspacePlanner
materialize_plan
```

Cross-kernel intermediates that escape a single fused region get a `WorkspaceAllocation` and an LDS/global slot. The planner does buffer-reuse so independent live ranges share storage. `materialize_plan(plan, pool)` turns the abstract plan into concrete `WorkspacePool` slots.

## Scheduler (`helpers/fusion_scheduler.py`)

```text
GreedyFusionScheduler
RegionCost
```

Picks fusion boundaries when multiple patterns could apply. Greedy by an estimated cost (rough memory + flops model). This is the hardest part of a real fusion compiler (Inductor inherits its scheduling from a cost model); the current implementation is deliberately simple.

## Validation (`helpers/fusion_validation.py`)

```text
BackendTiming
BenchmarkCase
FusionMatrixRunner
ValidationReport
run_fusion_validation_matrix
```

The matrix runner exercises a list of `BenchmarkCase`s through CK DSL fused vs unfused vs torch reference and produces a `ValidationReport` with per-case correctness and timing. This is the right harness for a fusion-pattern PR.

## Patterns This Subsystem Does *Not* Cover

Things you'd need to add before claiming Inductor parity:

- attention QKV fusion (the tiled attention kernels are not yet exposed through `compile_fn`);
- layernorm / RMSnorm fusion (those have their own kernel builders);
- cross-kernel scheduling with shared workspace beyond the simple matmul-epilogue case;
- buffer-reuse for intermediates that escape the fusion (the `WorkspacePlanner` framework exists, but is exercised mostly by the canonical matmul-epilogue paths).

The `helpers/fuse.py` module docstring is explicit about this: the file is the *proof* the architecture works end-to-end. Scaling up requires more patterns and a richer cost model on top of what's already there.

## When To Use Fusion vs A Direct Instance

- If the kernel exists as a shipped instance (universal GEMM, conv, attention, layernorm, etc.): use the instance directly. Fusion is for combining ops that don't already have a kernel.
- If the workload is captured by a torch.fx graph with one of the registered patterns: try `compile_fn`. Failing that, `explain_fn` will say what didn't match.
- If you need a new fusion: extend the pattern table and `FusedEpilogue`. See `development/extending.md` §4.

## Caveats

- The fusion subsystem is experimental relative to the production instance builders. Treat correctness errors as a sign that the pattern entry is wrong; treat performance regressions as a sign that the autotuner cache hasn't seen the new shape.
- `compile_fn` works best when input shapes are reasonably stable. Shape-polymorphic dispatch is supported by the autotuner cache, but each new shape pays a one-shot codegen cost.
- The torch.compile backend (`torch_backend.py`) is a passthrough; it lifts subgraphs that `compile_fn` matches and falls back to eager for the rest.
