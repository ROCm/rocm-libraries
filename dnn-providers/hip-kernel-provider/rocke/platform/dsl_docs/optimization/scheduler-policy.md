# AMDGPU Scheduler Policy

The AMDGPU machine scheduler can materially change instruction ordering,
register pressure, memory-clause formation, and occupancy without changing a
kernel's algorithm. rocKE exposes the scheduler choice as a validated,
per-kernel code-generation policy so it can be swept as a bounded performance
tuning knob.

## Table of Contents

- [When to use this knob](#when-to-use-this-knob)
- [Supported strategies](#supported-strategies)
- [Apply a policy](#apply-a-policy)
- [Run a bounded sweep](#run-a-bounded-sweep)
- [Validate a candidate](#validate-a-candidate)
- [Artifact and cache handling](#artifact-and-cache-handling)
- [Raw compiler options](#raw-compiler-options)
- [Limitations](#limitations)

## When to use this knob

Start with the structural levers in the
[optimization runbook](./optimization_runbook.md): algorithm, tile geometry,
MFMA atom, memory layout, pipeline, and epilogue. Scheduler policy is useful
after correctness is stable and static inspection suggests that instruction
ordering, waits, register pressure, or occupancy may be limiting the kernel.

Treat each strategy as a hypothesis, not as an architecture-wide default. A
strategy that helps one target or shape may regress another, and the same name
may behave differently across LLVM versions.

## Supported strategies

`SchedulerStrategy` is a closed set:

| Value | Tuning intent |
|---|---|
| `max-ilp` | Favor instruction-level parallelism. |
| `max-memory-clause` | Favor formation of larger memory clauses. |
| `iterative-ilp` | Use the iterative scheduler with an ILP objective. |
| `iterative-minreg` | Use the iterative scheduler with a register-pressure objective. |
| `iterative-maxocc` | Use the iterative scheduler with an occupancy objective. |

The default is `None`. It omits the LLVM attribute and preserves the compiler's
normal scheduler selection. Unsupported strings, empty strings, and non-string
values are rejected before compilation.

## Apply a policy

Build the kernel normally, then attach a typed policy before lowering:

```python
from rocke import CodegenPolicy, SchedulerStrategy, apply_codegen_policy
from rocke.helpers import compile_kernel

kernel = build_kernel(spec, arch="gfx950")
apply_codegen_policy(
    kernel,
    CodegenPolicy(scheduler_strategy=SchedulerStrategy.ITERATIVE_ILP),
)
artifact = compile_kernel(kernel, arch="gfx950")
```

The LLVM lowerer emits the function attribute:

```llvm
"amdgpu-sched-strategy"="iterative-ilp"
```

Apply `CodegenPolicy()` to remove an existing scheduler policy and return to
the compiler default. The policy is carried by `KernelDef`, so it survives the
serialized-IR handoff to the C++ lowering engine. The Python and C++ engines
emit the attribute in the same position.

## Run a bounded sweep

Sweep only the default plus the supported strategies, and build a fresh kernel
for each candidate:

```python
from rocke import CodegenPolicy, SchedulerStrategy, apply_codegen_policy
from rocke.helpers import AutotuneConfig

policies = [CodegenPolicy()] + [
    CodegenPolicy(scheduler_strategy=strategy)
    for strategy in SchedulerStrategy
]
configs = [
    AutotuneConfig(
        spec=spec,
        name=f"baseline-shape-{policy.scheduler_strategy or 'default'}",
        extra={"codegen_policy": policy},
    )
    for policy in policies
]

def build_candidate(config):
    kernel = build_kernel(config.spec, arch="gfx950")
    apply_codegen_policy(kernel, config.extra["codegen_policy"])
    return kernel
```

`extra` is the existing caller-owned extension point, so this does not change
the `AutotuneConfig(spec, name, extra={})` signature. Use a distinct `name` for
every policy variant: the current autotuner persists winners by config name and
does not derive a compiled-object identity from `KernelDef`.

## Validate a candidate

Use the runbook's correctness-first loop. For every target, LLVM version, and
representative shape cohort:

1. Compile the default and each supported strategy from otherwise identical
   `KernelDef` input.
2. Run the kernel's numeric parity or verification harness before timing it.
3. Inspect HSACO resource metadata for VGPR, SGPR, LDS, scratch, and occupancy
   changes.
4. Inspect ISA for spills, wait placement, memory-clause changes, and the
   instruction mix in the hot loop.
5. Benchmark in the same process and stream with the established harness.
6. Retest neighboring shapes and every supported target before promoting a
   winner into dispatch policy.

Reject a candidate that is faster only because correctness changed, spills
appeared, occupancy collapsed unexpectedly, or results do not reproduce across
the intended toolchain range.

## Artifact and cache handling

Scheduler policy changes the compiled object even though it does not change the
kernel's launch ABI. Treat policy variants like the existing `agpr_alloc`
control:

- build a fresh `KernelDef` for each policy;
- use distinct autotune config names and output/cache locations;
- retain the serialized IR or LLVM IR with the compiled HSACO when provenance
  matters; and
- do not reuse a cached HSACO across policy variants.

rocKE does not currently have a generic compiled-object identity that covers
all compile-affecting kernel attributes and toolchain inputs. Scheduler policy
does not add a one-off identity field to dispatcher IDs or runtime manifests;
that broader cache design is tracked separately.

## Raw compiler options

The typed scheduler policy is the supported path for durable kernel tuning.
`compile_kernel()` intentionally does not accept an arbitrary `options` list.
For isolated compiler diagnostics, lower to LLVM IR and call
`build_hsaco_from_llvm_ir(..., options=[...])` directly. Raw options are not
automatically validated or included in cache identity, so they must not be used
for persistent dispatch or autotune decisions.

## Limitations

- The policy applies to the LLVM-direct COMGR path used by `compile_kernel()`.
  `compile_kernel_via_hipcc()` rejects an explicit scheduler policy because the
  HIP lowering path does not currently carry this LLVM function attribute.
- Strategy names are an LLVM backend interface. Revalidate generated code and
  numeric correctness when changing the LLVM or ROCm toolchain.
- The policy controls machine scheduling only. It does not replace kernel-level
  scheduling, synchronization, tiling, memory-layout, or occupancy decisions.
