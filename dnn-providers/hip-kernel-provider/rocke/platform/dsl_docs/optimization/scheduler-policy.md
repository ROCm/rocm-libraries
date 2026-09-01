# AMDGPU Scheduler Policy

The AMDGPU machine scheduler can materially change instruction ordering,
register pressure, memory-clause formation, and occupancy without changing a
kernel's algorithm. rocKE exposes the scheduler choice as a validated,
per-kernel code-generation policy so it can be swept as a bounded performance
tuning knob while remaining part of artifact and cache identity.

## Table of Contents

- [When to use this knob](#when-to-use-this-knob)
- [Supported strategies](#supported-strategies)
- [Apply a policy](#apply-a-policy)
- [Run a bounded sweep](#run-a-bounded-sweep)
- [Validate a candidate](#validate-a-candidate)
- [Artifact and cache identity](#artifact-and-cache-identity)
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
        name="baseline-shape",
        extra={"codegen_policy": policy},
    )
    for policy in policies
]

def build_candidate(config):
    kernel = build_kernel(config.spec, arch="gfx950")
    apply_codegen_policy(kernel, config.codegen_policy)
    return kernel
```

`extra["codegen_policy"]` is reserved and must contain a `CodegenPolicy`.
`AutotuneConfig.identity` includes its policy key when a non-default policy is
present. This preserves the existing constructor signature, permits candidates
with the same human-readable name, and avoids collisions in persistent winner
caches.

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

## Artifact and cache identity

Scheduler policy changes the compiled object, so it is provenance rather than
an incidental benchmark label:

- `KernelArtifact.codegen_policy` records the validated policy used to lower
  the kernel.
- Generated manifests contain `codegen_policy` and `codegen_policy_key`.
- `KernelId.with_codegen_policy(...)` adds the policy key to compile and
  selection identities.
- `AutotuneConfig.identity` adds the policy key to persistent tuning records.

The default policy retains existing dispatcher and autotuner identity strings.
Do not reuse a cached HSACO compiled under one explicit policy for another.

## Raw compiler options

The typed scheduler policy is the supported path for durable kernel tuning.
`compile_kernel()` intentionally does not accept an arbitrary `options` list.
For isolated compiler diagnostics, lower to LLVM IR and call
`build_hsaco_from_llvm_ir(..., options=[...])` directly. Raw options are not
automatically validated or included in rocKE artifact identity, so they must
not be used for persistent dispatch or autotune decisions.

## Limitations

- The policy applies to the LLVM-direct COMGR path used by `compile_kernel()`.
  `compile_kernel_via_hipcc()` rejects an explicit scheduler policy because the
  HIP lowering path does not currently carry this LLVM function attribute.
- Strategy names are an LLVM backend interface. Revalidate generated code and
  numeric correctness when changing the LLVM or ROCm toolchain.
- The policy controls machine scheduling only. It does not replace kernel-level
  scheduling, synchronization, tiling, memory-layout, or occupancy decisions.
