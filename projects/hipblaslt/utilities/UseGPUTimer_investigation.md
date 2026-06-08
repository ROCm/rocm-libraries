# `UseGPUTimer` YAML Parameter Investigation

## Verdict

**(B) Dead-on-arrival.** `UseGPUTimer` was **never** a registered entry in the Tensile `globalParameters` dict and was **never** read by any Python or C++ code in this repository, at any point in the accessible git history. It appears in 5 YAML files under `tensilelite/Tensile/Tests/common/gemm/gfx12/`, each of which was authored with `UseGPUTimer: False` from its first commit, alongside the correct registered key `KernelTime: False` for the same setting. The two keys are structural duplicates in every file that carries `UseGPUTimer`; removing `UseGPUTimer` changes nothing at runtime.

`UseGPUTimer` is NOT the legacy name for `KernelTime`. They are unrelated in the Tensile parameter registry. `use_gpu_timer` (snake_case) does exist as a live argument in the hipBLASLt bench/test client (`clients/bench/src/client.cpp`, `clients/common/include/hipblaslt_arguments.hpp`), and `KernelTime` in the Tensile global registry maps to the `--use-gpu-timer` CLI flag passed to the Tensile benchmark client (`ClientWriter.py:691`). The YAML key `UseGPUTimer` has never been part of either wiring.

---

## Step 1 — Live-code search at HEAD

`git grep -n UseGPUTimer -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml' '*.md' '*.rst'` returned **two matches**, neither of which is a parameter reader:

| File | What it is |
|------|-----------|
| `projects/hipblaslt/utilities/input_yaml_validation_implementation_audit.md` | An audit document that inventories `UseGPUTimer: 5 files` as a known unknown-key offender |
| `projects/hipblaslt/utilities/UseGPUTimer_investigation.md` | This file |

There are **9 YAML occurrences** (5 live, 4 commented out) in:

```
tensilelite/Tensile/Tests/common/gemm/gfx12/f4b8ss_gfx1250.yaml:28:  # UseGPUTimer: False
tensilelite/Tensile/Tests/common/gemm/gfx12/f8b8ss_gfx1250.yaml:28:  UseGPUTimer: False
tensilelite/Tensile/Tests/common/gemm/gfx12/f8f4ss_gfx1250.yaml:28:  # UseGPUTimer: False
tensilelite/Tensile/Tests/common/gemm/gfx12/fp8_gfx1250.yaml:28:  UseGPUTimer: False
tensilelite/Tensile/Tests/common/gemm/gfx12/i8ii_gfx1250.yaml:30:  UseGPUTimer: False
tensilelite/Tensile/Tests/common/gemm/gfx12/mxf8_gfx1250.yaml:29:  UseGPUTimer: False
tensilelite/Tensile/Tests/common/gemm/gfx12/mxf8f4ss_gfx1250.yaml:30:  # UseGPUTimer: False
tensilelite/Tensile/Tests/common/gemm/gfx12/tdm_multicast_gfx1250.yaml:29:  UseGPUTimer: False
tensilelite/Tensile/Tests/common/streamk/gfx1250/sk_mxf8f4gemm_quick.yaml:31:  # UseGPUTimer: False
```

Every file that carries a live (uncommented) `UseGPUTimer: False` also carries `KernelTime: False` at an earlier line in the same `GlobalParameters:` block. `KernelTime` is the real registered parameter (at `GlobalParameters.py:54`).

Conclusion from Step 1: **no live reader exists at HEAD** — proceeding to Step 2.

---

## Step 2 — Git history archaeology

### Ever in a code file?

`git log --all --oneline -S 'UseGPUTimer' -- '*.py' '*.cpp' '*.cc' '*.h' '*.hpp' '*.cu' '*.cmake' '*.json' '*.toml'`
returned **no commits**. `UseGPUTimer` has never appeared in any Python, C++, CMake, or JSON file in the accessible history of this repository.

### Introduction in YAML files

`git log --all --oneline -S 'UseGPUTimer' -- '*.yaml' '*.yml'` returned results only on the current feature branch `users/alvasile/input_yaml` and its ancestors for the gfx12 test YAML files.

The earliest ancestor commit touching `f8b8ss_gfx1250.yaml` is `4f8ab911cb51` ("Fix f8b8 with f8f6f4 instructions. (#111)"), where `UseGPUTimer: False` was already present in the `+` side of the diff — meaning it was there from file creation. There is no prior commit where the file existed without `UseGPUTimer`. The file was authored with both `KernelTime: False` and `UseGPUTimer: False` simultaneously; the author appears to have double-entered the timer toggle under two different names.

The same pattern holds for `fp8_gfx1250.yaml`, `i8ii_gfx1250.yaml`, `mxf8_gfx1250.yaml`, and `tdm_multicast_gfx1250.yaml` — all gfx1250-targeted files authored together as a set.

### Why `KernelTime` is the correct name

`ClientWriter.py:691`:
```python
param("use-gpu-timer", globalParameters["KernelTime"])
```

`GlobalParameters.py:54`:
```python
globalParameters["KernelTime"] = False  # T=use device timers, F=use host timers
```

`KernelTime` is the Tensile global parameter that controls GPU vs. host timing. It is passed to the benchmark client as `--use-gpu-timer`. The hipBLASLt test client has a separate `use_gpu_timer` argument field (snake_case) with its own wiring — this is a different system and is not connected to the Tensile `GlobalParameters` YAML block.

---

## Recommendation

Delete `UseGPUTimer: False` (and the commented-out `# UseGPUTimer: False`) from all 9 affected YAML files. The line has never had any effect. `KernelTime: False` in the same files performs the intended function.

Do not add `UseGPUTimer` to `globalParameters` or to `ignoreKeys` — it was never a valid parameter and there is no backward-compat reason to retain it.

The 5 files with live (uncommented) `UseGPUTimer: False` will trigger `ConfigTypeError: Unknown global parameter` once the strict gate is active. The 4 commented-out occurrences are cosmetically noisy but harmless.

---

## Worktree path

`/home/alvasile/rocm-libraries/.claude/worktrees/agent-a4a141376902c40ee`
