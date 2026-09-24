# BF16 D128 CFVST Revalidation

## Background

The `gfx942` dense-attention kernel already contains a conflict-free V store-path transpose (CFVST) implementation for D128.

CFVST stores V in a transposed LDS layout so that the subsequent PV operand reads are contiguous and avoid the less efficient access pattern used by the baseline V path.

However, the CFVST was enabled only for FP16 D128 and not for BF16:

```python
dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx942/attention_dense.py

def _use_cfvst(head_size: int, dtype: str) -> bool:

    return _rows_per_instr(head_size) == 1 and dtype == "fp16"
```

BF16 D128 was excluded because an earlier configuration had been observed to exceed the desired register/resource budget.

After revalidation with CFVST enabled, register spilling is no longer observed. Infact, performance increases.

## Change

Enable the existing CFVST path for both supported D128 dtypes:

```python
dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx942/attention_dense.py

def _use_cfvst(head_size: int, dtype: str) -> bool:

    return _rows_per_instr(head_size) == 1
```

This changes the default policy for BF16 D128 from:

```text
CFVST = false
```

to:

```text
CFVST = true
```

No new kernel algorithm or layout is introduced. The change enables an existing path for an additional dtype.

The explicit `use_cfvst=False` path remains available for reproducing the previous behavior and for A/B testing.

## Validation

The old and new policies are compared using the same kernel implementation and workload.

For each BF16 D128 shape:

1. The baseline policy is selected, with CFVST disabled.
2. Numerical correctness is checked.
3. The candidate policy is selected, with CFVST enabled.
4. Numerical correctness is checked again.
5. Multiple same-session A/B timing rounds are run.
6. Odd rounds execute A then B; even rounds execute B then A to reduce systematic ordering bias.
7. Reduction and speedup are computed for each paired A/B round before taking the median.

The harness also verifies the resolved CFVST state before accepting a validation or timing result.

The tested cohort covers:

```text
B=1,  S=4096,  Hq=32, Hkv=8,  D=128
B=1,  S=4096,  Hq=32, Hkv=16, D=128
B=1,  S=8192,  Hq=32, Hkv=8,  D=128
B=1,  S=8192,  Hq=32, Hkv=16, D=128
B=1,  S=16384, Hq=32, Hkv=8,  D=128
B=16, S=4096,  Hq=32, Hkv=8,  D=128
B=16, S=4096,  Hq=32, Hkv=16, D=128
B=16, S=8192,  Hq=32, Hkv=8,  D=128
```

All measurements use BF16, causal attention, and D128.

Measured performance results are retained in the approved results location.
## Reproduce

Create the standard rocKE development environment:

```bash
cmake -S dnn-providers/hip-kernel-provider -B build \
    -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON

cmake --build build --target rocke-pyenv
```

Run the CFVST A/B experiment:

```bash
dnn-providers/hip-kernel-provider/rocke/library/builders/gfx942/attention/prefill/run_ab_cfvst.sh
```

By default, the harness performs five A/B rounds per shape.

A different number of rounds can be supplied as the first argument:

```bash
dnn-providers/hip-kernel-provider/rocke/library/builders/gfx942/attention/prefill/run_ab_cfvst.sh 10
```

The harness temporarily switches between the previous and candidate CFVST policies and restores the original `attention_dense.py` when it exits.

Raw timings and validation logs are written under:

```text
results/
```

## Result

The current BF16 D128 kernel no longer reproduces the condition that originally justified excluding it from CFVST.

Numerical validation passes with both the previous and candidate policies, and same-session A/B measurements support enabling the existing CFVST path by default for BF16 D128.

The change therefore removes the BF16-specific restriction while retaining the explicit non-CFVST path for comparison and debugging.

