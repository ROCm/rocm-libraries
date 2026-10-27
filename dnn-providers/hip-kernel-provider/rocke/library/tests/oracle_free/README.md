# Oracle-free P3 attention tests

This suite launches the shipped gfx942 dense attention kernel and checks masked
uniform probabilities without a complete attention oracle. Zero Q makes every
legal score equal. Four shifted-identity V launches expose all 256 probability
columns, and each expected output is derived directly from legal-key membership:

```text
V[k,d] = 1 when k = basis_offset + d, otherwise 0
O[q,d] = 1 / |A(q)| when basis_offset + d is legal, otherwise 0
```

The supported first matrix is:

| Dtype | Shape | Mask | Launches |
|---|---|---|---:|
| FP16 | `B1,Sq256,Sk256,H4/4,D64` | top-left causal | 4 |
| BF16 | `B1,Sq256,Sk256,H4/4,D64` | sliding window 64 | 4 |

Both cases require every logical output element to be finite, check every exposed
probability, require numerically exact zero for masked and basis-tail lanes, and
check combined row mass. Diagnostics focus on query rows `0`, `62`, `63`, `64`,
`127`, `128`, and `255`. The same real output must also reject the corresponding
off-by-one mask rule.

The checked-in absolute-tolerance candidates are fixed before the run:

| Metric | FP16 | BF16 |
|---|---:|---:|
| elementwise probability | `5e-4` | `4e-3` |
| combined row mass | `5e-3` | `2e-2` |

CPU contracts prove these ceilings cover unavoidable storage rounding and still
reject targeted mutations. They are not widened from a run under judgment. They
remain candidates until the full positive and wrong-model matrix is recorded on
matching gfx942 hardware; tighten them where repeated runs are stable.

## Oracle prohibition

Verdicts may use only deterministic profile construction, independent legal-key
arithmetic, scalar `1 / |A(q)|` expectations, and diagnostic reductions. They
must not call Torch SDPA, Rocke numeric references, SDPA reference helpers, or a
pinned-reference bundle. `_analytical.py` is Torch-free, and a CPU source contract
checks the suite for those dependencies.

P4 and P10 are intentionally deferred until this P3 matrix and its tolerances are
ratified.

## Running the suite

From the `rocke` directory:

```bash
# CPU formulas, mutations, source policy, and adapter validation.
python -m pytest library/tests/oracle_free/test_contracts.py -vv

# P3 production launches on a visible gfx942 device.
HIP_VISIBLE_DEVICES=0 python -m pytest \
  library/tests/oracle_free/test_analytical_gfx942.py -vv -rs

# Complete source suite.
HIP_VISIBLE_DEVICES=0 python -m pytest library/tests/oracle_free -vv -rs
```

From the installed `bin/hip_kernel_provider` directory:

```bash
ctest -R '^rocke_oracle_free_pytest$' \
  --output-on-failure --no-tests=error
```

The installed `rocke_oracle_free_pytest` CTest entry already collects this whole
directory; no CMake list is needed for new support files.
