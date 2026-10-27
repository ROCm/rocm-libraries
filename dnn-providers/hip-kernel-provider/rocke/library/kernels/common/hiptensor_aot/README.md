# hiptensor_aot: shape-generic kernels for hipTensor's rocKE backend

hipTensor (`projects/hiptensor`, option `HIPTENSOR_ENABLE_ROCKE_BACKEND=ON`) builds these kernels
at its build time into a kernel pack (`library/src/backends/rocke/pack/make_rocke_pack.py`) and launches them from C++ without
Python. Every kernel reads its problem (extents, strides, scalars, operator codes) from kernel
arguments, so one compiled kernel serves every problem of its data-type row.

| Module | Kernels | ABI |
|---|---|---|
| `ew.py` | `D = opABC(opAB(alpha*opA(A), beta*opB(B)), gamma*opC(C))`, arity 1 to 3, real or complex (pack / epilogue) | `hiptensor-aot-ew/v1` |
| `reduce.py` | `D = alpha*reduce(opA(A)) + beta*opC(C)`, `row` (innermost mode reduced, one wave per output), `col` (innermost mode kept, one thread per output), split partials + `final` | `hiptensor-aot-reduce/v1` |
| `host.py` | Python statement of the launch-argument rules (canonical modes, slots, magic numbers, grids, splits) that hipTensor's `rocke_plan.cpp` implements | - |
| `jit_ew.py` | the JIT form of `ew.py`: every integer argument (operator codes, sizes, magic numbers, strides, limits) a compile-time constant, and `vec` elements of mode 0 per thread with vector accesses where the layout allows; rolled at hipTensor's build time into one parametric recipe per kernel class (`roll_jit_ew`), which hipTensor's C++ runtime replays with the recipe VM for `HIPTENSOR_JIT_MODE_DEFAULT` plans | `hiptensor-jit-ew/v1` |

Rank bucket R8: at most 8 modes per index space after the host drops extent-1 modes and merges
contiguous modes. Coordinates are i32 (magic-number division); offsets i32 or i64 (`idx64`).
Operator codes are `hiptensorOperator_t` values; unary IDENTITY, SQRT, RELU, CONJ, RCP, SIGMOID,
TANH, EXP, LOG, ABS, NEG; binary ADD, MUL, MAX, MIN (MAX / MIN propagate NaN). A tensor whose
scale is zero is not read.

Conversions: every value rounds once to nearest even into the output type. Kernels with an f64
operand compute in f64; their f16 / bf16 outputs go through an f32 rounded to odd
(`Emit.f64_to_f32_odd`), which then rounds to nearest even exactly as one rounding from f64 would
(f32 has at least two more significand bits than f16 and bf16), so f64 -> f16 / bf16 never rounds
twice.

Tests: `library/tests/test_hiptensor_aot_numeric.py` (GPU, float64 references; the `row`
reduction cases need a wave64 device; the conversions against the exact rational rounding of
`library/tests/_hiptensor_aot_rounding.py`), `library/tests/test_hiptensor_aot_jit.py` (recipes
roll and reproduce the builder; specialized kernels against float64 references).
