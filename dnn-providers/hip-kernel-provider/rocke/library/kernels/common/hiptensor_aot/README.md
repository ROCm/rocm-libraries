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
| `dgett.py` | the direct GETT: `D = alpha * sum_K opA(A) opB(B) + beta * opC(C)` read and written in place (every data type, compute type, operator and layout; no workspace), register-staged MFMA tiles; the host plan in `gett_host.py` | `hiptensor-aot-dgett/v1` |
| `rgett.py` | rocKE's ring GETT body (`rocke.instances.common.tensor.gett`, f16 / bf16, gfx950) with its index maps as kernel arguments (one kernel per lane-width / D-vector class and digit-slot bucket: classes for problems of at most 2 or 4 digits per digit space skip the division of the unused slots), or as constants for a problem (the JIT form); the host plan in `gett_host.py` | `hiptensor-aot-rgett/v1` |
| `gett_host.py` | Python statement of the planner rules of `dgett` and `rgett` (K plans: in-tile and tile digits, padding of the K space) | - |
| `ewt.py` | tiled elementwise kernel for transposing layouts: inputs fastest along another mode than D read through an LDS tile, 16-byte vectors on both sides | `hiptensor-aot-ewt/v1` |
| `jit_ew.py` | the JIT form of `ew.py`: every integer argument (operator codes, sizes, magic numbers, strides, limits) a compile-time constant, and `vec` elements of mode 0 per thread with vector accesses where the layout allows; rolled at hipTensor's build time into one parametric recipe per kernel class (`roll_jit_ew`), which hipTensor's C++ runtime replays with the recipe VM for `HIPTENSOR_JIT_MODE_DEFAULT` plans; with no geometry, the same class with the geometry as kernel arguments (`hiptensor_aot_ewv...`, the vector classes of the kernel pack) | `hiptensor-jit-ew/v1`, `hiptensor-aot-ewv/v1` |

Rank bucket R8: at most 8 modes per index space after the host drops extent-1 modes and merges
contiguous modes. Coordinates are i32 (magic-number division); offsets i32 or i64 (`idx64`).
Operator codes are `hiptensorOperator_t` values: every unary operator, binary ADD, MUL, MAX, MIN
(MAX / MIN propagate NaN). Every kernel implements the unary IDENTITY, SQRT, RELU, CONJ, RCP,
SIGMOID, TANH, EXP, LOG, ABS, NEG (`common.BASIC_UNARY_OPS`); the `math` rows of the elementwise
and reduction kernels (kernel names ending in `_m`, pack attribute `math=1`) also SIN, COS, TAN,
SINH, COSH, ASIN, ACOS, ATAN, ASINH, ACOSH, ATANH, CEIL, FLOOR (`common.MATH_OPS`): the f64
programs of `rocke.helpers.elementary` (f32 compute: evaluated in f64 and rounded once to f32),
one wave-uniform branch per program group, constants anchored on an opaque zero so that the
kernels' loops do not hoist them into VGPRs (the anchored bit patterns are built from 32-bit
halves: recipes carry integers as doubles, exact to 2^53). The plain rows keep their code and
registers; a math row has one JIT class, every input a vector (its other layouts classify to vec 1,
which has no recipe: those launches keep the pack kernel). SIGMOID is `1 / (1 + t)` for x >= 0 and `t / (1 + t)`
below, `t = e^-|x|` (no overflow of `e^-x`, so large negative arguments keep their small
results). A tensor whose scale is zero is not read and its operator's result is dropped.

Conversions: every value rounds once to nearest even into the output type. Kernels with an f64
operand compute in f64; their f16 / bf16 outputs go through an f32 rounded to odd
(`Emit.f64_to_f32_odd`), which then rounds to nearest even exactly as one rounding from f64 would
(f32 has at least two more significand bits than f16 and bf16), so f64 -> f16 / bf16 never rounds
twice.

Tests of the contraction and tiled kernels: `library/tests/test_hiptensor_aot_dgett.py` (every compute class,
layouts, mixed types, operators, complex, zero scales with NaN / Inf, in-place C, batch modes, 64-bit offsets, K
padding; CPU: the K plans walk exactly the K space), `library/tests/test_hiptensor_aot_rgett.py` (the classes and the
JIT form bit for bit against float64 references; CPU: the plans against rocKE's GETT planner),
`library/tests/test_hiptensor_aot_ewt.py`.

Tests: `library/tests/test_hiptensor_aot_math_ops.py` (GPU: the math operators of every row,
generic and JIT forms, several inputs, reductions, specials, all exact against the host model of
`rocke.helpers.elementary`; CPU: rows, classes, recipes, their integers, and their replay by the C++ engine when the portable engine library is built), `library/tests/test_hiptensor_aot_numeric.py` (GPU, float64 references; the `row`
reduction cases need a wave64 device; the conversions against the exact rational rounding of
`library/tests/_hiptensor_aot_rounding.py`), `library/tests/test_hiptensor_aot_jit.py` (recipes
roll and reproduce the builder; specialized kernels against float64 references).
