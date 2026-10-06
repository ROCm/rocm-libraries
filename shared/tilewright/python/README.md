# tilewright Python bindings

Python bindings of the tilewright GEMM kernel ranking engine. They expose the
same API as the C++ library (`tilewright/model.hpp`): loading models, routing a
problem to its cell, computing features, and ranking a pool of candidate
kernels with `CandidateSet` or `rank_configs`.

```bash
pip install shared/tilewright/python
```

The package is built from the engine sources in the parent directory, so
install it from a source checkout. See `shared/tilewright/README.md` for the
engine, the model format and the API.

```python
import tilewright as tw

model = tw.load_model("model.tilewright.bin")
pool = tw.CandidateSet(model, [tw.Config(mt=tw.Dim3(256, 128, 64), mi=tw.Dim3(16, 16, 32))])
bf16 = tw.DataType.BFloat16
problem = tw.Problem(
    size=tw.Dim3(4096, 4096, 4096),
    a_dtype=bf16,
    b_dtype=bf16,
    c_dtype=bf16,
    d_dtype=bf16,
    mi_dtype=bf16,
)
hardware = tw.Hardware(N_CU=64, lds_capacity=65536, L2_capacity=1 << 22)
results = pool.rank(problem, hardware)
```

Kernels can carry named integer attributes, which the shipped models ignore:

```python
config = tw.Config(mt=tw.Dim3(256, 128, 64), mi=tw.Dim3(16, 16, 32),
                   attributes={"workgroup_mapping": 8, "global_split_u": 1})
config.attributes  # {'workgroup_mapping': 8, 'global_split_u': 1}, a copy
```

A ranking can name the execution context it is for; a model scores only the
contexts it was trained for (shipped models: every CU and `Schedule.Default`),
and leaves every kernel unscored otherwise:

```python
context = tw.ExecutionContext(cu_budget=32, schedule=tw.Schedule.Default)
tw.supports(model, context, hardware)  # False: a budget below hardware.N_CU
results = pool.rank(problem, hardware, context=context)  # all unscored
```

With `TieBreak.Prior`, configs with equal scores are ordered by ascending
`prior` (one float per config, non-finite values last), then by input order:

```python
pool = tw.CandidateSet(model, configs, tie_break=tw.TieBreak.Prior)
results = pool.rank(problem, hardware, prior=[0.5] * len(configs))
```

`context=None` means exclusive use of the device. Attribute names must be
strings and values `int64` integers (`TypeError` otherwise); `CandidateSet` and
`rank_configs` raise `ValueError` for an empty attribute name and for a prior of
the wrong length.
