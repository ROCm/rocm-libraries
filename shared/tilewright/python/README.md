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

A ranking can name the execution context it is for. A CU budget outside the
model's range is ranked at the nearest one in it (`scoring_context`), and a
schedule it does not list leaves every kernel unscored (for an MLREC_v2 model,
trained with every CU and `Schedule.Default`):

```python
context = tw.ExecutionContext(cu_budget=32, schedule=tw.Schedule.Default)
tw.supports(model, context, hardware)         # False: a budget below hardware.N_CU
tw.scoring_context(model, context, hardware)  # ExecutionContext(cu_budget=0, ...)
results = pool.rank(problem, hardware, context=context)  # ranked as every CU
dynamic = tw.ExecutionContext(schedule=tw.Schedule.Dynamic)
results = pool.rank(problem, hardware, context=dynamic)  # all unscored
```

`uniform_summation_order` (default `False`) and `workspace_bytes` (default
`2**64 - 1`, a workspace that holds every partial tile) describe the call's
launch. No feature reads them; a launch model plans with them, and
`scoring_context` keeps them.

A library directory's `tilewright_index` can list a model per PCI chip ID:

```python
# The stem's row for the device's chip ID, else its row without one; None when neither.
model = tw.load_model_by_index(stem, library_dir, pci_chip_id=device_chip_id)
# The same lookup, with the name of the model file the chosen row lists; its blocklist and
# guards are named after it (`<name up to its first '.'>.tilewright.blocklist`).
model, model_file = tw.load_model_by_index_with_file(stem, library_dir, device_chip_id)
```

With `TieBreak.Prior`, configs with equal scores are ordered by ascending
`prior` (one float per config, non-finite values last), then by input order:

```python
pool = tw.CandidateSet(model, configs, tie_break=tw.TieBreak.Prior)
results = pool.rank(problem, hardware, prior=[0.5] * len(configs))
```

`rank_top` returns only the best `k` scored results, `prepare` builds at load
time what the first rankings of each cell would build, and the cache limit
bounds the memory of expanded cell weights and per-cell caches:

```python
best = pool.rank_top(problem, hardware, 1)  # at most one Result
pool.prepare()  # every cell; prepare(cells=[...]) or prepare(problem=p, hardware=h) for fewer
tw.set_cache_limit(256 << 20)
tw.describe(model).provenance.config_id  # what an MLREC_v3 file records, else ""
```

An MLREC_v3 model may declare a drift fallback (see the engine README): a cell
whose pool lacks too many of its trained whitelisted view keys is served
degraded, which the candidate set decides once, when it is built:

```python
tw.describe(model).drift_keep_ppm  # its share in millionths, 0 without the fallback
pool.degraded_cells()  # the degraded cells' indices, ascending
```

It may also list stand-ins for its whitelisted view keys (see the engine
README): on a pool that lacks a trained whitelisted key, the first listed
stand-in the pool holds joins the cell's whitelist, which the candidate set
decides once, when it is built:

```python
tw.describe(model).standin_lists  # its stand-in records, 0 without stand-ins
pool.standin_cells()  # the cells whose whitelist stand-ins join, ascending
pool.standins(cell)  # the view-key hashes of those stand-ins, ascending
```

`describe` reports the model's other serving options as well, among them an
alignment veto and whether the model was trained with kernel guards (see the
engine README):

```python
tw.describe(model).alignment_veto.on  # with k_min and align_bits; False without one
tw.describe(model).kernel_guards  # True for a model trained with kernel guards
```

An MLREC_v3 model trained with a launch model records its id
(`provenance.launch_model`); `CandidateSet` then raises `ValueError` unless
`launch` has that `id` attribute:

```python
class Launch:
    id = "hipblaslt:gfx950:1"

    def plan(self, config, problem, context):
        return tw.LaunchPlan()

pool = tw.CandidateSet(model, configs, launch=Launch())
```

`context=None` means exclusive use of the device. Attribute names must be
strings and values `int64` integers (`TypeError` otherwise); `CandidateSet` and
`rank_configs` raise `ValueError` for an empty attribute name and for a prior of
the wrong length.
