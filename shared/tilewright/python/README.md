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
results = pool.rank(problem, tw.Hardware(N_CU=64, lds_capacity=65536, L2_capacity=1 << 22))
```
