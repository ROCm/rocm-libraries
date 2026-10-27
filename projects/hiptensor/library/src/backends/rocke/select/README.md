# Kernel selection models of the rocKE backend

The rocKE backend selects the kernel program of a plan in `hiptensorCreatePlan`
(`../rocke_select.hpp`): capability checks of every candidate, rocKE's tuned route table, then
the selection model of the problem's class, then rocKE's rules. This directory holds the models.
The build writes them into the route table of the kernel pack of their target
(`HIPTENSOR_ROCKE_SELECT_MODEL_DIR`, default this directory); the library reads them from the pack.
A target without models selects by rocKE's rules.

## Layout

```
<target>/<class>.model     one model per problem class, for example gfx950/contraction.h.model
tools/gen_corpus.py        problem corpus of a training sweep
tools/sweep_driver.cpp     runs a corpus through hipTensor and records every candidate's time
tools/train_select.py      trains, exports and evaluates the models
```

Classes: `contraction.h` (f16, bf16, and f32 with a 16-bit compute type), `contraction.s` (f32),
`contraction.c` (complex f32), `contraction.d` (f64), `contraction.z` (complex f64),
`reduction.s` (f16, bf16, f32), `reduction.d` (f64). The library also looks up `contraction` and
`reduction` (one model for all data types of an operation) when there is no model of the class.

## Model format

One line of text:

```
v1;schema=<n>;features=<name>,<name>,...;trees=<n>|<tree>|<tree>|...
tree: <leaves>;<split features>;<thresholds>;<left children>;<right children>;<leaf values>
```

Lists are space separated. Split features index the model's feature list; the library maps those
names to its feature schema (`kFeatureNames` in `../rocke_select.cpp`, version `kSelectSchema`).
Node `n` goes to its left child when `feature <= threshold`; a child `c < 0` is leaf `~c`;
children come after their parent. The prediction is the sum of the trees' leaves: the
candidate's expected slowdown against the best candidate of the problem, as log2. The lowest
prediction wins; equal predictions keep rocKE's order. A model of another schema, or one that reads
a feature the library does not compute, is not used (the selection then follows rocKE's rules, and
the plan log says why).

Features are integers (log2 values in 1/256 steps, counts, tile sizes, kernel resources from the
pack), so the selection is the same on every host for the same problem, device and pack.

## Retraining

The models must be retrained when the kernel pack's candidates change (new kernels or bodies, a
new rocKE tree), when the feature schema changes, and for a new target. On a machine with the
target GPU and a hipTensor build with `-DHIPTENSOR_ENABLE_ROCKE_BACKEND=ON`:

1. Write a corpus: `python tools/gen_corpus.py contraction --seed 1 --per-row 1500 --out c.txt`
   (and `reduction`). It draws the GEMM-view extents of each data type from a space-filling
   sequence, splits them into modes and draws storage orders.
2. Build the driver against the library (`tools/sweep_driver.cpp` uses the public API only) and run
   the corpus with every candidate timed:
   `HIPTENSOR_BACKEND=rocke HIPTENSOR_BACKEND_FALLBACK=0 HIPTENSOR_ROCKE_AUTOTUNE_TOPK=all
   HIPTENSOR_ROCKE_SELECT_DUMP=dump.jsonl sweep_driver c.txt`. Each problem is one
   `HIPTENSOR_ALGO_DEFAULT_PATIENT` plan, whose measured selection appends every candidate's
   features and median time to the dump.
3. Train and export: `python tools/train_select.py train --dumps 'dump*.jsonl' --arch <target>
   --models . --report report.json`. It reports the selection efficiency of the model on problems
   held out by group k-fold next to rocKE's rules, and checks that the exported trees reproduce
   LightGBM's predictions exactly.
4. Check a held-out corpus with the new models in the pack: the dump of step 2 then also records
   the library's scores and order, and `train_select.py eval` reports the library's picks and that
   its scores equal the exported trees' predictions.
