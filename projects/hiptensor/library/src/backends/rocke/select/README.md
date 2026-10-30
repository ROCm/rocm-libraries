# Kernel selection models of the rocKE backend

The rocKE backend selects the kernel program of a plan in `hiptensorCreatePlan`
(`../rocke_select.hpp`): capability checks of every candidate, rocKE's tuned route tables, then
the selection model of the problem's class, then rocKE's rules. This directory holds the models.
The build writes them into the route table of the kernel pack of their target
(`HIPTENSOR_ROCKE_SELECT_MODEL_DIR`, default this directory); the library reads them from the pack.
A target without models selects by rocKE's rules.

## Layout

```
<target>/<class>.model     one model per problem class, for example gfx950/contraction.h.model
<target>/<class>.check     check vectors of a format v3 model (its exporter's predictions)
tools/gen_corpus.py        problem corpus of a training sweep
tools/sweep_driver.cpp     runs a corpus through hipTensor and records every candidate's time
tools/train_select.py      trains, exports and evaluates the models; proposes the next sweep round
```

Classes: `contraction.h` (f16, bf16, and f32 with a 16-bit compute type), `contraction.s` (f32),
`contraction.c` (complex f32), `contraction.d` (f64), `contraction.z` (complex f64),
`reduction.s` (f16, bf16, f32), `reduction.d` (f64). Contractions of f32 and complex f32 data
whose recipe (`HIPTENSOR_ROCKE_F32_RECIPE`) is not `x6` look up `<class>.<recipe>`, for example
`contraction.s.native` for the default recipe. The library also looks up `contraction` and
`reduction` (one model for all data types of an operation) when there is no model of the class.

## Model format

One line of text:

```
v3;schema=<n>;routes=<route>,...;margin=<m>;gemm=<sha256>|-;features=<name>,...;trees=<n>|<tree>|...
tree: <leaves>;<split features>;<thresholds>;<left children>;<right children>;<leaf values>
```

Lists are space separated. `routes` names the routes the model was trained on (the values of
`Route` in `../rocke_plan.hpp`, for example 7 for the ring GETT, 8 for the direct GETT and 11 for
the route of rocKE's tensor GETT planner); the model scores the candidates of those routes, and
rocKE's rules order the others (the planner's route first, then a preferred candidate, then the
rest after the scored ones). `margin`: the model's best candidate replaces the first scored
candidate of rocKE's static order only when its prediction is lower by more than the margin (0: always;
`train_select.py` chooses it per class on held-out problems). `gemm`: a model whose features
include `lg_pred_ns` (the predicted time of rocKE's tensor GETT planner, which depends on rocKE's
GEMM selection models) names the SHA-256 of the `index.txt` of the planner's GEMM selection models
it was trained with (`<pack dir>/gemm_models`, written by the pack build); the pack build fails when
it differs from the index it writes, and the library does not use the model (the plan log says why)
for a problem with a JIT candidate when it differs from the pack's (`-`: none named; such a model is
not used with JIT candidates either). Models of the earlier formats are still read:
`v2;schema=<n>;routes=...;margin=...;features=...;trees=...` (no planner models named) and
`v1;schema=<n>;features=...;trees=...` (the GEMM and strided GEMM routes only).

`<class>.check` (written by `train_select.py` next to every model it exports) holds feature vectors
of the model's training rows (every route it scores among them), each with the exported model's
prediction (the exporter's walk of the text) and LightGBM's. hipTensor's CPU test
`rocke_select_model_test` walks every shipped model of format v3 on its vectors with the library's
code: the first must match exactly, the second within the compact export's rounding.

Split features index the model's feature list; the library maps those names to its feature schema
(`kFeatureNames` in `../rocke_select.cpp`, version `kSelectSchema`). Schemas are append only, so a
model of an earlier schema is used; a model of a later schema is not. Node `n` goes to its left
child when `feature <= threshold`; a child `c < 0` is leaf `~c`; children come after their parent.
The prediction is the sum of the trees' leaves: the candidate's expected slowdown against the best
candidate of the problem, as log2. The lowest prediction wins; equal predictions keep rocKE's
order. A model the library cannot use (another format, a later schema, a feature the library
does not compute, malformed routes or trees) is not used (the selection then follows rocKE's rules,
and the plan log says why).

Features are integers (log2 values in 1/256 steps, counts, tile sizes, kernel resources from the
pack, the planner's route, predicted time and bytes moved of a JIT candidate), so the selection is
the same on every host for the same problem, device and pack. A JIT candidate has the same features
before and after its kernels are built. Schema 3 gives the JIT candidate its bytes moved (`lg_bytes`,
`intensity_q8`; 0 in schema 2); a model of schema 2 is scored with the values of schema 2.

The training corpus covers what the API admits: contractions with M and N modes and any number of
contracted modes, including none (outer products). Contractions with a batch mode (a mode of A, B
and D), a mode of one tensor only, or no M or no N mode are refused by `hiptensorCreateContraction`
and `hiptensorCreateContractionTrinary` (`HIPTENSOR_STATUS_NOT_SUPPORTED`), so no model scores them;
`train_select.py` refuses training rows with a batch mode.

## Retraining

The models must be retrained when the kernel pack's candidates change (new kernels, bodies or
routes, a new rocKE tree), when the feature schema changes, and for a new target. On a machine
with the target GPU and a hipTensor build with `-DHIPTENSOR_ENABLE_ROCKE_BACKEND=ON` (and
`-DHIPTENSOR_ENABLE_ROCKE_JIT=ON` for the JIT route):

1. Write a corpus: `python tools/gen_corpus.py contraction --seed 1 --per-row 1500 --aligned 0.6
   --padded 0.15 --out c.txt` (and `reduction`). It draws the GEMM-view extents of each data type
   from a space-filling sequence, splits them into modes and draws storage orders; `--aligned`
   makes a share of the extents multiples of 8 (rocKE's tensor GETT planner serves more of those),
   `--padded` pads a share of the strides; `--big` draws large problems (past the default design's
   flop and tensor caps), which the models need for the large rows of hipTensor's benchmarks;
   `--outer` makes a share of the problems outer products (no contracted mode); `--style api`
   draws the problems of the API validation harness (packed tensors, 1-3 modes per group).
2. Build the driver against the library (`tools/sweep_driver.cpp` uses the public API only) and time
   every candidate of every problem cold (a cache-evicting read before each timed call):
   `HIPTENSOR_BACKEND=rocke HIPTENSOR_BACKEND_FALLBACK=0 HIPTENSOR_ROCKE_SELECT_DUMP=dump.jsonl
   sweep_driver c.txt --default --jit --cold 5 --cold-out cold.jsonl`. The library writes every
   candidate's features to the dump; the driver writes each candidate's cold time and whether its
   output agrees with the f64 route's. (Without `--cold`, `HIPTENSOR_ALGO_DEFAULT_PATIENT` plans
   with `HIPTENSOR_ROCKE_AUTOTUNE_TOPK=all` time the candidates in the library, back to back.)
3. Train and export: `python tools/train_select.py train --dumps 'dump*.jsonl' --cold
   'cold*.jsonl' --calib 'dump_round1*.jsonl' --arch <target> --models . --report report.json
   --gemm-digest <sha256 of the build's share/hiptensor/rocke/<target>/gemm_models/index.txt>`.
   The model size, objective and margin are chosen on the held-out folds of the space-filling
   round's problems (`--calib`; the model-guided rounds oversample decision boundaries); a problem
   with a JIT candidate also trains without it (the candidates of a `HIPTENSOR_JIT_MODE_NONE` plan),
   and `--ensemble K` averages K models on bootstrap samples (the shipped gfx950 `contraction.h`
   model). It drops candidates whose
   output disagrees, reports the selection efficiency of the model on problems held out by group
   k-fold next to rocKE's rules, and checks that the exported trees reproduce LightGBM's
   predictions exactly.
4. Choose the next round's problems with the models (model-guided sweeps): write the features of a
   large problem pool without timing (`sweep_driver pool.txt --default --jit` with the dump and
   `HIPTENSOR_ROCKE_JIT_NO_COMPILE=1`), then `train_select.py propose --dumps ... --cold ...
   --pool 'pooldump*.jsonl' --n 2000 --out ids.txt --prev-models <target dir>`: bagged models pick
   the pool problems whose pick they disagree on or whose two best candidates are closest, plus a
   random share; `gen_corpus.py pick --from pool.txt --worst ids.txt --out round2.txt` writes them.
   Time them (step 2), retrain on every round, and stop when the picks of the pool no longer change
   between rounds.
5. Check a held-out corpus with the new models, without a new pack: `HIPTENSOR_ROCKE_SELECT_MODELS=
   <dir>` (models `<dir>/<target>/<class>.model`) replaces the pack's models; the dump of step 2
   then also records the library's scores and picks, and `train_select.py eval` reports the
   library's picks against the best candidate and that its scores equal the exported trees'
   predictions.
