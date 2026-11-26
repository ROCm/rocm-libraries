# Lessons

Why the rules in this skill exist. Read the section of the stage you are in; add a line when a run teaches something new.

## Oracle and judgeability

- SDPA backward computes `P = exp(S - LSE)` from a given fp32 LSE. Two correct fp32 computations of a score differ by
  `|S| x 2^-24`, so at a large `|S|` the LSE is off by far more than any tolerance, and no backward can pass. The harness reference
  looked right only because it shares the forward's arithmetic. Fix: bound the score in every graded space, and probe extreme scales
  with small Q and K. A 16-sample calibration shows this in minutes (*Calibrate*).
- A probe family at `buildable` enforcement that the engine accepts ends UNEXERCISED, which neither `declined` nor
  `declined_or_pass` allows. Make it gradeable instead.
- Draws that put most fp16 values below the subnormal threshold measure the hardware flush, not the parameter under test.
- A grader that ignores a feature (for example the mask of a padded graph) produces false failures across that whole feature. A
  feature-wide disagreement with the truth means the grader's rebuild, not the engine.
- AOTriton's rule (`test_error <= max(1e-5, ref_error x fudge)`): a PASS whose limit is at least half the tensor's largest value is
  near-vacuous; a FAIL with `ref_error == 0` is an exact-zero gradient filled with engine noise. Grade against two references: the
  truth from Q, K, V, and the fp64 backward from the given O and LSE; a row is an issue only if it fails both.

## Probes

- Copy a probe's `expect` from the engine's own matcher rule, never from another engine's space: the same overflow limit can be
  enforced by one matcher and not by another, and a copied expectation produces false violations.
- Tensors with extreme strides may not be fillable on the device; `HIPDNN_TEST_HOST_INPUT_FILL=1` fills inputs on the host.

## Triage

- A WRONG cited as "tail mismatches" without a domain A/B and a baseline is a guess. Replay with the default domain: rows that pass
  follow their inputs. Run the same graphs on a shipped engine or on base before calling a failure new.
- A BULK row (off by a large share of an output) deserves a replay before it is called a tolerance class.
- A failure that disappears on solo replay is state carried between cases in one process when reruns agree and `--batch-size 1`
  passes; it is not a race by default.
- A forced knob the engine cannot build reports CRASH at stage `build` with the engine's own `BAD_PARAM` text: by design.
- Engine TOML class rows match bundle case ids only; fuzz ids carry no class token and the gauntlet writes `bfp16`.

## Widening the net (ideas that paid off)

- Every head size the engine says it serves; asymmetric head sizes (needs `derive` for the V/O dims); GQA groups outside powers of
  two; head-dim pairs at ladder edges; long key sequences; a feature crossed with padding; a space per feature so its WRONG rate is
  its own number.
- A knob forced per case: `metadata.engine_variant` (`{"id": "<x>", "knobs": {"<name>": "<value>"}}`, see `running-tests.md`).
- Scratch variants with `--spaces-dir` (`scripts/make_variant_space.py`; the directory must sit next to `gauntlet/spaces/`; never
  committed; validate with `--generate-only`).

## Runs

- `--shrink` inside a budgeted pairwise run starves coverage: shrink afterwards.
- With several GPUs on one machine, set `HIP_VISIBLE_DEVICES` only. Setting `ROCR_VISIBLE_DEVICES` as well renumbers first and
  leaves the runtime with no device.
- Probe and stride runs with dumps can write hundreds of gigabytes: never dump them.
- Do not change a worktree, or a script, while a run is reading it: the driver reads spaces lazily and shell scripts are read as they
  run.
- Missing kernel-pack target: every accepted case CRASHes with `kpack_open failed`.
- The harness rewrites `<work>/bundles/` every batch: reproduce by seed and `--batch-size`, not by bundle.
- Space descriptions are long single lines: read the raw line before editing.
- A `-k` filtered test run missed an ordering test the full suite caught.
