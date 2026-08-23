# Fixed 22-feature linear arbiter — final result

## Contract

This experiment reused the existing frozen model unchanged:

- Model ID: `fresh-model-d745603256e5648f18dfbcf49316e38c565510e8dc7b16856fe0b886a70e0df4`
- Model SHA-256: `e4d35b1f5530dfa3dffb71b6b490e57ed256ba76464e2b2e680dbf4a47b8dd06`
- Features: original 22 from `offline_tournament.py`
- Weights: original 22 values
- Bias: none
- SK mode feature: none
- Tie policy: G0
- Retraining/threshold fitting: none

For every shape, GridBased first selected one SK0 candidate and Resource/Edge Origami selected one SK3 candidate. The frozen linear model compared only those two candidates.

## Locked retrospective result on confirmed 1,500-shape evidence

| Metric | Result |
|---|---:|
| G0 choices | 793 |
| O3 choices | 707 |
| Exact score ties (G0) | 61 |
| Correct faster-arm classification | 51.73% |
| Geomean versus always G0 | **99.73%** |
| Mean versus always G0 | **100.76%** |
| Median versus always G0 | **100.00%** |
| P10 / P5 versus G0 | **90.32% / 81.07%** |
| Minimum versus G0 | **45.50%** |
| Geomean versus always O3 | **102.23%** |
| Geomean fraction of best(G0,O3) | **94.13%** |

Interpretation: the fixed SK0-trained geometry model almost matches GridBased in aggregate and improves substantially over always-O3, but it is not a reliable oracle between the two systems. Its 51.7% arm accuracy and severe tail confirm that SK3 scoring is extrapolation.

## Real TensileLite runtime integration

Implemented `FixedLinearArbiterLibrary` with:

- GridBased child over 298 SK0 solutions;
- Resource/Edge Prediction child over 192 SK3 solutions;
- embedded frozen weights/model ID;
- exact original feature parsing/scoring;
- G0 tie handling;
- forced-G0 and forced-O3 debug modes;
- distinct `MatchingTag::FixedLinearArbiter`.

The combined library contains 490 solutions and is loaded via `HIPBLASLT_TENSILE_LIBPATH`. No `--solution_index` or `--algo_method all` is used.

## Verification

- C++ feature and choice tests: passed.
- Forced G0 reproduced standalone GridBased candidate.
- Forced O3 reproduced standalone Resource/Edge candidate.
- Automatic mode emitted both scores and selected the expected arm.
- Offline/runtime arm parity: **20/20** representative shapes.
- GPU correctness: **20/20**.
- `git diff --check`: clean.

Artifacts:

- Offline evaluation SHA-256: `40ca63cfc4e08877b0f85f93867ec98abc732d82f0ad873c3a8ec36d6c20a9e3`
- Runtime parity SHA-256: `aaf208a2cc124381acd6448d6332194b6bc7c5eb96d5ba1fa3c2bc82f7258667`
- Combined library hashes: `state/fixed_linear_library_hashes.txt`

## Limitation

The linear model does not parse `_SK0_`/`_SK3_` and was never trained for cross-mode arbitration. This result is a valid locked test of the original model, not evidence of calibrated SK3 prediction. No post-hoc changes were made.
