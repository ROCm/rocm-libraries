# Fuzz policy

## Precedence

On conflict, the first source wins: the engine's own documented contract (its coverage doc and rule ids,
`projects/hipdnn/docs/KNOWN_DIVERGENCES.md`), then the harness reference, then this file. The harness reference is assumed right
**unless** the two-reference check ([lanes/group.md](lanes/group.md)) shows the graph is one no fixed tolerance can judge; then the
space is wrong, not the engine.

## Kinds

| Kind | Means | Typical evidence |
|---|---|---|
| `defect` | The engine computes a wrong answer, crashes, or writes NaN or nothing, on a graph it accepted and the oracle can judge | fails both references; does not follow the input domain; BULK, LARGE or NONFINITE on small inputs; a mask or bit mismatch at every probability |
| `regression` | PASS on base, WRONG, CRASH or DECLINED on the PR build, same graph (primary rows or same-seed random) | replay of `row.json` on both builds, shrunk repro on the PR build |
| `applicability` | Accepted a graph its rules forbid, or declines a graph it serves (lost coverage) | the rule id, the graph, a replay at full enforcement |
| `conditioning` | The graph is ill-posed for any implementation at the test dtype: correct answers differ by more than the tolerance | the engine matches the given-inputs reference but not the truth; the harness reference disagrees with the truth |
| `tolerance` | The engine is inside its own tolerance model (class tolerance, AOTriton's rule) and outside the flat one | `class_check.py` WITHIN, or a pass under the rule against both references |
| `divergence` | A documented difference from cuDNN or the reference (a hardware flush) | the doc entry and a graph that shows exactly that signature |
| `pre-existing` | The same graph fails on base or on the shipped engine | the base or shipped-engine run, same graph |
| `harness-gap` | The fuzzer or harness cannot build, fill or grade the graph (UNEXERCISED with its reason, a refused feature, a fill kernel that fails to launch) | the UNEXERCISED reason read from the outcome, the harness source that refuses it |
| `unknown` | Not settled | what would settle it |

## Severity

| Severity | Use for |
|---|---|
| `HIGH` | A silent wrong result or a crash on a graph a user can build and the engine accepts; any regression |
| `MEDIUM` | Accept-then-fail (an error after accepting); wrong only at extreme but legal inputs; an applicability hole; a probe violation the engine causes |
| `LOW` | Tolerance or documented divergence a user may hit; lost coverage (declines what it serves); a harness gap that hides a feature |
| `INFO` | Conditioning (the space's mistake, fixed in the space), pre-existing items, notes for the PR authors |

A `conditioning` or `tolerance` finding is never `HIGH`: if the engine is also wrong, that is a separate `defect` finding.

## Proof per kind

A finding states `probably` in `Verdict` unless its kind's proof is complete.

| Kind | Required proof |
|---|---|
| `defect` | Fails against both references (SDPA backward) or against the CPU and GPU references (other ops); a solo replay reproduces it; a shrunk repro; the failure does not follow the input domain (`triage_wrong.py` default-domain replay) |
| `regression` | The same graph PASSes on base and fails on the PR build, both replayed solo with `--row`; the shrunk repro on the PR build |
| `applicability` | The matcher rule id and line; the graph accepted (or declined); for accepted-forbidden, the replay at full enforcement showing what happens |
| `conditioning` | The engine matches the reference computed from its own given inputs; the score or magnitude bound the graph breaks |
| `tolerance` | `class_check.py` and the class-tolerance replay, or AOTriton's rule against both references; plus the same signature on a shipped engine or base at an accepted shape (else `probably`) |
| `divergence` | The doc entry; the observed signature matches it (which tensor, which elements, which inputs) |
| `pre-existing` | The base or shipped-engine outcome for the same graph |
| `harness-gap` | The reason string from the outcome, and the harness source line |

A failure that disappears on solo replay is **batch-dependent**: rerun the whole run, rerun with `--batch-size 1`, replay solo 10
times, then A/B on base with a variant space restricted to the failing family (same seed). It is a `defect` only after that.

## Judgeability (applies to every graded space)

A graph goes into a graded space only if a fixed tolerance can judge its correct answer: the answer must be stable to the rounding
of its own inputs. The rules are in `dnn-providers/integration-tests/docs/running-tests.md`, "Writing a space the oracle can
judge". For SDPA backward: `|scale| x dqk x mag^2 (+ |bias|) <= atol/10 x 2^24`. A family the engine may accept never runs at
`buildable` enforcement. Anything else goes to a probe with judgeable inputs, or is `unreachable`.

## Decisions for the user

Decide everything else alone and record it in `DECISIONS.md` and the report. Ask only for:

- the GPU plan, if not approved for this round;
- anything that leaves the workspace (PR comments, issues, pushes to branches that are not yours);
- a change to the PR's own code;
- whether a `HIGH` or `MEDIUM` finding is `ACCEPTED`;
- whether to stop iterating while the engine is not closed.
