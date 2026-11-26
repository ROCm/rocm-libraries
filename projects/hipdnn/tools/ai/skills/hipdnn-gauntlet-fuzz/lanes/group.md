# Lane: failure group (`GROUP<k>`)

**Trigger:** one per group in `R/groups.tsv` (groups with one likely cause may share a lane). **Agent:** `task` (it replays on the
GPU within `Approved`). **Output:** `R/triage/GROUP<k>.md`, findings.

Explain the group: which kind ([../policy.md](../policy.md) *Kinds*) every row is, with the proof that kind requires. A WRONG count
is not a finding; a cause is.

## Method

1. **What.** For every row: tensor, mismatch count and percent, max abs diff against the tolerance in force, finite or not (from the
   failure folder's `log_tail.txt`, parsed by `sweep.json#<case>`: it is the batch log). Confirm the rows share an axis signature;
   split rows that do not into a separate finding.
2. **Tolerance class.** `class_check.py --root <work> --run <dir> --repo $WT --toml <engine.toml> --out cc.json` (plain host
   python3 with NumPy) assigns each row the class the bundle generator would and reads its atol/rtol from the TOML. Verdicts: WITHIN
   (inside the engine's contract), EXCEEDS (needs the replay), NO CLASS (outside the engine's claimed magnitudes), UNMODELLED
   (dropout, padding, ragged, bool bias: provisional). Then `triage_wrong.py --class-tol cc.json --no-domain` replays each row with
   its class tolerance. SdpaBackward only; elsewhere read the TOML by hand and say so.
3. **Domain A/B.** `triage_wrong.py` replays each row with the default domain: rows that pass follow their inputs.
4. **Two references** (SDPA backward). From the grader output: the truth recomputed from Q, K, V, and the fp64 backward from the O and
   LSE the engine was given. Fails both: engine. Matches the given-inputs reference only: conditioning. `harness_ref_vs_torch` above
   1% of the tensor's range: the oracle cannot judge the graph, or the grader's rebuild is wrong for this feature. For other ops,
   compare the CPU and GPU references: for bundle runs set `HIPDNN_TEST_VERIFICATION_MODE=cpu` or `gpu` and check the outcome record's
   `verifier` (`cpu_ref`, `gpu_ref`); `HIPDNN_TEST_REFERENCE_EXECUTOR` does not apply to bundle runs.
5. **Baseline.** The same graphs on base or on a shipped engine (another arch): pre-existing or not.
6. **Batch dependence.** Replay solo; if the failure disappears, follow [../policy.md](../policy.md) *Proof per kind* (rerun,
   `--batch-size 1`, 10 solo replays, base A/B with a restricted variant space).
7. **Signature checks.** Exact zeros where the reference is not zero (flush, unwritten output); NaN or Inf with a finite reference
   (unwritten or overflow); a failure at every value of an axis that should not matter (every probability, every alignment: a
   systematic mismatch such as a mask bit contract, not precision); BULK on small inputs (a real error).
8. **Repro.** For a `defect` or `regression`: the shrunk repro (`--row ... --shrink 20`) and the failure folder.

Do not stop at "wide domain" for a BULK row: the case that is off by a large share of an output deserves the replay before a class.
