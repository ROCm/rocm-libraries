# RDNA general SDPA backward tuned-table selection — case study

**Scope.** gfx1151, gfx1201. General family only (the dense family is not
built on RDNA). 27 tuning keys per arch, all at fill bucket `over` (the grid
of every entry's representative requests is at least twice the device's
`num_cus` — see the caveat in §3).

**Result.** On every shipped key, the entry's finalist beats rocKE's shipped
plan: gfx1201 **0.24×–0.28×** of shipped at `d=64` (10 keys), **0.50×–0.53×**
at `d=128` (17 keys); gfx1151 **0.77×–0.87×** at `d=64`, **0.56×–0.67×** at
`d=128` (ranges of the per-key geometric means). Against PyTorch's own
backward the two arches differ (§2): on gfx1201 PyTorch is faster than both
the shipped plan and the entries on every measured cell; on gfx1151 it is
faster on most cells but not all. Either way the CDNA rule (an entry must
beat the competitor on every cell of its key) would leave most RDNA keys at
the defaults, so both arches use the rule in §1 instead.

Regeneration: `benchmarks/common/attention_bwd_rdna_entry_rule.py` (checked,
unit-tested selection logic); the judge documents and constants that fed it
live outside the source tree (`tuned_tables.py` "Regeneration").

---

## 1. The rule measured here

The rule is stated once, in `select_rdna_general_entries`
(`benchmarks/common/attention_bwd_rdna_entry_rule.py`): per tuning key, the
`runtime` finalist that is correct, A/A-clean and not slower than rocKE's
shipped plan on every dense census cell of the key, with the lowest
geometric-mean time ratio against shipped — only when that ratio is below 1.
"Not slower" is the bench's paired verdict, which resolves a near-tie with its
gate constant `TIE_EPS`; the same module states the gate's false-positive
behaviour.

The gate constants used were calibrated per device from the sweep's own A/A
spread (the repo's private shipped constants do not cover these devices);
their values are kept with the sweep outside the source tree, not here, per
the project's performance-evidence policy (`platform/AGENTS.md` "Public
software-performance evidence": relative results only, in a qualifying
document such as this one).

## 2. What was measured, and with what

| item | value |
|---|---|
| rocKE commit measured | `84518de35e2` (this PR's branch: develop `4cd9d47d91f` plus empty RDNA tables and test-only changes) |
| baseline ("shipped plan") | the untuned default plan at that commit (the RDNA general tables were empty) |
| rocKE code generation | LLVM flavor `llvm22`; comgr from ROCm 7.2 (`libamd_comgr.so`, the one in the PyTorch wheel) |
| HIP runtime | 7.2.53211 |
| competitor | PyTorch `2.13.0+rocm7.2`, `torch.nn.functional.scaled_dot_product_attention` backward, default backend selection (bench arm `default`) |
| gfx1201 device | one Radeon RX 9070-class card, 32 CUs reported, both slices |
| gfx1151 device | Strix Halo APUs: `d=64` timed on a node reporting 16 CUs, `d=128` on one reporting 20 CUs (§3) |
| cells | dense, non-THD census cohort cells: 23 at `d=64`, 55 at `d=128`, per arch |
| harness | `attention_bwd_bench.py`, shipped + 2 finalists + PyTorch, each rocKE arm with an `#aa` twin |

PyTorch's backward against the same cells (time ratio PyTorch / rocKE; below
1 means PyTorch is faster):

| arch | `d` | vs shipped plan: PyTorch faster on | ratio range (PyTorch / shipped plan) | vs shipped entry: keys where the entry is faster on every cell |
|---|---|---|---|---|
| gfx1201 | 64 | 23 / 23 cells | 0.025×–0.12× | 0 / 10 |
| gfx1201 | 128 | 55 / 55 cells | 0.029×–0.27× | 0 / 17 |
| gfx1151 | 64 | 22 / 23 cells | 0.42×–10.1× | 1 / 10 |
| gfx1151 | 128 | 46 / 55 cells | 0.34×–4.8× | 4 / 17 |

The gfx1151 keys where the entry beats PyTorch on every cell are mostly
windowed (`window` at `d=64` and `d=128`, and one `causal_tl` GQA key with
unequal lengths at `d=128`); they cover 14 of the 78 gfx1151 cells. That
comparison is by median and was not run through the bench's paired
`confirm` stage; it is reported here and does not affect any entry, since
the RDNA rule compares against the shipped plan only.

## 3. Config-labeled coverage

Every varying axis of the 27 keys per arch (both arches ship the same 27
tuning keys):

| axis | values measured | count |
|---|---|---:|
| head size (`d`) | 64, 128 | 10, 17 |
| dtype | bf16, fp16 | 20, 7 |
| mask kind | none, causal (top-left), causal (bottom-right), window | 7, 11, 5, 4 |
| head mode | GQA, MHA | 18, 9 |
| length relation | square (`s_q == s_kv`), unequal | 21, 6 |
| kv-tile bucket | `t64`, `t>64` | 7, 20 |
| length mode | fixed | 27 |
| stage_vec | 8 | 27 |
| fill bucket (**sensitive axis**) | `over` (grid ≥ 2 × `num_cus`) | 27 |

**The sensitive axis is fill bucket.** Every entry is measured, and only
applies, at `over`: the request's grid fills the device at least twice over.
A request of the same shape at `under` or `full` (fewer batches, or a device
with many more CUs than the sweep's) keeps the defaults — it is a different,
unmeasured tuning key, not an extrapolation of these entries. Head size 32 is
not covered at all (the census has no `d=32` cell on any arch).

## 4. Caveats

* **Two gfx1151 nodes.** The gfx1151 `d=64` cells were timed on a node
  reporting 16 CUs and the `d=128` cells on one reporting 20 CUs (different
  Strix Halo parts). Every measured cell falls in the same tuning key at
  either count (all `over`), so the shipped keys do not depend on which node
  was used, but the `d=64` ratios are from the smaller part.
* **One route, not two.** RDNA has no dense family, so there is no second
  route to gate against the oracle here; the general route's own numeric
  correctness gate (an independent fp32 reference, declared dtype and
  tolerance) is the bench's pruning-set gate, applied before any cell is
  timed (criterion 1 of the rule in §1).
* **Calibrated, not shipped, gate constants.** The repository ships no RDNA
  gate constants; the `TIE_EPS` the bench used to resolve near-ties was
  calibrated per device from this sweep's own measurements, not taken from an
  approved reference. A smaller `TIE_EPS` would ship fewer, more conservative
  entries; a larger one, more.
* **Best rocKE configuration, not a competitive win.** The entry ratios are
  rocKE against rocKE (finalist against the shipped plan). On gfx1201
  PyTorch's backward stays faster than the entries on every cell; on gfx1151
  the entries close the gap on a minority of keys (§2). The entries improve
  rocKE's own best effort; they do not make it the faster backward overall.
* **Baseline moves with develop.** The ratios are against the default plan
  at the measured commit; a later change to the general defaults changes the
  baseline, not the entries' knobs.
* **`d=32` is unswept.** No table on any arch (RDNA or CDNA) has a `d=32`
  general entry; the census has no cell there.

## 5. Provenance

`rdna_entries_manifest.json` records, for every shipped entry, its qualifying
verdict (the finalist id, cells checked, `correct`) and, for every finalist
it names, the finalist's sweep record (`Candidate.record()`).
`tests/dispatch/attention_bwd/test_tuned_tables.py` checks that the manifest
and `GENERAL_TUNED_TABLE` name the same keys, that each record hashes to its
finalist id, and that each shipped entry's knobs are its finalist's: an entry
cannot be added, removed or edited without a matching manifest change.
