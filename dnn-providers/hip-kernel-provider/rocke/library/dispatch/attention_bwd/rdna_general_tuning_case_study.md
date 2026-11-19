# RDNA general SDPA backward tuned-table selection — case study

**Scope.** gfx1151, gfx1201. General family only (the dense family is not
built on RDNA). 27 tuning keys per arch, all at fill bucket `over` (the grid
of every entry's representative requests is at least twice the device's
`num_cus` — see the caveat in §3).

**Result.** On every shipped key, the entry's finalist beats rocKE's shipped
plan: gfx1201 **0.24×–0.28×** of shipped at `d=64` (10 keys), **0.50×–0.53×**
at `d=128` (17 keys); gfx1151 **0.77×–0.87×** at `d=64`, **0.56×–0.67×** at
`d=128`. PyTorch's own backward is faster than rocKE's shipped plan on every
measured cell on both arches, so the CDNA rule (an entry must also beat the
competitor) admits no RDNA entry; see §1 for the rule this document measures
against instead.

Regeneration: `rdna_entry_rule.py` (checked, unit-tested selection logic);
the judge documents and constants that fed it live outside the source tree
(`tuned_tables.py` "RDNA entries" / "Regeneration").

---

## 1. The rule measured here

The agreed RDNA rule (`tuned_tables.py` "RDNA entries"): a sweep finalist is
the entry of tuning key K when, on every dense census cell of K, it is (1)
correct, (2) A/A-clean, and (3) not slower than rocKE's own shipped plan
(paired verdict `pass`, or `ambiguous` within a device-calibrated `tie_eps`);
among finalists passing every cell of K, the lowest geometric-mean time ratio
against shipped wins, only when that geomean is below 1.

`tie_eps` is calibrated per device from the sweep's own A/A spread (not the
repo's private shipped constants, which do not cover these devices); its
value is recorded with the sweep outside the source tree, not here, per the
project's performance-evidence policy (`platform/AGENTS.md` "Public
software-performance evidence": relative results only, in a qualifying
document such as this one).

## 2. Config-labeled coverage

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

## 3. Caveats

* **One route, not two.** RDNA has no dense family, so there is no second
  route to gate against the oracle here; the general route's own numeric
  correctness gate (an independent fp32 reference, declared dtype and
  tolerance) is the bench's pruning-set gate, applied before any cell is
  timed (criterion 1 of the rule in §1).
* **Calibrated, not shipped, `tie_eps`.** The repository ships no RDNA gate
  constants; the tolerance used to call a cell's `ambiguous` verdict
  "not slower" was calibrated per device from this sweep's own measurements,
  not taken from an approved reference. A smaller `tie_eps` would ship fewer,
  more conservative entries; a larger one, more.
* **Best rocKE configuration, not a competitive win.** These ratios are
  rocKE-vs-rocKE (finalist vs the shipped plan). PyTorch's own backward beats
  rocKE's shipped plan by a wide margin on every measured cell on both arches
  (this is *why* the CDNA confirm rule does not apply here, per §1) — these
  entries do not close that gap, they only improve rocKE's own best effort.
* **`d=32` is unswept.** No table on any arch (RDNA or CDNA) has a `d=32`
  general entry; the census has no cell there.

## 4. Provenance

`rdna_entries_manifest.json` records the qualifying verdict of every shipped
entry (the finalist id, cells checked, `correct`) and is checked against
`GENERAL_TUNED_TABLE` by `test_every_rdna_entry_has_a_manifest_verdict`
(`tests.dispatch.attention_bwd.test_tuned_tables`): an entry cannot be added
to or removed from the shipped table without a matching manifest record, and
vice versa.
