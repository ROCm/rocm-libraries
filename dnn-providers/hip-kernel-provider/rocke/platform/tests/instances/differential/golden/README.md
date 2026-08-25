# Cpp L5 golden anchor

`llvm_gfx_all.json` is the **absolute regression anchor** for the C engine's
emitted output. It records, per emit mode, per family, per sampled config, the
canonical reference sha256:

- `"ll"` — sha of the lowered LLVM IR text (the Python reference, which equals
  the C engine output for every GREEN config).
- `"ir_canonical"` — sha of the *canonicalized* `ck.dsl.ir/v1` serialization
  (stable across incidental SSA-id renumbering), again from the Python
  reference.

Recorded from a verified-good state. The counts below are **historical** — what
the blessing run saw, not a description of the current tree. Families have been
added since, and `RANGE_DRIFT` now fails the gate rather than being waived, so
the two `RANGE_DRIFT` families here blessed fewer configs than a clean run
would. For what the tree does today, run the gate; do not read these as a
target. Re-blessing from a green run will replace them.
- `--mode ll`: 62 GREEN / 2 RANGE_DRIFT / 0 DRIFT
- `--mode ir --canonical`: 61 GREEN / 1 DRIFT / 2 RANGE_DRIFT
  (the single DRIFT, `gfx950_attention_tiled_2d_fastkv_regp`, is a known
  pre-existing C-vs-Python divergence; the golden still anchors its Python
  reference sha, which is well-defined.)

## Check (CI / before every commit)

    # build a non-sanitized engine first, e.g. /tmp/rocke_harden/librocke_core.a
    python3 ../run_diff.py --mode ll              --archive <librocke_core.a> --check-golden
    python3 ../run_diff.py --mode ir --canonical  --archive <librocke_core.a> --check-golden

Both must print `GOLDEN OK` (exit 0). A non-zero exit with `UNBLESSED CHANGE`
or `MISSING` means an emitted `.ll` / canonical-ir changed for a blessed config
— i.e. a behavioral change.

`GOLDEN OK` answers *"does output match the blessed shas"*, and nothing else.
It is not the parity verdict: the golden anchors the **Python** reference shas,
so a C++-side divergence leaves them unchanged. A run can print its
`=== GATE FAILURES ===` block and still exit 0 here. Read both.

## Re-bless (only with a reviewed, intended output change)

If a diff is expected (you deliberately changed codegen), re-record:

    python3 ../run_diff.py --mode ll              --archive <librocke_core.a> --record-golden
    python3 ../run_diff.py --mode ir --canonical  --archive <librocke_core.a> --record-golden

`--record-golden` overwrites the matching top-level key in this file. It records
the Python reference shas and will note (but not block on) any C-vs-Python
DRIFT. Commit the regenerated `llvm_gfx_all.json` alongside the code change that
justifies it, and call out the re-bless in review.
