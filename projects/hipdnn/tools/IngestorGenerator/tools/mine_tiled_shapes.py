#!/usr/bin/env python3
"""Mine a PAGED/VARLEN shape corpus for the gfx950 tiled attention integration.

A sibling of `mine_shapes.py`, not a patch to it. The skill is explicit that the
shape corpus is op-shaped and that another op family needs its own miner -- "that is
a per-op file, not a defect, because a corpus format IS op-specific". The tiled path
is that case, three times over:

1. **`mine_shapes.py` emits a DENSE vocabulary.** Its three producers each hard-code
   `{batch, nhead_q, nhead_k, seqlen_q, seqlen_k, hdim_q, hdim_v, dtype, mask_type}`.
   Neither `block_size` nor `num_seqs` appears anywhere -- and `block_size` is absent
   even from `_shape_key`'s 9-tuple, so two paged shapes differing ONLY in block size
   would silently collide and one would be dropped.
2. **The dense corpus cannot reach the cohort that matters.** The shipped
   `gfx950_attention_dense.corpus-shapes.json` has `hdim_q in {128, 64}` and **no 256
   at all**, so it cannot exercise the D256 cohort decision D5 is about. This corpus
   carries **17** D256 scenarios.
3. **There is no plugin seam to extend.** `mine_shapes.py` is monolithic -- no
   registry, no `--op` flag, no abstract producer. Adding paged fields to it would
   mean editing three unrelated dense producers and widening a dedup key that four
   other integrations depend on.

**The source is the kernel team's own parity corpus**: the `Scenario` list driving
`ref_paged_attn` in `builders/gfx950/attention/prefill/parity_unified_attention.py`.
That choice is deliberate and it is the point:

- These shapes carry **real paged geometry** -- `block_size` and a per-sequence
  `seq_lens` list of `(query_len, kv_len)` pairs -- which no dense source has.
- They are **genuinely varlen**. `AttentionRequest._problem()` hard-codes
  `total_q = batch * seqlen_q` and `num_seqs = batch`, so every problem it builds is
  uniform. Here `total_q = sum(query_len)` over an actually-ragged list.
- They are **the shapes stage 8 verifies against** (decision D1). Sizing the variant
  set from the same corpus the numeric oracle uses means the shipped kernels and the
  correctness evidence describe the same thing, rather than two populations that
  happen to overlap.

Output is the same JSON list of request-field mappings the generator tools consume,
so `dispatch_parity.py`, `knob_sweep.py` and `variant_reachability.py` read it
unchanged. The field names match `TiledAttentionRequest` (and therefore
`UnifiedAttentionProblem`), not `AttentionRequest`.

**Provenance is carried on every shape and every reported result must be split by
it.** A geomean over a mixed corpus reports one population's win as everyone's.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

#: Scenario groups, in the order the parity module defines them. Enumerated from the
#: module at runtime rather than hard-coded counts: the module's own README claims
#: 13/21/26 and `creative_scenarios`'s docstring says "the default 11 scenarios",
#: and BOTH are stale against source (real: 23/26/30 = 79). A count written down here
#: would be the third stale copy. The GROUP NAMES are contract-ish; the counts are not.
_SCENARIO_GROUPS = ("default_scenarios", "fmha_scenarios", "creative_scenarios")

#: torch dtype -> the spelling `UnifiedAttentionProblem.dtype` expects.
_DTYPE_SPELLINGS = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "bfloat16": "bf16",
    "float16": "fp16",
}


def _load_parity_module(path: Path):
    """Load the parity module by FILE PATH, not by import name.

    It lives under `builders/gfx950/attention/prefill/`, which is only importable as
    a package once `rocke/library` is on `sys.path` AND the intervening directories
    have `__init__.py`. The existing rocKE tests load it exactly this way
    (`test_attn_bf16_d128_ring.py:158-169`), so this follows the tree's own pattern
    rather than inventing a second one.
    """
    if not path.is_file():
        raise SystemExit(
            f"FAIL: no parity module at {path}\n"
            f"  This is a coordinate, and coordinates go stale. Find the real one "
            f"with:  grep -rn 'def ref_paged_attn' <provider>/rocke/library"
        )
    spec = importlib.util.spec_from_file_location("_tiled_parity_corpus", path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so any internal self-import resolves to this object.
    sys.modules["_tiled_parity_corpus"] = module
    spec.loader.exec_module(module)
    return module


def _normalise_dtype(value) -> str:
    """`torch.bfloat16` -> `"bf16"`, REFUSING anything unrecognised.

    The op-shaped-miner contract is explicit: an unrecognised categorical must be
    REFUSED, never defaulted. Defaulting one is how a windowed graph got served as
    plain causal -- a wrong answer, not a decline. A new dtype landing in the corpus
    must fail loudly here and be added deliberately.
    """
    text = str(value)
    if text in _DTYPE_SPELLINGS:
        return _DTYPE_SPELLINGS[text]
    raise SystemExit(
        f"FAIL: unrecognised dtype {text!r} in the parity corpus.\n"
        f"  Refusing rather than defaulting: a mis-spelled dtype resolves to a "
        f"DIFFERENT binary and the mistake surfaces as wrong numbers, not an error.\n"
        f"  Known: {sorted(_DTYPE_SPELLINGS)}. Add the new spelling deliberately."
    )


def shape_from_scenario(scenario, group: str) -> dict:
    """One `Scenario` -> one request-field mapping.

    `seq_lens` is a list of `(query_len, kv_len)` pairs -- the whole reason this
    corpus exists. `total_q` is their genuine sum, not `batch * seqlen_q`.
    """
    seq_lens = list(scenario.seq_lens)
    query_lens = [int(q) for q, _ in seq_lens]
    kv_lens = [int(k) for _, k in seq_lens]

    total_q = sum(query_lens)
    num_seqs = len(seq_lens)
    max_seqlen_q = max(query_lens)
    max_seqlen_k = max(kv_lens)

    # `sliding_window` is Optional[int] on the Scenario and a plain int on the
    # problem, where 0 means "unbounded". None and 0 mean the same thing here; the
    # kernel's own `sliding_window > 0` test is what both feed.
    sliding_window = int(scenario.sliding_window or 0)

    return {
        # --- the 9 UnifiedAttentionProblem requires ---
        "total_q": total_q,
        "num_seqs": num_seqs,
        "num_query_heads": int(scenario.num_query_heads),
        "num_kv_heads": int(scenario.num_kv_heads),
        "head_size": int(scenario.head_size),
        # The paged geometry no dense source carries. graph_contract.md §5 G2:
        # on the hipDNN side this is K.dim[2] -- the K/V tensor IS the container.
        "block_size": int(scenario.block_size),
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "dtype": _normalise_dtype(scenario.dtype),
        # --- recorded request attributes, NOT tuning choices ---
        # Carried even for features this integration declines, so the shape stays
        # visible to the step-9 reconciler. Filtering here would hide it entirely,
        # and a decline nobody can see is indistinguishable from a decline nobody
        # made.
        "sliding_window": sliding_window,
        "softcap": float(scenario.softcap),
        "use_sinks": bool(scenario.use_sinks),
        "use_alibi": bool(scenario.use_alibi),
        "use_qq_bias": bool(scenario.use_qq_bias),
        # Physical pool size. 0 would mean "unknown"; the corpus states it.
        "num_kv_blocks": int(scenario.num_blocks),
        "_provenance": {
            "source": "rocke_parity",
            "suite": group,
            # A STABLE name, because an index shifts the moment the corpus is
            # re-mined and a declines file keyed on it then marks a DIFFERENT shape
            # -- silently, since a key matching nothing is a hard error but a key
            # matching the WRONG row is not.
            "graph": f"rocke_parity__{group}__{scenario.name}",
            "scenario": scenario.name,
            # The ragged detail that does not survive into the flat request fields.
            # Kept so a varlen shape can be reconstructed for a bundle at 8a.
            "seq_lens": seq_lens,
            "is_varlen": len(set(query_lens)) > 1 or len(set(kv_lens)) > 1,
        },
    }


def _shape_key(shape: dict) -> tuple:
    """The dedup identity. **`block_size` and `num_seqs` are IN it**, unlike the dense
    miner's 9-tuple, where their absence would silently merge two paged shapes that
    compile to different binaries."""
    return (
        shape["total_q"],
        shape["num_seqs"],
        shape["num_query_heads"],
        shape["num_kv_heads"],
        shape["head_size"],
        shape["block_size"],
        shape["max_seqlen_q"],
        shape["max_seqlen_k"],
        shape["dtype"],
        shape["sliding_window"],
        shape["softcap"],
        shape["use_sinks"],
        shape["use_alibi"],
        shape["use_qq_bias"],
    )


def deduplicate(shapes: list[dict]) -> tuple[list[dict], int]:
    """One entry per distinct shape, keeping the first provenance and counting the rest.

    A corpus is a set of shapes, not a set of rows -- two suites asking for the same
    shape is one variant to compile. But it is two votes for that shape mattering, so
    the duplicate count is reported rather than discarded.
    """
    seen: dict = {}
    duplicates = 0
    for shape in shapes:
        key = _shape_key(shape)
        if key in seen:
            duplicates += 1
            seen[key]["_provenance"].setdefault("also", []).append(
                shape["_provenance"]["graph"]
            )
            continue
        seen[key] = shape
    return list(seen.values()), duplicates


def _histogram(shapes: list[dict], key) -> dict:
    out: dict = {}
    for shape in shapes:
        out[key(shape)] = out.get(key(shape), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine a paged/varlen shape corpus from rocKE's parity scenarios."
    )
    parser.add_argument(
        "--parity-module",
        required=True,
        help="Path to the gfx950 parity_unified_attention.py holding the Scenario "
        "corpus that drives ref_paged_attn.",
    )
    parser.add_argument("--out", required=True, help="Write the shape corpus here.")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print the axis distributions, which is what sizing decisions are made "
        "against.",
    )
    args = parser.parse_args(argv)

    module = _load_parity_module(Path(args.parity_module))

    shapes: list[dict] = []
    for group in _SCENARIO_GROUPS:
        factory = getattr(module, group, None)
        if factory is None:
            # Zero hits is information, not an empty corpus: it means the module
            # moved out from under this miner. Never silently skipped.
            raise SystemExit(
                f"FAIL: the parity module defines no {group}(). Enumerate what it "
                f"does define with:  grep -nE '^def .*scenarios' <module>"
            )
        found = factory()
        print(f"  {group:20s}: {len(found):3d} scenarios")
        shapes += [shape_from_scenario(s, group) for s in found]

    unique, duplicates = deduplicate(shapes)
    print(f"  {'distinct':20s}: {len(unique):3d}  ({duplicates} merged)")

    if not unique:
        print(
            "\nFAIL: no shapes mined; nothing downstream can use this.", file=sys.stderr
        )
        return 1

    if args.report:
        print("\n  head_size     :", _histogram(unique, lambda s: s["head_size"]))
        print("  block_size    :", _histogram(unique, lambda s: s["block_size"]))
        print("  dtype         :", _histogram(unique, lambda s: s["dtype"]))
        print(
            "  (hq,hkv)      :",
            _histogram(unique, lambda s: (s["num_query_heads"], s["num_kv_heads"])),
        )
        print("  num_seqs      :", _histogram(unique, lambda s: s["num_seqs"]))
        print(
            "  varlen        :",
            _histogram(unique, lambda s: s["_provenance"]["is_varlen"]),
        )
        for feature in (
            "sliding_window",
            "softcap",
            "use_sinks",
            "use_alibi",
            "use_qq_bias",
        ):
            on = sum(1 for s in unique if s[feature])
            print(f"  {feature:14s}: {on} of {len(unique)} shapes have it set")

    Path(args.out).write_text(json.dumps(unique, indent=2))
    print(f"\n  wrote {args.out}")
    print("  Provenance is carried on every shape. Split every reported result by it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
