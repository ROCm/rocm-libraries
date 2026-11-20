# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reference plans of the alternate main kernels of ``hipkernel:AttentionGeneralBwd``.

The main slot admits every main kernel of the class a graph pins whose own plan can
stand in for the engine's pick (same prep launch and workspace layout; see
``resolveCandidate`` in AttentionGeneralBwdGeometry.hpp). This tool is the Python
side of that contract. For every accepted entry of the replay corpus it forces rocKE's
planner (``attn_bwd_plan``) to each catalog row of the pick's class, one tuned-table
entry (or the class default) and one head-pack choice at a time, and keeps the row as
an alternate when

* the planner builds it and its main launch is the row's main kernel,
* its prep launch and workspace layout equal the pick's, and
* its main KMD fields equal the row's (the tuning key apart, which the forced policy
  carries as a hook rather than a table entry).

rocKE builds more rows than the engine offers. Three engine rules are narrower than the
planner, and a row the planner builds that breaks one is recorded under ``declined``
with its reason instead of ``alternates``:

* ``thd_pick_only``: a THD (ragged) graph admits only the engine's own pick.
  hipdnn_bench fills ragged offset tables with its input pattern, so THD candidates
  cannot be timed and no trained ranker has seen one.
* ``tuned_class``: a tuned entry serves only graphs of the performance class it was tuned
  on (arch, head size, dtype, mask kind, stage vec); its head mode, length relation,
  length mode and buckets may differ. The planner applies any entry's knobs.
* ``head_pack_group``: head packing needs a query-head group above 1 unless the tuned
  entry pins it. The planner builds a packed kernel for a group of 1.

The C++ replay (TestAttentionGeneralBwdReplay.cpp) then requires the engine's admitted
main candidates to be the pick plus exactly these rows, and every one's C++ plan to
reproduce the main launches recorded here, and never to admit a declined row. The committed corpora
are untouched; the companion file holds one record per entry that has an alternate or a
declined row, joined on ``id``.

Usage::

    general_bwd_alternates.py --arch gfx942 --corpus corpus_gfx942.jsonl --out alternates_gfx942.jsonl [--check]

The output is deterministic. The first line is a ``meta`` record.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

_HKP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _sub in ("rocke/platform/python", "rocke/library"):
    _p = str(_HKP / _sub)
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
os.environ.setdefault("ROCKE_BACKEND", "python")

from dispatch.attention_bwd import hipdnn as H  # noqa: E402
from dispatch.attention_bwd import hipdnn_catalog as C  # noqa: E402
from general_bwd_corpus import canonical  # noqa: E402
from kernels.common.attention_bwd_plan import (  # noqa: E402
    _group,
    attn_bwd_perf_class,
    attn_bwd_plan,
)

GENERATOR = "general_bwd_alternates.py"
# The planner declines with `<CODE>: <detail>` (`INSTANCE_UNAVAILABLE` for a row it cannot
# build under the forced policy); any other ValueError is a defect, not a decline.
_DECLINE = re.compile(r"[A-Z][A-Z0-9_]*: ")
# The class the graph pins; the engine admits only main kernels that agree on all of it.
PINNED = (
    "dtype",
    "head_size",
    "seq_mode",
    "stage_vec",
    "dkv_mode",
    "mask_class",
    "dq_mode",
)
# Positions in the performance class (arch, route, d, dtype, mask kind, head mode, length
# relation, length mode, stage vec) a tuned entry must share with the graph.
CLASS_SERVED_BY_ENTRY = (0, 2, 3, 4, 8)
FIXTURES = _HKP / "src" / "tests" / "fixtures" / "attention_general_bwd"


def _forced_policy(row: C.CatalogRow):
    """The engine policy with the tuned-table lookup replaced by the row's entry."""
    policy = H.engine_policy()
    knobs = dict(policy.tuned_table[row.tuning_key]) if row.tuning_key else {}
    if row.head_pack and "head_pack" not in knobs and knobs.get("global_path") == "dma":
        # The planner's own short-query packing moves the DMA ring to the register path.
        knobs["global_path"] = "vgpr"
    knobs["head_pack"] = row.head_pack
    return dataclasses.replace(
        policy, tuned_table={}, tuning=lambda req, arch, route, pclass: dict(knobs)
    )


def _main_launches(record: dict) -> list[dict]:
    return [
        {
            "launch": launch["launch"],
            "kernel": launch["kernel"],
            "grid": launch["grid"],
            "block": launch["block"],
            "values": [k[2] for k in launch["kernargs"]],
        }
        for launch in record["launches"]
        if launch["slot"] == "main"
    ]


def _prep_launch(record: dict) -> dict:
    (prep,) = [launch for launch in record["launches"] if launch["slot"] == "prep"]
    return prep


def _declined_reason(req, arch: str, row: C.CatalogRow, pick_fields: dict):
    """Why the engine does not offer ``row`` though the planner builds it, or None."""
    if pick_fields["seq_mode"] == "thd":
        return "thd_pick_only"
    if row.tuning_key:
        pclass = attn_bwd_perf_class(req, arch)
        if any(row.tuning_key[i] != pclass[i] for i in CLASS_SERVED_BY_ENTRY):
            return "tuned_class"
    pinned = (
        bool(row.tuning_key)
        and "head_pack" in H.engine_policy().tuned_table[row.tuning_key]
    )
    if row.head_pack and not pinned:
        if _group(req, pick_fields["dkv_mode"] == "atomic") <= 1:
            return "head_pack_group"
    return None


def alternates_of(req, arch: str, pick) -> tuple[list[dict], list[dict]]:
    """``(alternates, declined)`` main kernels of the engine's ``pick`` for ``req`` (see
    module doc)."""
    pick_fields = H.main_fields(pick)
    pick_key = (H.tuning_key_id(arch, H.plan_table_key(pick)), pick_fields["head_pack"])
    pick_record = H.plan_record(pick, req, arch)
    pick_prep = _prep_launch(pick_record)
    out, declined = [], []
    for row in C.catalog_rows(arch):
        if any(getattr(row, name) != pick_fields[name] for name in PINNED):
            continue
        if (row.tuning_key_id, row.head_pack) == pick_key:
            continue
        try:
            plan = attn_bwd_plan(req, arch, policy=_forced_policy(row))
        except ValueError as exc:
            if not _DECLINE.match(str(exc)):
                raise
            continue
        fields = H.main_fields(plan)
        if any(fields[name] != pick_fields[name] for name in PINNED):
            continue
        if any(
            fields[name] != row.main_kmd[name]
            for name in fields
            if name != "tuning_key"
        ):
            continue
        record = H.plan_record(plan, req, arch)
        mains = _main_launches(record)
        if not mains or mains[0]["kernel"] != row.kernels["main"]:
            continue
        if _prep_launch(record) != pick_prep:
            continue
        if record["workspace"] != pick_record["workspace"]:
            continue
        reason = _declined_reason(req, arch, row, pick_fields)
        if reason is not None:
            declined.append(
                {
                    "tuning_key": row.tuning_key_id,
                    "head_pack": int(row.head_pack),
                    "reason": reason,
                }
            )
            continue
        out.append(
            {
                "tuning_key": row.tuning_key_id,
                "head_pack": int(row.head_pack),
                "launches": mains,
            }
        )
    out.sort(key=lambda a: (a["tuning_key"], a["head_pack"]))
    declined.sort(key=lambda a: (a["tuning_key"], a["head_pack"]))
    return out, declined


def build(arch: str, corpus: Path) -> list[dict]:
    lines = corpus.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines[1:]]
    records, accepted, alternates, declined = [], 0, 0, 0
    for entry in entries:
        if entry["arch"] != arch:
            raise SystemExit(f"{corpus}: entry {entry['id']} is not a {arch} entry")
        if entry["expect"]["code"] != "OK":
            continue
        accepted += 1
        req = H.from_hipdnn_graph(entry["graph"], arch, num_cus=entry["num_cus"])
        code, detail, plan = H.engine_verdict(req, arch)
        if plan is None:
            raise SystemExit(f"{entry['id']}: the corpus accepts it, rocKE says {code}")
        found, refused = alternates_of(req, arch, plan)
        if found or refused:
            records.append(
                {"id": entry["id"], "alternates": found, "declined": refused}
            )
            alternates += len(found)
            declined += len(refused)
    meta = {
        "arch": arch,
        "generator": GENERATOR,
        "accepted_entries": accepted,
        "entries_with_alternates": len(records),
        "alternates": alternates,
        "declined": declined,
    }
    return [{"meta": meta}] + records


def render(records: list[dict]) -> str:
    return "".join(canonical(r) + "\n" for r in records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", required=True, choices=H.ENGINE_ARCHS)
    ap.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="the replay corpus (default: the committed corpus_<arch>.jsonl)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true", help="fail when the file differs")
    args = ap.parse_args(argv)
    corpus = args.corpus or FIXTURES / f"corpus_{args.arch}.jsonl"
    out = args.out or FIXTURES / f"alternates_{args.arch}.jsonl"
    text = render(build(args.arch, corpus))
    if args.check:
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            print(f"stale alternates {out} (rerun {GENERATOR})", file=sys.stderr)
            return 1
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
