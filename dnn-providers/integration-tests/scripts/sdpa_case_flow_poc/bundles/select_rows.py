#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Select the SDPA forward case rows that become integration-test bundles.

Reads levels.csv and never edits it. Writes:
  selected.csv    the selected rows, byte-identical lines of the input (same columns)
  exclusions.csv  case_key, level, reasons (';'-joined) for every row that is not selected
  summary.json    counts per reason and per tier

Exclusion reasons:
  content reasons     a bundle cannot express the row (an output written over an input,
                      inputs that share memory in a way the case builder keeps as separate
                      buffers, arch-dependent dtypes), or its outputs have no element to
                      compare (a ragged Q that gives no query token)
  dropout_without_rng_dump, reference_declines_descale_non_fp8
                      the extra exclusions. The second one is off by default (--descale-non-fp8-filter
                      turns it on). It takes the clauses of the model's answer-key rule
                      AK-DESCALE-WITH-FP8 that forbid a present Q, K or V descale. Levels that ignore
                      the answer-key rules admit such rows, and no reference can check them.
  not_selected:tier, not_selected:min_sweep_size         the selection parameters
  frontend_rejects, realise_error, content_not_expressible, export_error, export_missing
                      outcomes of a runner export, when --export-outcomes is given

With the default parameters, and an input whose SHA-256 equals the one in
test-plans/sdpa-fwd/levels.csv.sha256, the script asserts the counts recorded below
for that CSV.
"""
import argparse
import csv
import functools
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

IT_DIR = Path(__file__).resolve().parents[3]
PLAN_DIR = IT_DIR / "test-plans" / "sdpa-fwd"
MIGRATION_DIR = IT_DIR / "migration-scripts"
BUNDLE_TREE = IT_DIR / "integration-test-bundles"
EXPECTED_SHA256_FILE = PLAN_DIR / "levels.csv.sha256"
LEVELS = ["quick", "standard", "comprehensive", "full"]
BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

EXTRA_REASONS = (
    "dropout_without_rng_dump",
    "reference_declines_descale_non_fp8",
)
DESCALE_RULE = "AK-DESCALE-WITH-FP8"
DESCALE_AXES = ("descale_q", "descale_k", "descale_v")

# Counts for the CSV named by levels.csv.sha256, with the default parameters.
EXPECTED = {
    "rows": 35255,
    "content_reasons": {
        "tensor_overlap": 310,
        "ragged_query_without_tokens": 132,
        "arch_dtype": 103,
    },
    "extra_of_expressible": {
        "dropout_without_rng_dump": 18,
    },
    "per_tier": {
        # rows, bundle rows, strict sweeps, alignment-split sweeps, largest, 1-case, then the summed
        # est_seconds on gfx942 and gfx950 of the expressible rows before the extra exclusions
        "quick": (1461, 1445, 120, 184, 400, 56, 50.9, 39.4),
        "standard": (8631, 8560, 314, 534, 2142, 33, 457.8, 341.3),
        "comprehensive": (4070, 4033, 264, 423, 969, 116, 1487.4, 1059.5),
        "full": (21093, 20814, 702, 1062, 3755, 83, 7294.8, 5205.0),
    },
    "bundle_rows": 34852,
    "sweeps": 1400,
    "sweeps_alignment_split": 2203,
}


@functools.lru_cache(maxsize=None)
def descale_clauses(model_path):
    """Clauses of the descale answer-key rule that forbid a present Q, K or V descale."""
    model = yaml.safe_load(Path(model_path).read_text())
    rules = [
        r for r in model["level_labels"]["answer_key_needs"] if r["id"] == DESCALE_RULE
    ]
    if len(rules) != 1:
        sys.exit(
            f"select_rows: {model_path} has {len(rules)} rules named {DESCALE_RULE}"
        )
    clauses = [
        c
        for c in rules[0]["forbid_any"]
        if any(ax in c and "absent" not in c[ax] for ax in DESCALE_AXES)
    ]
    if not clauses:
        sys.exit(
            f"select_rows: {DESCALE_RULE} in {model_path} forbids no present descale"
        )
    return tuple(tuple((ax, tuple(vals)) for ax, vals in c.items()) for c in clauses)


def descale_on_non_fp8(row, model_path):
    return any(
        all(row[ax] in vals for ax, vals in c) for c in descale_clauses(model_path)
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_sha256(path=EXPECTED_SHA256_FILE):
    """The SHA-256 in a sha256sum-format file, or None when the file does not exist."""
    path = Path(path)
    return path.read_text(encoding="utf-8").split()[0] if path.is_file() else None


class Model:
    """Class tables from class-to-value.json that decide which tensors a row has."""

    def __init__(self, class_values_path):
        values = json.loads(Path(class_values_path).read_text())
        self.ragged = values["ragged_tensors"]
        self.ragged_stats = values["ragged_stats"]
        self.paged = values["paged_sides"]
        self.page_tables = values["page_tables"]
        self.packed_table = values["packed_page_table"]
        self.head_dim = values["head_dim"]

    def ragged_offsets(self, row):
        """The ragged offset tensors of a row's graph, named as the case builder names them."""
        letters = [
            letter
            for letter in self.ragged[row["batch_packing"]]
            if not (letter == "O" and row["o_packing"] == "padded")
            and not (letter == "V" and row["v_packing"] == "padded")
        ]
        if not letters:
            return []
        if (
            "Q" in letters
            and row["stats_output"] == "requested"
            and active(row, "stats_packing")
        ):
            letters += self.ragged_stats[row["stats_packing"]]
        if row["ragged_offset_form"] == "shared_tokens":
            sides = {
                "kv_side" if letter in ("K", "V") else "q_side" for letter in letters
            }
            return ["ragged_offset_" + side for side in sorted(sides)]
        return ["ragged_offset_" + letter.lower() for letter in letters]

    def tensors(self, row, strict=True):
        """Sorted tuple of the tensor roles of a row's graph, with value-form tags when strict."""
        t = ["q", "k", "v", "o"]
        bv = (lambda n: n + "[by_value]") if strict else (lambda n: n)

        t += self.ragged_offsets(row)

        paged = (
            set() if row["paged_kv"] == "absent" else set(self.paged[row["paged_kv"]])
        )
        if row["sequence_lengths"] in ("variable", "variable_no_flag") or paged:
            t += ["seq_len_q", "seq_len_kv"]

        if row["block_mask"] == "present":
            t.append("block_mask")
        if row["bias"] == "present":
            t.append("bias")
        if row["attention_sinks"] == "present":
            t.append("sink")

        src = row["scale_source"]
        if src == "by_value_constant_tensor":
            t.append(bv("attn_scale"))
        elif src == "runtime_parameter":
            t.append("attn_scale[runtime]" if strict else "attn_scale")
        elif src == "device_tensor":
            t.append("attn_scale")

        d = row["dropout"]
        rng = row["dropout_rng_state"]
        if d == "random" and rng == "caller_supplied":
            t += [bv("dropout_seed"), bv("dropout_offset")]
        elif d == "random" and rng == "caller_supplied_device_memory":
            t += ["dropout_seed", "dropout_offset"]
        elif d == "caller_mask":
            t += ["dropout_mask", bv("dropout_scale")]
        if d != "absent" and row["dropout_pattern_output"] == "requested":
            t.append("rng_dump")

        if row["stats_output"] == "requested":
            t.append("stats")
        ms = row["max_sumexp_output"]
        if ms in ("requested", "max_only"):
            t.append("logit_max")
        if ms in ("requested", "sum_exp_only"):
            t.append("score_sum_exp")
        if row["amax_s"] == "requested":
            t.append("amax_s")
        if row["amax_o"] == "requested":
            t.append("amax_o")

        for name in ("descale_q", "descale_k", "descale_v"):
            g = row[name]
            if g == "present":
                t.append(bv(name))
            elif g in ("per_head", "finer"):
                t.append(name)
        if row["s_quantization"] == "present":
            t += [bv("descale_s"), bv("scale_s")]
        if row["scale_o"] == "present":
            t.append(bv("scale_o"))

        if paged:
            packed = self.packed_table[row["page_table_storage"]]
            shared = (
                active(row, "page_table_sharing")
                and self.page_tables[row["page_table_sharing"]] == 1
            )
            tables = (
                ["page_table_kv"]
                if shared
                else ["page_table_" + s.lower() for s in sorted(paged)]
            )
            for tb in tables:
                t.append(tb)
                if packed:
                    t.append(tb + "_offset")
        return tuple(sorted(t))


def active(row, axis):
    return row[axis] not in ("", "-")


def head_dims(row, model):
    """(head_dim_qk, head_dim_v) of the row: its sizes, else the class values."""
    sizes = dict(
        (key, int(value))
        for key, value in re.findall(r"([A-Za-z]+?)(\d+)", row["sizes"])
    )
    if "Dqk" in sizes and "Dv" in sizes:
        return sizes["Dqk"], sizes["Dv"]
    return model.head_dim[row["head_dim_qk"]], model.head_dim[row["head_dim_v"]]


def overlap_not_expressible(row, model):
    """True when the row's tensor overlap is not inputs that share memory as one buffer.

    The view is V when head_dim_v <= head_dim_qk, else K; its owner is K, else Q (as in the
    case builder). A view whose dims do not fit inside its owner leaves through the export
    outcomes. The case builder keeps inputs that share memory on a ragged row as separate
    buffers.
    """
    overlap = row["tensor_overlap"]
    if overlap == "separate":
        return False
    if overlap != "inputs_overlap":
        return True
    if active(row, "ragged_offsets"):
        return True
    if row["paged_kv"] != "absent":
        return True
    dqk, dv = head_dims(row, model)
    view, owner = ("v", "k") if dv <= dqk else ("k", "q")
    if row["strides_" + view] != "inherited" or row["dim_order_" + view] != "inherited":
        return True

    def relation(tensor):
        return "same" if tensor == "q" else row[f"dtype_{tensor}_vs_q"]

    return relation(view) != relation(owner) or (
        row["alignment_" + view] != row["alignment_" + owner]
    )


def ragged_query_without_tokens(row, model):
    """True when a ragged Q gives no query token to compare.

    Every sequence length is 0, or the lengths come from the Q offsets and a zero sequence
    stride makes every offset 0.
    """
    if "Q" not in model.ragged[row["batch_packing"]]:
        return False
    if row["per_seq_len_q"] == "all_zero":
        return True
    has_seq_len = (
        row["sequence_lengths"] in ("variable", "variable_no_flag")
        or row["paged_kv"] != "absent"
    )
    return row["strides_q"] == "zero_broadcast" and not has_seq_len


def content_reasons(row, model):
    """Why a bundle cannot express the row; empty when it can."""
    why = []
    if overlap_not_expressible(row, model):
        why.append("tensor_overlap")
    if ragged_query_without_tokens(row, model):
        why.append("ragged_query_without_tokens")
    if (
        row["dtype_q"] in ("fp16", "bf16")
        and "differs" in (row["dtype_k_vs_q"], row["dtype_v_vs_q"])
    ) or (
        row["matmul_input_precision"] == "fp8" and not row["dtype_q"].startswith("fp8")
    ):
        why.append("arch_dtype")
    return why


def extra_reasons(row, descale_filter=False, model_path=PLAN_DIR / "model.yaml"):
    why = []
    if row["dropout"] == "random" and row["dropout_pattern_output"] != "requested":
        why.append("dropout_without_rng_dump")
    if descale_filter and descale_on_non_fp8(row, model_path):
        why.append("reference_declines_descale_non_fp8")
    return why


def _ebytes(dt):
    return BYTES.get(dt, 1 if dt.startswith("fp8") else dt)


def alignment_signature(model, row):
    """Strict signature plus each tensor's effective alignment (sweeps when alignment is not templated)."""
    al = []
    for t, dt in (
        ("q", row["dtype_q"]),
        ("k", row["dtype_q"] if row["dtype_k_vs_q"] == "same" else "?"),
        ("v", row["dtype_q"] if row["dtype_v_vs_q"] == "same" else "?"),
        ("o", row["dtype_o"]),
    ):
        a = row["alignment_" + t]
        al.append(
            16
            if a == "wide_aligned"
            else (1 if a == "sub_element" else ("el", _ebytes(dt)))
        )
    return (model.tensors(row, True), tuple(map(str, al)))


def read_rows(path):
    """(header line, data lines, parsed rows). The CSV must have no quoted fields."""
    raw = Path(path).read_bytes().decode("ascii")
    if '"' in raw:
        sys.exit(f"select_rows: {path} has quoted fields")
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    else:
        sys.exit(f"select_rows: {path} does not end with a newline")
    if any(line.endswith("\r") for line in lines):
        sys.exit(f"select_rows: {path} has CR line endings")
    header = lines[0].split(",")
    rows = []
    for n, line in enumerate(lines[1:], 2):
        fields = line.split(",")
        if len(fields) != len(header):
            sys.exit(
                f"select_rows: line {n} has {len(fields)} fields; the header has {len(header)}"
            )
        rows.append(dict(zip(header, fields)))
    return lines[0], lines[1:], rows


EXPORT_REASON = {
    "frontend-rejected": "frontend_rejects",
    "realise-error": "realise_error",
    "content-not-expressible": "content_not_expressible",
    "export-error": "export_error",
}


def read_export_outcomes(paths):
    """case_key -> outcome over every given runner export JSONL (all files must agree)."""
    out = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("record") != "export":
                continue
            key, outcome = rec["case_key"], rec["outcome"]
            if out.get(key, outcome) != outcome:
                sys.exit(
                    f"select_rows: export outcomes disagree for {key}: {out[key]} vs {outcome}"
                )
            out[key] = outcome
    return out


def classify(
    rows,
    model,
    *,
    bundle_tiers=LEVELS,
    min_sweep_size=1,
    export_outcomes=None,
    descale_filter=False,
    model_path=PLAN_DIR / "model.yaml",
):
    """Reasons per row (list per row, in input order). An empty list means the row is selected."""
    reasons = []
    for r in rows:
        why = content_reasons(r, model) + extra_reasons(r, descale_filter, model_path)
        if export_outcomes is not None and not why:
            outcome = export_outcomes.get(r["case_key"])
            if outcome is None:
                why.append("export_missing")
            elif outcome != "exported":
                why.append(
                    EXPORT_REASON.get(outcome, "export_" + outcome.replace("-", "_"))
                )
        if r["level"] not in bundle_tiers:
            why.append("not_selected:tier")
        reasons.append(why)

    size = Counter(
        (r["level"], model.tensors(r, True)) for r, w in zip(rows, reasons) if not w
    )
    for r, w in zip(rows, reasons):
        if not w and size[(r["level"], model.tensors(r, True))] < min_sweep_size:
            w.append("not_selected:min_sweep_size")
    return reasons


def summarise(rows, reasons, model):
    content = Counter()
    extra_of_expressible = Counter()
    for r, w in zip(rows, reasons):
        c = content_reasons(r, model)
        content.update(c)
        if not c:
            extra_of_expressible.update(x for x in w if x in EXTRA_REASONS)
    per_tier = {}
    for lv in LEVELS:
        sel = [r for r, w in zip(rows, reasons) if not w and r["level"] == lv]
        expr = [r for r in rows if r["level"] == lv and not content_reasons(r, model)]
        strict = Counter(model.tensors(r, True) for r in sel)
        aligned = Counter(alignment_signature(model, r) for r in sel)
        per_tier[lv] = {
            "rows": sum(1 for r in rows if r["level"] == lv),
            "bundle_rows": len(sel),
            "sweeps": len(strict),
            "sweeps_alignment_split": len(aligned),
            "largest_sweep": max(strict.values(), default=0),
            "one_case_sweeps": sum(1 for v in strict.values() if v == 1),
            "est_seconds_gfx942": round(
                sum(float(r["est_seconds"] or 0) for r in sel), 1
            ),
            "est_seconds_gfx950": round(
                sum(float(r["est_seconds_gfx950"] or 0) for r in sel), 1
            ),
            "est_seconds_expressible_gfx942": round(
                sum(float(r["est_seconds"] or 0) for r in expr), 1
            ),
            "est_seconds_expressible_gfx950": round(
                sum(float(r["est_seconds_gfx950"] or 0) for r in expr), 1
            ),
        }
    return {
        "rows": len(rows),
        "content_reasons": dict(content),
        "extra_of_expressible": dict(extra_of_expressible),
        "all_reasons": dict(Counter(x for w in reasons for x in w)),
        "per_tier": per_tier,
        "bundle_rows": sum(t["bundle_rows"] for t in per_tier.values()),
        "sweeps": sum(t["sweeps"] for t in per_tier.values()),
        "sweeps_alignment_split": sum(
            t["sweeps_alignment_split"] for t in per_tier.values()
        ),
    }


def check_expected(summary):
    failures = []

    def eq(what, got, want):
        if got != want:
            failures.append(f"{what}: got {got}, expected {want}")

    eq("rows", summary["rows"], EXPECTED["rows"])
    eq("content reasons", summary["content_reasons"], EXPECTED["content_reasons"])
    eq(
        "extra exclusions of expressible rows",
        summary["extra_of_expressible"],
        EXPECTED["extra_of_expressible"],
    )
    for lv, want in EXPECTED["per_tier"].items():
        t = summary["per_tier"][lv]
        got = (
            t["rows"],
            t["bundle_rows"],
            t["sweeps"],
            t["sweeps_alignment_split"],
            t["largest_sweep"],
            t["one_case_sweeps"],
            t["est_seconds_expressible_gfx942"],
            t["est_seconds_expressible_gfx950"],
        )
        eq(f"tier {lv}", got, want)
    eq("bundle rows", summary["bundle_rows"], EXPECTED["bundle_rows"])
    eq("sweeps", summary["sweeps"], EXPECTED["sweeps"])
    eq(
        "sweeps when alignment splits",
        summary["sweeps_alignment_split"],
        EXPECTED["sweeps_alignment_split"],
    )
    return failures


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="levels.csv from sdpa_case_flow_poc.levels",
    )
    ap.add_argument(
        "--class-values",
        type=Path,
        required=True,
        help="class-to-value.json from python3 -m sdpa_case_flow_poc",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=PLAN_DIR / "model.yaml",
        help="model.yaml (read only by --descale-non-fp8-filter)",
    )
    ap.add_argument(
        "--expected-sha256",
        type=Path,
        default=EXPECTED_SHA256_FILE,
        help="sha256sum-format file; the recorded counts are asserted when the input matches it",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-sweep-size", type=int, default=1)
    ap.add_argument("--bundle-tiers", default=",".join(LEVELS))
    ap.add_argument(
        "--descale-non-fp8-filter",
        action="store_true",
        help="exclude rows with a Q/K/V descale on non-fp8 inputs (no reference checks them)",
    )
    ap.add_argument(
        "--export-outcomes",
        type=Path,
        action="append",
        default=[],
        help="runner --export-graphs JSONL; rows not exported are excluded (repeatable)",
    )
    args = ap.parse_args()

    tiers = [t for t in args.bundle_tiers.split(",") if t]
    bad = [t for t in tiers if t not in LEVELS]
    if bad:
        sys.exit(f"select_rows: unknown tier(s) {bad}")
    if args.min_sweep_size < 1:
        sys.exit("select_rows: --min-sweep-size must be at least 1")

    digest = sha256(args.input)
    header, lines, rows = read_rows(args.input)
    model = Model(args.class_values)
    outcomes = (
        read_export_outcomes(args.export_outcomes) if args.export_outcomes else None
    )
    reasons = classify(
        rows,
        model,
        bundle_tiers=tiers,
        min_sweep_size=args.min_sweep_size,
        export_outcomes=outcomes,
        descale_filter=args.descale_non_fp8_filter,
        model_path=args.model,
    )
    summary = summarise(rows, reasons, model)
    summary["input"] = str(args.input)
    summary["input_sha256"] = digest
    summary["model"] = str(args.model)
    summary["model_sha256"] = sha256(args.model)
    summary["parameters"] = {
        "min_sweep_size": args.min_sweep_size,
        "bundle_tiers": tiers,
        "descale_non_fp8_filter": args.descale_non_fp8_filter,
        "export_outcomes": [str(p) for p in args.export_outcomes],
    }

    defaults = (
        args.min_sweep_size == 1
        and tiers == LEVELS
        and not args.export_outcomes
        and not args.descale_non_fp8_filter
    )
    if defaults and digest == expected_sha256(args.expected_sha256):
        failures = check_expected(summary)
        if failures:
            for f in failures:
                print("select_rows: ASSERT FAILED:", f, file=sys.stderr)
            sys.exit(1)
        summary["asserted_recorded_counts"] = True
        print("select_rows: the recorded counts hold for this CSV", file=sys.stderr)
    else:
        summary["asserted_recorded_counts"] = False

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "selected.csv", "w", newline="") as f:
        f.write(header + "\n")
        for line, w in zip(lines, reasons):
            if not w:
                f.write(line + "\n")
    with open(args.out_dir / "exclusions.csv", "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["case_key", "level", "reasons"])
        for r, w in zip(rows, reasons):
            if w:
                wr.writerow([r["case_key"], r["level"], ";".join(w)])
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
        f.write("\n")
    print(
        f"select_rows: {summary['bundle_rows']} selected rows in {summary['sweeps']} sweeps;"
        f" wrote {args.out_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
