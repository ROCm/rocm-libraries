#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Deterministic topology (sweep folder) names for SDPA forward case rows.

A row's strict signature is the set of tensor roles of its graph plus the value form of
each by-value or runtime tensor (select_rows.Model.tensors). The name lists the optional
roles (every role except Q, K, V and O) in a fixed order, one token per role, joined by
'_'. A row with no optional role is 'Base'. A name longer than 64 characters keeps its
leading tokens and ends with '_' and 6 hex digits of the sha256 of the full name.

A superset template holds the tensors of several signatures: a row whose signature is a
subset of the template's runs as a sweep case that names its missing tensors absent. Per
tier, superset_map sorts the signatures largest first, ties by the sorted roles, and puts
each one into the smallest earlier superset that can hold it, ties again by the sorted
roles; a signature that fits none starts a superset. A superset can hold a signature when
the signature's roles are a subset of its roles and the two have the same scale form, the
same dropout input form, the same page tables and the same ragged offsets. Every superset
is the signature of a row, and its name is the topology name of that signature.

The command line computes the name of every row of the CSV, asserts that signatures and
names map one-to-one, and writes topologies.csv. With --grouping-summary it also writes,
per tier, the sweep counts with one template per signature and with superset templates.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_rows  # noqa: E402

MAX_LEN = 64
ALWAYS = ("q", "k", "v", "o")

# Fixed order of the optional roles and the token of each.
ROLE_TOKENS = (
    ("ragged_offset_q", "RgQ"),
    ("ragged_offset_k", "RgK"),
    ("ragged_offset_v", "RgV"),
    ("ragged_offset_o", "RgO"),
    ("seq_len_q", "SlQ"),
    ("seq_len_kv", "SlKv"),
    ("block_mask", "Bm"),
    ("bias", "Bias"),
    ("sink", "Sink"),
    ("attn_scale[by_value]", "ScBv"),
    ("attn_scale[runtime]", "ScRt"),
    ("attn_scale", "ScDev"),
    ("dropout_seed[by_value]", "DSeedBv"),
    ("dropout_offset[by_value]", "DOffBv"),
    ("dropout_seed", "DSeed"),
    ("dropout_offset", "DOff"),
    ("dropout_mask", "DMask"),
    ("dropout_scale[by_value]", "DScBv"),
    ("rng_dump", "Rng"),
    ("stats", "Stats"),
    ("logit_max", "Max"),
    ("score_sum_exp", "SumExp"),
    ("amax_s", "AmaxS"),
    ("amax_o", "AmaxO"),
    ("descale_q[by_value]", "DqBv"),
    ("descale_q", "Dq"),
    ("descale_k[by_value]", "DkBv"),
    ("descale_k", "Dk"),
    ("descale_v[by_value]", "DvBv"),
    ("descale_v", "Dv"),
    ("descale_s[by_value]", "DsBv"),
    ("scale_s[by_value]", "SsBv"),
    ("scale_o[by_value]", "SoBv"),
    ("page_table_k", "PtK"),
    ("page_table_k_offset", "PtKOff"),
    ("page_table_v", "PtV"),
    ("page_table_v_offset", "PtVOff"),
    ("page_table_kv", "PtKv"),
    ("page_table_kv_offset", "PtKvOff"),
)
_ORDER = {role: i for i, (role, _) in enumerate(ROLE_TOKENS)}
_TOKEN = dict(ROLE_TOKENS)


def topology_name(signature):
    """The topology name of a strict signature (a tuple of tensor roles)."""
    for role in ALWAYS:
        if role not in signature:
            raise ValueError(f"signature {signature} has no {role}")
    optional = [r for r in signature if r not in ALWAYS]
    unknown = [r for r in optional if r not in _ORDER]
    if unknown:
        raise ValueError(f"no topology token for role(s) {unknown}")
    tokens = [_TOKEN[r] for r in sorted(optional, key=_ORDER.__getitem__)]
    full = "_".join(tokens) or "Base"
    if len(full) <= MAX_LEN:
        return full
    suffix = "_" + hashlib.sha256(full.encode()).hexdigest()[:6]
    head = ""
    for tok in tokens:
        cand = f"{head}_{tok}" if head else tok
        if len(cand) + len(suffix) > MAX_LEN:
            break
        head = cand
    return head + suffix


def row_topology(model, row):
    return topology_name(model.tensors(row, True))


SCALE_ROLES = ("attn_scale[by_value]", "attn_scale[runtime]", "attn_scale")
DROPOUT_INPUT_ROLES = (
    "dropout_seed[by_value]",
    "dropout_offset[by_value]",
    "dropout_seed",
    "dropout_offset",
    "dropout_mask",
    "dropout_scale[by_value]",
)
DESCALE_ROLES = ("descale_q", "descale_k", "descale_v")


def exclusive_form(signature):
    """The roles that a signature and its superset template must have alike.

    These are the scale form, the dropout input form, the page tables and the ragged
    offsets: a template with one of these never runs a case with another.
    """
    s = set(signature)
    return (
        tuple(r for r in SCALE_ROLES if r in s),
        tuple(r for r in DROPOUT_INPUT_ROLES if r in s),
        tuple(sorted(r for r in s if r.startswith("page_table_"))),
        tuple(sorted(r for r in s if r.startswith("ragged_offset_"))),
    )


def _descale_form(signature, role):
    return {r for r in signature if r.split("[", 1)[0] == role}


def can_hold(template, signature):
    """True when a superset template with these roles can run a row with this signature."""
    if not set(signature) <= set(template):
        return False
    if exclusive_form(signature) != exclusive_form(template):
        return False
    for role in DESCALE_ROLES:
        mine, theirs = _descale_form(signature, role), _descale_form(template, role)
        if mine and theirs and mine != theirs:
            return False
    return True


def _largest_first(signature):
    return (-len(signature), signature)


def superset_map(signatures):
    """{signature: superset signature} over one tier's signatures; independent of their order."""
    supersets = []
    out = {}
    for sig in sorted(set(signatures), key=_largest_first):
        homes = [t for t in supersets if can_hold(t, sig)]
        if homes:
            out[sig] = min(homes, key=lambda t: (len(t), t))
        else:
            supersets.append(sig)
            out[sig] = sig
    return out


def check_superset_map(mapping):
    """{superset signature: name}; exits unless every signature has one superset that holds it."""
    names = {}
    for sig, sup in mapping.items():
        if mapping.get(sup) != sup:
            sys.exit(f"topology: superset {sup} of {sig} is not its own superset")
        if not can_hold(sup, sig):
            sys.exit(f"topology: superset {sup} cannot hold {sig}")
        name = names.setdefault(sup, topology_name(sup))
        if len(name) > MAX_LEN:
            sys.exit(f"topology: name {name} is longer than {MAX_LEN}")
    if len(set(names.values())) != len(names):
        sys.exit("topology: two superset templates share a name")
    return names


def superset_topologies(placed):
    """{(level, signature): superset topology name} for (level, signature) pairs, one
    superset map per tier."""
    by_level = defaultdict(set)
    for level, sig in placed:
        by_level[level].add(sig)
    out = {}
    for level, sigs in by_level.items():
        mapping = superset_map(sigs)
        names = check_superset_map(mapping)
        for sig, sup in mapping.items():
            out[(level, sig)] = names[sup]
    return out


def grouping_summary(placed):
    """Per tier of the (level, signature) pairs, one per row: rows, sweeps and one-case
    sweeps with one template per signature and with superset templates, and the largest
    number of absent tensors in one case."""
    supersets = superset_topologies(placed)
    out = {}
    for level in select_rows.LEVELS:
        sigs = [sig for lv, sig in placed if lv == level]
        if not sigs:
            continue
        exact = Counter(sigs)
        grouped = Counter(supersets[(level, s)] for s in sigs)
        size = {supersets[(level, s)]: 0 for s in sigs}
        for s in set(sigs):
            size[supersets[(level, s)]] = max(size[supersets[(level, s)]], len(s))
        out[level] = {
            "rows": len(sigs),
            "sweeps_exact": len(exact),
            "one_case_sweeps_exact": sum(1 for n in exact.values() if n == 1),
            "sweeps_superset": len(grouped),
            "one_case_sweeps_superset": sum(1 for n in grouped.values() if n == 1),
            "cases_with_absent_tensors": sum(
                1 for s in sigs if size[supersets[(level, s)]] != len(s)
            ),
            "max_absent_tensors": max(
                size[supersets[(level, s)]] - len(s) for s in sigs
            ),
        }
    out["total"] = {
        k: sum(t[k] for t in out.values())
        for k in (
            "rows",
            "sweeps_exact",
            "one_case_sweeps_exact",
            "sweeps_superset",
            "one_case_sweeps_superset",
            "cases_with_absent_tensors",
        )
    }
    return out


def check_one_to_one(model, rows):
    """{signature: name} over the rows; exits when two signatures share a name."""
    by_sig = {}
    by_name = {}
    for r in rows:
        sig = model.tensors(r, True)
        name = by_sig.get(sig)
        if name is None:
            name = topology_name(sig)
            by_sig[sig] = name
        other = by_name.setdefault(name, sig)
        if other != sig:
            sys.exit(f"topology: name {name} is shared by {other} and {sig}")
        if len(name) > MAX_LEN:
            sys.exit(f"topology: name {name} is longer than {MAX_LEN}")
    return by_sig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, required=True, help="levels.csv")
    ap.add_argument(
        "--class-values", type=Path, required=True, help="class-to-value.json"
    )
    ap.add_argument("--out", type=Path, required=True, help="topologies.csv to write")
    ap.add_argument(
        "--grouping-summary",
        type=Path,
        help="JSON file to write the per-tier sweep counts without and with superset templates",
    )
    args = ap.parse_args()

    _, _, rows = select_rows.read_rows(args.input)
    model = select_rows.Model(args.class_values)
    by_sig = check_one_to_one(model, rows)

    counts = defaultdict(Counter)
    for r in rows:
        counts[by_sig[model.tensors(r, True)]][r["level"]] += 1
    shortened = sum(
        1
        for n in by_sig.values()
        if "_" in n
        and len(n.rsplit("_", 1)[-1]) == 6
        and all(c in "0123456789abcdef" for c in n.rsplit("_", 1)[-1])
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["topology", "signature"] + select_rows.LEVELS)
        for sig, name in sorted(by_sig.items(), key=lambda kv: kv[1]):
            w.writerow(
                [name, " ".join(sig)] + [counts[name][lv] for lv in select_rows.LEVELS]
            )
    print(
        f"topology: {len(rows)} rows, {len(by_sig)} signatures, {len(by_sig)} names, one-to-one;"
        f" {shortened} shortened names; wrote {args.out}",
        file=sys.stderr,
    )
    if args.grouping_summary:
        summary = grouping_summary([(r["level"], model.tensors(r, True)) for r in rows])
        args.grouping_summary.parent.mkdir(parents=True, exist_ok=True)
        args.grouping_summary.write_text(json.dumps(summary, indent=1) + "\n")
        t = summary["total"]
        print(
            f"topology: sweeps {t['sweeps_exact']} with one template per signature,"
            f" {t['sweeps_superset']} with superset templates; one-case sweeps"
            f" {t['one_case_sweeps_exact']} and {t['one_case_sweeps_superset']};"
            f" wrote {args.grouping_summary}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
