#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LEDGER.md, ID-MAP.md and gate bookkeeping for a deep review.

Usage: ledger.py <seed|ids|coverage|verdicts|mlines|counts> <evidence dir> <round>
       ledger.py seed <evidence dir> <round> [--lane NAME=<status text>]...
       ledger.py set <evidence dir> <round> <ID> <FIXED@sha|ACCEPTED-RISK|OPEN> [--class C] [--note TEXT]

Each subcommand reads `<evidence>/r<round>/` and `<evidence>/LEDGER.md` and exits 1 with a
one-line reason on failure. Rows are only ever updated or appended, never deleted.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reviewlib as rl  # noqa: E402

COVERAGE_COLS = ["File", "Lines", "Owner lane", "Read depth", "Tests reaching it"]
FINDING_COLS = [
    "ID",
    "Tier",
    "Kind",
    "Status",
    "Verdict",
    "Claim",
    "Where",
    "Evidence",
    "Round opened",
    "Round closed",
    "known-open-item",
]
LANE_COLS = ["Lane", "Status", "Report", "STOP"]
DECISION_COLS = ["Round", "Question", "Decision", "Findings affected"]
QUESTION_COLS = ["ID", "Question", "Options", "Recommendation"]
IDMAP_COLS = ["Lane / provisional", "Final ID", "SUPERSEDED-BY"]
M_OWNERS = {1: "TESTQ", 2: "TESTB", 3: "TESTQ", 4: "AUTH", 5: "AUTH"}
BLOCKING_TIERS = ("BLOCKING", "IMPORTANT")


def ledger_path(evidence):
    return Path(evidence) / "LEDGER.md"


def load_ledger(evidence):
    path = ledger_path(evidence)
    if not path.is_file():
        raise rl.Fail(f"missing {path}; run seed")
    return rl.parse_tables(path.read_text(encoding="utf-8"), "LEDGER.md")


def save_ledger(evidence, tables):
    rl.write(ledger_path(evidence), rl.render_tables(tables))


def collapse(text):
    return " ".join(text.split())


def read_reports(rp, pattern="lanes/*.md"):
    return [(p.stem, p.read_text(encoding="utf-8")) for p in sorted(rp.glob(pattern))]


def cmd_seed(a):
    """Create LEDGER.md (round 1) or append this round's Coverage and Lanes tables
    (round 2+), from owners.tsv, numstat.txt and packets/. One Coverage row per changed
    file (Read depth `pending`, Tests `-`); one Lanes row per packet (`Ran`) and per
    `--lane NAME=<status text>`."""
    rp = rl.round_dir(a.evidence, a.round)
    pin_round = rl.pin_value(rp, "Round")
    if pin_round not in (None, str(a.round)):
        raise rl.Fail(f"PIN.md Round is {pin_round}, not {a.round}")
    for name in ("owners.tsv", "numstat.txt"):
        if not (rp / name).is_file():
            raise rl.Fail(f"missing {rp / name}")
    owners = dict(
        reversed(line.split("\t", 1))
        for line in (rp / "owners.tsv").read_text(encoding="utf-8").splitlines()
        if "\t" in line
    )
    cov = rl.Table(f"Coverage (round {a.round})", COVERAGE_COLS)
    for line in (rp / "numstat.txt").read_text(encoding="utf-8").splitlines():
        added, deleted, path = line.split("\t", 2)
        lines = 0 if added == "-" else int(added) + int(deleted)
        cov.add(
            **{
                "File": path,
                "Lines": lines,
                "Owner lane": owners.get(path, "-"),
                "Read depth": "pending",
                "Tests reaching it": "-",
            }
        )
    lanes = {p.stem: "Ran" for p in sorted(rp.glob("packets/*.md"))}
    for item in a.lane:
        name, sep, status = item.partition("=")
        if not sep:
            raise rl.Fail(f"--lane needs NAME=<status text>: {item}")
        lanes[name] = status
    lane_table = rl.Table(f"Lanes (round {a.round})", LANE_COLS)
    for name, status in lanes.items():
        report = f"r{a.round}/lanes/{name}.md" if status == "Ran" else ""
        lane_table.add(**{"Lane": name, "Status": status, "Report": report})
    if ledger_path(a.evidence).is_file():
        tables = load_ledger(a.evidence)
        if any(t.title == cov.title for t in tables):
            raise rl.Fail(f"round {a.round} is already seeded")
        tables += [cov, lane_table]
    else:
        tables = [
            cov,
            rl.Table("Findings", FINDING_COLS),
            lane_table,
            rl.Table("Decisions", DECISION_COLS),
            rl.Table("Open questions", QUESTION_COLS),
        ]
    save_ledger(a.evidence, tables)
    print(
        f"seeded round {a.round}: {len(cov.rows)} files, {len(lane_table.rows)} lanes"
    )


def cmd_ids(a):
    """Assign final `<LANE>-<n>` IDs (lane digits dropped, so all CORR<k> share CORR-<n>)
    to the provisional IDs in lanes/*.md and verify/VER-*.md, continuing each prefix's
    counter in LEDGER.md; write ID-MAP.md; add a candidate row per new finding. IDs already
    in an existing ID-MAP.md keep their value; a SUPERSEDED-BY entry there sets the row's
    status."""
    rp = rl.round_dir(a.evidence, a.round)
    tables = load_ledger(a.evidence)
    ft = rl.find_table(tables, "Findings")
    map_path = rp / "ID-MAP.md"
    old = (
        rl.parse_tables(map_path.read_text(encoding="utf-8"), "ID-MAP.md")[0].rows
        if map_path.is_file()
        else []
    )
    mapping = {r["Lane / provisional"].rsplit("/", 1)[-1]: r for r in old}
    counters = {}

    def note(final):
        m = re.fullmatch(r"(.+)-(\d+)", final)
        if m:
            counters[m.group(1)] = max(counters.get(m.group(1), 0), int(m.group(2)))

    for r in ft.rows:
        note(r["ID"])
    for r in old:
        note(r["Final ID"])
    reported = {}
    for path in rl.lane_files(rp):
        for f in rl.parse_findings(path.read_text(encoding="utf-8")):
            prov = f["ID"]
            m = re.fullmatch(r"(.+)-p(\d+)", prov)
            if not m:
                raise rl.Fail(f"{path.name}: bad provisional ID {prov}")
            if prov in reported:
                raise rl.Fail(f"provisional ID {prov} reported twice")
            reported[prov] = (path.stem, f, re.sub(r"\d+$", "", m.group(1)))
    for prov, (stem, f, prefix) in sorted(reported.items(), key=lambda kv: kv[1][0]):
        if prov not in mapping:
            counters[prefix] = counters.get(prefix, 0) + 1
            mapping[prov] = {
                "Lane / provisional": f"{stem}/{prov}",
                "Final ID": f"{prefix}-{counters[prefix]}",
                "SUPERSEDED-BY": "",
            }
    rows = {r["ID"]: r for r in ft.rows}
    added = 0
    for prov, m in mapping.items():
        final = m["Final ID"]
        if final not in rows and prov in reported:
            f = reported[prov][1]
            known = re.search(
                r"understated: known-open-item (K\d+)", " ".join(f.values())
            )
            tier = f.get("Tier", "")
            ft.add(
                **{
                    "ID": final,
                    "Tier": f"{tier} -> {tier}",
                    "Kind": f.get("Kind", ""),
                    "Claim": collapse(f.get("Actual", "").split("\n")[0]),
                    "Where": collapse(f.get("Where", "")),
                    "Evidence": collapse(f.get("Proof", "")),
                    "Round opened": a.round,
                    "known-open-item": known.group(1) if known else "",
                }
            )
            rows[final] = ft.rows[-1]
            added += 1
        if m["SUPERSEDED-BY"] and final in rows:
            rows[final]["Status"] = f"SUPERSEDED-BY:{m['SUPERSEDED-BY']}"
    idmap = rl.Table(
        "ID-MAP",
        IDMAP_COLS,
        [{c: m.get(c, "") for c in IDMAP_COLS} for m in mapping.values()],
    )
    rl.write(map_path, rl.render_tables([idmap]))
    save_ledger(a.evidence, tables)
    print(f"ids: {len(mapping)} mapped, {added} rows added")


def cmd_coverage(a):
    """Copy each owner lane's read depth and tests (lane report Coverage rows) into this
    round's Coverage rows. Reports paths still `pending`; paths a lane lists that are not
    changed files are ignored."""
    rp = rl.round_dir(a.evidence, a.round)
    tables = load_ledger(a.evidence)
    cov = rl.find_table(tables, f"Coverage (round {a.round})")
    rows = {r["File"]: r for r in cov.rows}
    reports = read_reports(rp)
    own_reports = {stem for stem, _ in reports}
    updated = 0
    for stem, text in reports:
        lane = re.sub(r"-f\d+$", "", stem)
        for path, depth, tests in rl.parse_coverage(text):
            row = rows.get(path)
            if row and (
                row["Owner lane"] == lane or row["Owner lane"] not in own_reports
            ):
                row["Read depth"], row["Tests reaching it"] = depth, tests
                updated += 1
    save_ledger(a.evidence, tables)
    pending = [r["File"] for r in cov.rows if r["Read depth"] == "pending"]
    print(f"coverage: {updated} rows copied, {len(pending)} pending")
    for path in pending:
        print(f"pending: {path}")


def crosses(reported, final):
    return (reported in BLOCKING_TIERS) != (final in BLOCKING_TIERS)


def read_verify(path, fid):
    """The verifier block of path, or a reason string when it does not conform."""
    v = rl.parse_verifier(path.read_text(encoding="utf-8"))
    if v.get("ID") != fid:
        return f"{path.name}: ID is not {fid}"
    if v.get("Verdict") not in rl.VERDICTS:
        return f"{path.name}: bad Verdict"
    if v.get("Tier") not in rl.TIERS:
        return f"{path.name}: bad Tier"
    if v["Verdict"] == "PARTIAL" and not v.get("Claim"):
        return f"{path.name}: PARTIAL without Claim"
    return v


def cmd_verdicts(a):
    """Set verdict, status, final tier and (PARTIAL) the verifier's narrowed claim on each
    ledger row that has verify/<ID>.md; a verify/<ID>-2.md that differs from the first
    replaces it. Lists candidates without a verify file and findings that still need a
    second verifier (REFUTED at BLOCKING/IMPORTANT, or a final tier across the
    IMPORTANT/SUGGESTION line from the reported tier; PARTIAL keeping its side does not).
    """
    rp = rl.round_dir(a.evidence, a.round)
    tables = load_ledger(a.evidence)
    ft = rl.find_table(tables, "Findings")
    bad, updated = [], 0
    for row in ft.rows:
        status, fid = row["Status"], row["ID"]
        if status.startswith(("SUPERSEDED-BY", "FIXED")) or status == "ACCEPTED-RISK":
            continue
        first_path = rp / "verify" / f"{fid}.md"
        if not first_path.is_file():
            if not row["Verdict"]:
                print(f"missing: {fid}")
            continue
        vs = [read_verify(first_path, fid)]
        second_path = rp / "verify" / f"{fid}-2.md"
        if second_path.is_file():
            vs.append(read_verify(second_path, fid))
        if any(isinstance(v, str) for v in vs):
            bad += [v for v in vs if isinstance(v, str)]
            continue
        first, final = vs[0], vs[-1]
        reported = rl.tier_pair(row["Tier"])[0]
        if len(vs) == 1:
            why = None
            if first["Verdict"] == "REFUTED" and reported in BLOCKING_TIERS:
                why = f"REFUTED {reported}"
            elif crosses(reported, first["Tier"]):
                why = f"tier {reported} -> {first['Tier']}"
            if why:
                print(f"second verifier needed: {fid} ({why})")
        row["Verdict"] = final["Verdict"]
        row["Status"] = (
            final["Verdict"]
            if final["Verdict"] in ("REFUTED", "UNVERIFIED")
            else "OPEN"
        )
        row["Tier"] = f"{reported} -> {final['Tier']}"
        if final["Verdict"] == "PARTIAL":
            row["Claim"] = collapse(final["Claim"])
        updated += 1
    save_ledger(a.evidence, tables)
    print(f"verdicts: {updated} rows updated")
    for reason in bad:
        print(f"non-conforming: {reason}")
    if bad:
        raise rl.Fail(f"{len(bad)} non-conforming verify files")


def cmd_mlines(a):
    """Collect each lane's `M<n>:` lines (the last one per rule wins). Prints one line per
    rule; each FAIL and UNVERIFIED also prints an `ORCH process` filing line. A rule whose
    owner lane (M1/M3 TESTQ, M2 TESTB, M4/M5 AUTH) returned only `N/A:` is a vacuous
    PASS; any other rule with no line is missing and the command exits 1."""
    rp = rl.round_dir(a.evidence, a.round)
    found, kinds = {}, {}
    for stem, text in read_reports(rp):
        kinds[re.sub(r"-f\d+$", "", stem)] = rl.report_kind(text)
        for n, status, evidence in rl.parse_mlines(text):
            found[n] = (stem, status, evidence)
    missing = 0
    for n in sorted(M_OWNERS):
        owner = M_OWNERS[n]
        if n in found:
            stem, status, evidence = found[n]
            print(f"M{n} {status} ({stem}) {evidence}".rstrip())
            if status == "FAIL":
                print(f"file ORCH process BLOCKING: M{n} FAIL: {evidence}")
            elif status == "UNVERIFIED":
                print(
                    f"file ORCH process (tier per policy M-rules): M{n} UNVERIFIED: {evidence}"
                )
        elif kinds.get(owner) == "NA":
            print(f"M{n} PASS ({owner} N/A, vacuous)")
        else:
            missing += 1
            print(
                f"missing: M{n} ({owner}: "
                + ("no M-line" if owner in kinds else "no report")
                + ")"
            )
    if missing:
        raise rl.Fail(f"{missing} M-lines missing")


def cmd_counts(a):
    """Per final tier: OPEN, UNVERIFIED, candidate (no verdict), REFUTED and other closed
    rows; rows with a known-open-item; and the SKILL.md Completion check: Closure is OPEN
    while a BLOCKING or IMPORTANT row is OPEN, UNVERIFIED or a candidate, or this round's
    Lanes table has a FAILED lane or a STOP. Unresolved decisions are not read."""
    tables = load_ledger(a.evidence)
    ft = rl.find_table(tables, "Findings")
    counts = {t: {} for t in rl.TIERS}
    waits = []
    for r in ft.rows:
        final = rl.tier_pair(r["Tier"])[1]
        key = {
            "": "candidate",
            "OPEN": "OPEN",
            "UNVERIFIED": "UNVERIFIED",
            "REFUTED": "REFUTED",
        }.get(r["Status"], "closed")
        counts.setdefault(final, {})
        counts[final][key] = counts[final].get(key, 0) + 1
        if key in ("candidate", "OPEN", "UNVERIFIED") and final in BLOCKING_TIERS:
            waits.append(f"{r['ID']} ({final} {key})")
        if r["known-open-item"]:
            print(
                f"known-open-item: {r['ID']} {r['known-open-item']} ({r['Status'] or 'candidate'})"
            )
    for tier, c in counts.items():
        print(
            f"{tier}: "
            + " ".join(
                f"{k} {c.get(k, 0)}"
                for k in ("OPEN", "UNVERIFIED", "candidate", "REFUTED", "closed")
            )
        )
    for t in tables:
        if t.title == f"Lanes (round {a.round})":
            waits += [
                f"lane {r['Lane']} ({r['STOP'] and 'STOP' or r['Status']})"
                for r in t.rows
                if r["Status"].startswith("FAILED") or r["STOP"]
            ]
    print("Closure: " + ("OPEN; waiting: " + ", ".join(waits) if waits else "CLOSED"))


def cmd_set(a):
    """Set one finding's status by hand: `FIXED@<sha>` (needs --class, the evidence class of
    the proof that it no longer holds), `ACCEPTED-RISK` (needs --note, the human decision;
    also appends a Decisions row) or `OPEN` (reopen a closed row; `verdicts` then re-verifies
    it). Sets or clears the row's Round closed. `verdicts` never touches FIXED or
    ACCEPTED-RISK rows."""
    status = a.status
    if (
        status != "OPEN"
        and status != "ACCEPTED-RISK"
        and not re.fullmatch(r"FIXED@[0-9a-f]{7,40}", status)
    ):
        raise rl.Fail(f"status must be FIXED@<sha>, ACCEPTED-RISK or OPEN: {status}")
    if status.startswith("FIXED") and a.evidence_class not in (
        "rerun",
        "text-read",
        "not-exercised",
    ):
        raise rl.Fail("FIXED needs --class rerun|text-read|not-exercised")
    if status == "ACCEPTED-RISK" and not a.note:
        raise rl.Fail("ACCEPTED-RISK needs --note with the human decision")
    tables = load_ledger(a.evidence)
    rows = [r for r in rl.find_table(tables, "Findings").rows if r["ID"] == a.id]
    if not rows:
        raise rl.Fail(f"no finding row {a.id}")
    row = rows[0]
    row["Status"] = status
    row["Round closed"] = "" if status == "OPEN" else str(a.round)
    if status.startswith("FIXED"):
        row["Evidence"] = collapse(f"{a.evidence_class}: {a.note or row['Evidence']}")
    if status == "ACCEPTED-RISK":
        rl.find_table(tables, "Decisions").add(
            **{
                "Round": a.round,
                "Question": f"Accept the risk of {a.id}?",
                "Decision": collapse(a.note),
                "Findings affected": a.id,
            }
        )
    save_ledger(a.evidence, tables)
    print(f"{a.id}: {status}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("seed", cmd_seed),
        ("ids", cmd_ids),
        ("coverage", cmd_coverage),
        ("verdicts", cmd_verdicts),
        ("mlines", cmd_mlines),
        ("counts", cmd_counts),
        ("set", cmd_set),
    ):
        s = sub.add_parser(name, help=fn.__doc__.split("\n\n")[0].split(". ")[0])
        s.add_argument("evidence")
        s.add_argument("round", type=int)
        if name == "seed":
            s.add_argument("--lane", action="append", default=[], metavar="NAME=STATUS")
        if name == "set":
            s.add_argument("id")
            s.add_argument("status")
            s.add_argument("--class", dest="evidence_class")
            s.add_argument("--note", default="")
        s.set_defaults(fn=fn)
    args = p.parse_args(argv)
    try:
        args.fn(args)
    except rl.Fail as e:
        print(f"ledger.py {args.cmd}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
