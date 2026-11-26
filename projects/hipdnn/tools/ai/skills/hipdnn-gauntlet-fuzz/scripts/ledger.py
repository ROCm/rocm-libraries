#!/usr/bin/env python3
"""IDs, the engine ledger and the stage gates of a fuzz round (formats: ../findings.md).

    ledger.py ids  <R> <LEDGER>                 assign final IDs to every idea and finding of round R, add ledger rows
    ledger.py gate lanes  <R>                   every dispatched search lane reported in the format
    ledger.py gate decide <R> [<LEDGER>]        every idea has a decision; decisive kept ideas are verified; sets idea Status
    ledger.py gate triage <R>                   every triage lane reported; every group in groups.tsv is explained by a finding
    ledger.py gate verify <R> <LEDGER>          every finding verified (second verifier where needed); sets ledger verdicts
    ledger.py gate report <R> <LEDGER>          FINDINGS.md cites every finding; no row of this round left PROPOSED or unverified

Prints one `missing: ...` line per problem and exits 1 when a gate fails; prints `ok: ...` and exits 0 when it holds.
"""
import os
import re
import sys

IDEA_FIELDS = ["ID", "Graph", "Why", "Expect", "Judgeable", "Covered", "Place"]
FINDING_FIELDS = ["ID", "Kind", "Severity", "Groups", "Graphs", "What", "Proof", "Verdict", "Fix"]
EXPECT = {"PASS", "DECLINED", "WRONG", "CRASH", "ACCEPT-UNSERVABLE", "DECLINE-SERVABLE", "FRONTEND-GAP", "BACKEND-MANGLE",
          "KERNEL-UNREACHED"}
DECISIVE = {"WRONG", "CRASH", "ACCEPT-UNSERVABLE", "DECLINE-SERVABLE", "FRONTEND-GAP", "BACKEND-MANGLE"}
KINDS = {"defect", "regression", "applicability", "conditioning", "tolerance", "divergence", "pre-existing", "harness-gap",
         "unknown"}
SEVERITIES = {"HIGH", "MEDIUM", "LOW", "INFO"}
DECISIONS = {"axis", "probe", "kernel-rows", "hand-probe", "unreachable", "dropped", "known"}
VERDICTS = {"CONFIRMED", "REFUTED", "PARTIAL", "UNVERIFIED"}
LEDGER_HEAD = ["ID", "Type", "Kind", "Severity", "Verdict", "Status", "Round", "Summary"]

problems = []


def miss(msg):
    problems.append(msg)
    print("missing:", msg)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


def family(lane):
    """FEAT2 and GROUP3 share the FEAT and GROUP ID series."""
    return re.sub(r"\d+$", "", lane)


def blocks(text, section):
    """Field dicts of every block under `## <section>`; None when the section is absent; [] for a `None:` section."""
    m = re.search(r"^## " + section + r"\b.*$", text, re.M)
    if not m:
        return None
    body = text[m.end():]
    nxt = re.search(r"^## ", body, re.M)
    body = body[:nxt.start()] if nxt else body
    out, cur, last = [], None, None
    for line in body.split("\n"):
        if not line.strip():
            if cur:
                out.append(cur)
            cur, last = None, None
            continue
        fm = re.match(r"^([A-Z][A-Za-z]*):\s?(.*)$", line)
        if fm and (cur is not None or fm.group(1) == "ID"):
            cur = cur if cur is not None else {}
            last = fm.group(1)
            cur[last] = fm.group(2).strip()
        elif cur is not None and last and line[:1].isspace():
            cur[last] += " " + line.strip()
    if cur:
        out.append(cur)
    return out


def parse_report(path, kind):
    """Validated blocks of one lane report, or None after printing what is wrong."""
    if not os.path.isfile(path):
        miss(f"{path}: no report")
        return None
    text = read(path)
    first = text.lstrip().split("\n", 1)[0]
    if first.startswith("STOP:"):
        miss(f"{path}: lane stopped: {first[5:].strip()}")
        return None
    section, fields = ("Ideas", IDEA_FIELDS) if kind == "search" else ("Findings", FINDING_FIELDS)
    bs = blocks(text, section)
    if bs is None:
        miss(f"{path}: no '## {section}' section")
        return None
    if not bs:
        m = re.search(r"^## " + section + r".*\n+\s*None:", text, re.M)
        if not m:
            miss(f"{path}: '## {section}' has no blocks and no 'None:' line")
            return None
        return []
    lane = os.path.splitext(os.path.basename(path))[0]
    ok = True
    for b in bs:
        bid = b.get("ID", "?")
        for f in fields:
            if not b.get(f):
                miss(f"{path}: {bid}: no {f}")
                ok = False
        if not re.fullmatch(re.escape(lane) + r"-p\d+", bid):
            miss(f"{path}: {bid}: provisional ID must be {lane}-p<n>")
            ok = False
        if kind == "search" and b.get("Expect") and b["Expect"].split()[0] not in EXPECT:
            miss(f"{path}: {bid}: Expect {b['Expect'].split()[0]} not one of {sorted(EXPECT)}")
            ok = False
        if kind == "triage":
            if b.get("Kind") and b["Kind"].split()[0] not in KINDS:
                miss(f"{path}: {bid}: Kind {b['Kind'].split()[0]} not one of {sorted(KINDS)}")
                ok = False
            if b.get("Severity") and b["Severity"].split()[0] not in SEVERITIES:
                miss(f"{path}: {bid}: Severity {b['Severity'].split()[0]} not one of {sorted(SEVERITIES)}")
                ok = False
    return bs if ok else None


def dispatched(r, kind):
    path = os.path.join(r, "dispatch.tsv")
    if not os.path.isfile(path):
        miss(f"{path}: no dispatch table")
        return []
    rows = []
    for line in read(path).split("\n"):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t") if "\t" in line else line.split()
        if cells[:3] == ["lane", "file", "kind"]:
            continue
        if len(cells) < 3:
            miss(f"{path}: bad row: {line}")
            continue
        if cells[2] == kind:
            p = cells[1] if os.path.isabs(cells[1]) else os.path.join(r, cells[1])
            rows.append((cells[0], p))
    return rows


def round_reports(r):
    """(kind, path) of every lane and triage report in round R."""
    out = []
    for sub, kind in (("lanes", "search"), ("triage", "triage")):
        d = os.path.join(r, sub)
        if os.path.isdir(d):
            out += [(kind, os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".md")]
    return out


def table(path):
    """Rows (cell lists, backticks stripped) of the first markdown table in a file; header and rule rows dropped."""
    if not os.path.isfile(path):
        return None
    rows = []
    for line in read(path).split("\n"):
        if line.startswith("|"):
            cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
                continue
            rows.append(cells)
    return rows[1:] if rows else []


def load_ledger(path):
    rows = table(path)
    return rows if rows is not None else []


def save_ledger(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = ["# Ledger", "", "| " + " | ".join(LEDGER_HEAD) + " |", "|" + "---|" * len(LEDGER_HEAD)]
    for r in rows:
        cells = [f"`{r[0]}`"] + [c.replace("|", "/") for c in r[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def load_idmap(r):
    """{(file, provisional): (final, type)}."""
    rows = table(os.path.join(r, "ID-MAP.md")) or []
    return {(row[2], row[0]): (row[1], row[3]) for row in rows if len(row) >= 4}


def round_name(r):
    a = os.path.abspath(r)
    return os.path.basename(os.path.dirname(a)).replace("gauntlet_", "") + "/" + os.path.basename(a)


def short(s, n=110):
    s = s.replace("|", "/")
    return s if len(s) <= n else s[:n - 3] + "..."


def cmd_ids(r, ledger):
    idmap = load_idmap(r)
    rows = load_ledger(ledger)
    used = {}
    for row in rows:
        m = re.fullmatch(r"([A-Z]+)-(\d+)", row[0])
        if m:
            used[m.group(1)] = max(used.get(m.group(1), 0), int(m.group(2)))
    for _, (final, _) in idmap.items():
        m = re.fullmatch(r"([A-Z]+)-(\d+)", final)
        if m:
            used[m.group(1)] = max(used.get(m.group(1), 0), int(m.group(2)))
    known = {row[0] for row in rows}
    rnd = round_name(r)
    added = 0
    for kind, path in round_reports(r):
        bs = parse_report(path, kind)
        if bs is None:
            continue
        rel = os.path.relpath(path, r).replace("\\", "/")
        fam = family(os.path.splitext(os.path.basename(path))[0])
        for b in bs:
            key = (rel, b["ID"])
            if key not in idmap:
                used[fam] = used.get(fam, 0) + 1
                idmap[key] = (f"{fam}-{used[fam]}", "idea" if kind == "search" else "finding")
            final, typ = idmap[key]
            if final in known:
                continue
            if typ == "idea":
                rows.append([final, "idea", b["Expect"].split()[0], "-", "-", "PROPOSED", rnd, short(b["Graph"])])
            else:
                rows.append([final, "finding", b["Kind"].split()[0], b["Severity"].split()[0], "-", "OPEN", rnd,
                             short(b["Verdict"])])
            known.add(final)
            added += 1
    lines = ["# ID map", "", "| Provisional | Final | File | Type |", "|---|---|---|---|"]
    for (rel, prov), (final, typ) in sorted(idmap.items(), key=lambda kv: kv[1][0]):
        lines.append(f"| `{prov}` | `{final}` | {rel} | {typ} |")
    with open(os.path.join(r, "ID-MAP.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    save_ledger(ledger, rows)
    if problems:
        return 1
    print(f"ok: {len(idmap)} IDs in ID-MAP.md, {added} new ledger rows")
    return 0


def blocks_by_final(r):
    """{final: (type, block)} for every block of the round that has a final ID."""
    idmap = load_idmap(r)
    out = {}
    for kind, path in round_reports(r):
        if not os.path.isfile(path):
            continue
        bs = blocks(read(path), "Ideas" if kind == "search" else "Findings") or []
        rel = os.path.relpath(path, r).replace("\\", "/")
        for b in bs:
            if (rel, b.get("ID")) in idmap:
                final, typ = idmap[(rel, b["ID"])]
                out[final] = (typ, b)
    return out


def verdict_of(r, final):
    """(verdict, file) from verify/<ID>-2.md if present, else verify/<ID>.md; (None, path) when missing or malformed."""
    for name in (f"{final}-2.md", f"{final}.md"):
        p = os.path.join(r, "verify", name)
        if os.path.isfile(p):
            m = re.search(r"^Verdict:\s*([A-Z]+)", read(p), re.M)
            if not m or m.group(1) not in VERDICTS:
                miss(f"{p}: no Verdict line with one of {sorted(VERDICTS)}")
                return None, p
            return m.group(1), p
    return None, os.path.join(r, "verify", f"{final}.md")


def set_ledger(ledger, final, **cols):
    rows = load_ledger(ledger)
    for row in rows:
        if row[0] == final:
            for k, v in cols.items():
                row[LEDGER_HEAD.index(k)] = v
    save_ledger(ledger, rows)


def gate_lanes(r):
    rows = dispatched(r, "search")
    for _, path in rows:
        parse_report(path, "search")
    return len(rows)


def gate_decide(r, ledger):
    if not os.path.isfile(os.path.join(r, "ID-MAP.md")):
        miss("ID-MAP.md: run `ledger.py ids` first")
        return 0
    ideas = {f: b for f, (t, b) in blocks_by_final(r).items() if t == "idea"}
    rows = table(os.path.join(r, "DECISIONS.md"))
    if rows is None:
        miss("DECISIONS.md: no decisions table")
        return 0
    dec = {}
    for row in rows:
        if len(row) < 2:
            continue
        if row[1] not in DECISIONS:
            miss(f"DECISIONS.md: {row[0]}: decision {row[1]!r} not one of {sorted(DECISIONS)}")
        dec[row[0]] = row[1]
    for final, b in sorted(ideas.items()):
        if final not in dec:
            miss(f"DECISIONS.md: no row for {final}")
            continue
        if b["Expect"].split()[0] in DECISIVE and dec[final] not in ("dropped", "known"):
            v, p = verdict_of(r, final)
            if v is None:
                miss(f"{os.path.relpath(p, r)}: decisive idea {final} ({b['Expect'].split()[0]}) placed as {dec[final]} unverified")
            elif ledger:
                set_ledger(ledger, final, Verdict=v)
        if ledger:
            set_ledger(ledger, final, Status=dec[final].upper())
    return len(ideas)


def gate_triage(r):
    rows = dispatched(r, "triage")
    named = set()
    for _, path in rows:
        for b in parse_report(path, "triage") or []:
            named |= {g.strip() for g in b["Groups"].split(",") if g.strip()}
    gpath = os.path.join(r, "groups.tsv")
    if not os.path.isfile(gpath):
        miss("groups.tsv: no group table")
        return len(rows)
    groups = set()
    for line in read(gpath).split("\n")[1:]:
        cells = line.split("\t")
        if len(cells) >= 4 and cells[3].strip():
            groups.add(cells[3].strip())
    for g in sorted(groups - named):
        miss(f"groups.tsv: group {g} is named by no finding's Groups field")
    for g in sorted(named - groups):
        miss(f"a finding names group {g}, which groups.tsv does not have")
    return len(rows)


def gate_verify(r, ledger):
    findings = {f: b for f, (t, b) in blocks_by_final(r).items() if t == "finding"}
    if not findings:
        miss("no finding has a final ID: run `ledger.py ids` after the triage gate")
    for final, b in sorted(findings.items()):
        first = os.path.join(r, "verify", f"{final}.md")
        if not os.path.isfile(first):
            miss(f"verify/{final}.md: not verified")
            continue
        v, _ = verdict_of(r, final)
        if v is None:
            continue
        m1 = re.search(r"^Verdict:\s*([A-Z]+)", read(first), re.M)
        v1 = m1.group(1) if m1 else v
        if b["Severity"].split()[0] in ("HIGH", "MEDIUM") and v1 in ("REFUTED", "UNVERIFIED") \
                and not os.path.isfile(os.path.join(r, "verify", f"{final}-2.md")):
            miss(f"verify/{final}-2.md: second verifier needed ({b['Severity'].split()[0]} {v1})")
            continue
        status = {"REFUTED": "REFUTED"}.get(v)
        set_ledger(ledger, final, Verdict=v, **({"Status": status} if status else {}))
    return len(findings)


def gate_report(r, ledger):
    path = os.path.join(r, "FINDINGS.md")
    if not os.path.isfile(path):
        miss("FINDINGS.md: not written")
        return 0
    text = read(path)
    finals = [f for f, (t, _) in blocks_by_final(r).items() if t == "finding"]
    for f in sorted(finals):
        if not re.search(r"(?<![A-Za-z0-9-])" + re.escape(f) + r"(?![0-9])", text):
            miss(f"FINDINGS.md: does not cite {f}")
    rnd = round_name(r)
    for row in load_ledger(ledger):
        if row[6] != rnd:
            continue
        if row[5] == "PROPOSED":
            miss(f"ledger: {row[0]} still PROPOSED")
        if row[1] == "finding" and row[4] == "-":
            miss(f"ledger: {row[0]} has no verdict")
    return len(finals)


def main(argv):
    if len(argv) >= 3 and argv[0] == "ids":
        return cmd_ids(argv[1], argv[2])
    if len(argv) >= 3 and argv[0] == "gate":
        stage, r = argv[1], argv[2]
        ledger = argv[3] if len(argv) > 3 else None
        need_ledger = stage in ("verify", "report")
        if need_ledger and not ledger:
            print(f"usage: ledger.py gate {stage} <R> <LEDGER>")
            return 2
        fn = {"lanes": lambda: gate_lanes(r), "decide": lambda: gate_decide(r, ledger),
              "triage": lambda: gate_triage(r), "verify": lambda: gate_verify(r, ledger),
              "report": lambda: gate_report(r, ledger)}.get(stage)
        if fn is None:
            print(__doc__)
            return 2
        n = fn()
        if problems:
            print(f"gate {stage}: FAIL ({len(problems)} problem(s))")
            return 1
        print(f"ok: gate {stage} holds ({n} item(s))")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
