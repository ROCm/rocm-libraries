# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Print the lessons.md sections one reader needs, so a review pass does not read the whole file.

Usage: python -B lessons_for.py verifier|quality|worker|orchestrator [--part K]
       python -B lessons_for.py --check

Prints the file's introduction (everything before the first "## " heading), then what the role reads, in file order:
whole "## " sections, "### " subsections, and the lessons a team member file cites (a lesson is its "- L-NNN · " line
and the indented lines under it), each under its headings; then, for the verifier and the code quality reviewer,
the review passes' commands card (lane_commands.md, beside this file); then a last line naming what was left out.
Headings are matched by prefix, so renaming one means updating ROLES below.

Bash keeps an output over 30,000 characters out of the context: it saves it to a file with a 2k preview, and the
reader reads that file again. So `--part K` prints part K, at most PART_BYTES, cut between subsections; its last line
names the next part. Without `--part`, the whole extract prints.

The site lessons (one site's hosts, people, tickets and pools; same format, never packaged) are the file
<SITE_LESSONS_FILE>, in the monitor's state, which the install names (the registry's paths.site_lessons): their
sections join the role's extract after lessons.md's, headed "(site lessons)", and match the same selectors. While
there is none: lessons.md alone.

`--check` prints each role's size and its parts' sizes, and exits 1 when a part is over LIMIT_BYTES, a lesson a team
member file cites is in neither file, the card is missing, or the site lessons cannot be read. Sizes are UTF-8 bytes,
never fewer than the characters Bash counts. Run it after editing lessons.md, the card or a review pass's team member file
(weekly.md).
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
LESSONS, CARD, TEAM = (
    HERE / "lessons.md",
    HERE / "lane_commands.md",
    HERE.parents[1] / "team_members",
)
if pathlib.Path(
    "<LESSONS_FILE>"
).is_file():  # the editable copy (the package: paths.lessons, lessons.md)
    LESSONS = pathlib.Path("<LESSONS_FILE>")
PART_BYTES, LIMIT_BYTES = 28_000, 29_000  # Bash's output limit is 30,000 characters
ROOM = 600  # kept in a part for its last line and a "(continued)" heading

ROLES = {
    # rocke_review_verifier.md: the Verifiers and Domain facts sections (two parts), and the card
    "verifier": {"take": ("## Verifiers", "## Domain facts that change"), "card": True},
    # rocke_code_quality_reviewer.md: the layout, host and run facts, the lessons it cites, and the card
    "quality": {
        "take": (
            "### Develop layout and lanes",
            "### This host",
            "### Running things here",
        ),
        "cites": "rocke_code_quality_reviewer.md",
        "card": True,
    },
    "worker": {"take": ("## Worker",)},
    "orchestrator": {
        "take": (
            "## Orchestrator",
            "## Domain facts that change",
            "## Enforced by the toolkit",
            "## Open problems",
        )
    },
}
LESSON, CITED = re.compile(r"- (L-\d{3}) · "), re.compile(r"\bL-\d{3}\b")
# The site lessons (paths.site_lessons), in the monitor's state.
SITE = pathlib.Path("<SITE_LESSONS_FILE>")


def site_file() -> pathlib.Path | None:
    """The site lessons' file, SITE; None while there is none."""
    return SITE if SITE.is_file() else None


def site_blocks() -> tuple[list[tuple[str, str | None, str]], str | None]:
    """The site lessons' blocks, each section headed "(site lessons)", and a note when they cannot be read."""
    site = site_file()
    if site is None:
        return [], None
    try:
        return [
            (f"{s} (site lessons)", sub, t)
            for s, sub, t in blocks(site.read_text(encoding="utf-8"))[1]
        ], None
    except OSError as e:
        return [], f"the site lessons {site} cannot be read ({e.strerror or e})"


def blocks(text: str) -> tuple[str, list[tuple[str, str | None, str]]]:
    """The introduction, and the rest as (section, subsection, text) blocks in file order: one per "### " subsection,
    its heading included, and one for a section's text before its first subsection (none when that is blank).
    """
    intro, out = [], []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            out.append([line.rstrip("\n"), None, ""])
        elif line.startswith("### ") and out:
            out.append([out[-1][0], line.rstrip("\n"), line])
        elif out:
            out[-1][2] += line
        else:
            intro.append(line)
    return "".join(intro), [
        (s, sub, t if sub else t.lstrip("\n")) for s, sub, t in out if t.strip()
    ]


def lessons(text: str) -> dict[str, str]:
    """The lessons in a text by ID, in file order (an ID that comes twice keeps both texts)."""
    out, cur = {}, None
    for line in text.splitlines(keepends=True):
        if m := LESSON.match(line):
            cur = m[1]
            out[cur] = out.get(cur, "") + line
        elif cur and line.startswith(" "):
            out[cur] += line
        else:
            cur = None
    return out


def extract(role: str) -> dict:
    """What the role reads: the introduction, the units [(section, text)] in file order (the card last, under no
    section), the selectors that matched nothing, what it left out, the cited IDs and those lessons.md lacks.
    """
    spec, (intro, blks), (site, site_note) = (
        ROLES[role],
        blocks(LESSONS.read_text(encoding="utf-8")),
        site_blocks(),
    )
    blks = blks + site
    ids = (
        sorted(set(CITED.findall((TEAM / spec["cites"]).read_text(encoding="utf-8"))))
        if "cites" in spec
        else []
    )
    units, used, found, left = [], set(), set(), {}
    for sec, sub, text in blks:
        hit = [
            t for t in spec["take"] if sec.startswith(t) or (sub or "").startswith(t)
        ]
        mine = {k: v for k, v in lessons(text).items() if k in ids}
        found |= set(mine)
        if hit:
            used.update(hit)
            units.append((sec, text))
            continue
        if mine:
            head = (
                f"{sub} (only the lessons cited for this role)\n"
                if sub
                else "(only the lessons cited for this role)\n"
            )
            units.append((sec, head + "".join(mine.values()) + "\n"))
        left.setdefault(sec, []).append((sub[4:] if sub else "", bool(mine)))
    card = bool(spec.get("card")) and CARD.is_file()
    if card:
        units.append((None, CARD.read_text(encoding="utf-8")))
    taken = {s for s, _t in units}
    names = (
        []
    )  # a section nothing was taken from, or the subsections of one left out, "(in part)" when partly taken
    for sec, subs in left.items():
        part = [n + (" (in part)" if p else "") for n, p in subs if n]
        names.append(
            sec[3:]
            if sec not in taken
            else f"{sec[3:]}: {', '.join(part)}" if part else f"{sec[3:]} (in part)"
        )
    return {
        "intro": intro,
        "units": units,
        "unmatched": [t for t in spec["take"] if t not in used],
        "left": names,
        "ids": ids,
        "missing": [k for k in ids if k not in found],
        "no_card": bool(spec.get("card")) and not card,
        "site_note": site_note,
    }


def render(
    intro: str, units: list[tuple[str | None, str]], cont: str | None = None
) -> str:
    """The units under their "## " headings; `cont` is the section the part before ended in, marked continued."""
    out, shown = [intro], None
    for sec, text in units:
        if sec and sec != shown:
            out.append(f"{sec}{' (continued)' if sec == cont else ''}\n\n")
        shown = sec
        out.append(text)
    return "".join(out)


def split(ex: dict) -> list[list[tuple[str | None, str]]]:
    """The units in parts: each part takes units while it stays within PART_BYTES (ROOM kept), at least one."""
    parts, cur = [], []
    for unit in ex["units"]:
        trial = render(
            "" if parts else ex["intro"],
            cur + [unit],
            parts[-1][-1][0] if parts else None,
        )
        if cur and len(trial.encode("utf-8")) + ROOM > PART_BYTES:
            parts.append(cur)
            cur = []
        cur.append(unit)
    return parts + [cur] if cur else parts


def show(role: str, ex: dict, k: int | None = None) -> str:
    """The whole extract (k None) or part k, with its last line."""
    parts = split(ex)
    left = f"left out {len(ex['left'])} sections: {'; '.join(ex['left'])}"
    if k is None:
        text, tail = render(ex["intro"], ex["units"]), f"(lessons_for {role}: {left}"
    else:
        cont = parts[k - 2][-1][0] if k > 1 else None
        text = render("" if k > 1 else ex["intro"], parts[k - 1], cont)
        tail = f"(lessons_for {role}, part {k} of {len(parts)}" + (
            f": next, `--part {k + 1}`" if k < len(parts) else f", the last; {left}"
        )
    if ex["missing"]:
        tail += f"; cited but in neither lessons file: {', '.join(ex['missing'])}"
    if ex["no_card"]:
        tail += f"; the commands card {CARD.name} is missing"
    if ex["site_note"]:
        tail += f"; {ex['site_note']}"
    return f"{text}\n{tail})\n"


def check() -> int:
    """Each role's size and its parts'; 1 when a part is over LIMIT_BYTES, a cited lesson is missing or the card."""
    bad = []
    print(
        f"lessons_for --check: UTF-8 bytes; parts of at most {PART_BYTES:,}, fail above {LIMIT_BYTES:,}; "
        f"site lessons: {site_file() or 'none'}"
    )
    for role in ROLES:
        ex = extract(role)
        sizes = [
            len(show(role, ex, k).encode("utf-8")) for k in range(1, len(split(ex)) + 1)
        ]
        over = [k for k, size in enumerate(sizes, 1) if size > LIMIT_BYTES]
        bad += [f"{role} part {k}" for k in over] + [
            f"{role}: {t} matches nothing" for t in ex["unmatched"]
        ]
        bad += [
            f"{role}: {ROLES[role]['cites']} cites {k}, in neither file"
            for k in ex["missing"]
        ]
        bad += [f"{role}: no {CARD.name}"] if ex["no_card"] else []
        bad += [f"{role}: {ex['site_note']}"] if ex["site_note"] else []
        print(
            f"{role}: whole {len(show(role, ex).encode('utf-8')):,}; "
            + "; ".join(
                f"part {k} {size:,} {'FAIL' if k in over else 'OK'}"
                for k, size in enumerate(sizes, 1)
            )
            + (
                f"; cites {', '.join(ex['ids'])}: {'all in it' if not ex['missing'] else 'some missing'}"
                if ex["ids"]
                else ""
            )
        )
    print("check OK" if not bad else "check FAILED: " + "; ".join(bad))
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if argv[1:] == ["--check"]:
        return check()
    usage = f"usage: python -B lessons_for.py {'|'.join(ROLES)} [--part K] | --check"
    ok = len(argv) == 2 or (
        len(argv) == 4 and argv[2] == "--part" and argv[3].isdigit()
    )
    if not ok or argv[1] not in ROLES:
        print(usage, file=sys.stderr)
        return 2
    ex = extract(argv[1])
    if ex["unmatched"]:
        print(
            f"lessons_for: a section for {argv[1]} is missing: {ex['unmatched']}",
            file=sys.stderr,
        )
        return 1
    k = int(argv[3]) if len(argv) == 4 else None
    if k is not None and not 1 <= k <= len(split(ex)):
        print(
            f"lessons_for: {argv[1]} has {len(split(ex))} part(s), not {k}",
            file=sys.stderr,
        )
        return 2
    sys.stdout.write(show(argv[1], ex, k))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(
        encoding="utf-8", newline="\n"
    )  # no CRLF from Windows Python: it counts toward the limit
    sys.exit(main(sys.argv))
