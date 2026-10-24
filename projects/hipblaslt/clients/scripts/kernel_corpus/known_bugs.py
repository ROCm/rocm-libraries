# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The kernel corpus known-bugs list: kernels known to fail, each with a ticket and a review date.

Known-broken kernels still run on every pass. A failing case that matches an entry is reported
as known_fail instead of fail, so one tracked defect does not fail every run. The list cannot
hide a fix or sit unreviewed: an entry whose matching cases all pass, or whose review date has
passed, makes the run fail until someone removes or renews it.

An entry names a kernel by architecture and solution name, the TensileLite name that stays the
same when a library is retuned and runtime indices shift. A few solutions share a name and differ
only in split-K (custom-GSU kernels), so an entry may add global_split_u. It may also narrow the
match to a device name, a problem shape tag or a test profile.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import re
from pathlib import Path
from typing import Optional

# clients/scripts/kernel_corpus -> clients/tests/data, next to the client suite's known_bugs.yaml.
DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "data" / "kernel_corpus_known_bugs.yaml"
)

_REQUIRED = ("arch", "solution_name", "ticket", "added", "review_by", "reason")
_OPTIONAL = ("global_split_u", "device", "shape", "profile")
_TICKET = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")


@dataclasses.dataclass(frozen=True)
class Entry:
    arch: str
    solution_name: str
    ticket: str
    added: datetime.date
    review_by: datetime.date
    reason: str
    global_split_u: Optional[int] = None
    device: Optional[str] = None
    shape: Optional[str] = None
    profile: Optional[str] = None

    def matches(self, row: dict, record: Optional[dict]) -> bool:
        """Whether a result row, with its solution's corpus record, is covered by this entry."""
        if row.get("arch") != self.arch or row.get("name") != self.solution_name:
            return False
        if self.global_split_u is not None:
            gsu = (record or {}).get("sizes", {}).get("globalSplitU")
            if gsu != self.global_split_u:
                return False
        if self.device is not None and not re.search(self.device, row.get("device") or ""):
            return False
        if self.shape is not None and row.get("shape") != self.shape:
            return False
        if self.profile is not None and row.get("profile", "smoke") != self.profile:
            return False
        return True

    def label(self) -> str:
        # Names run to hundreds of characters and often differ only near the end, so a short
        # hash of the whole name tells apart entries whose prefixes match.
        digest = hashlib.sha1(self.solution_name.encode()).hexdigest()[:8]
        parts = [self.arch, f"{self.solution_name[:48]}... ({digest})"]
        if self.global_split_u is not None:
            parts.append(f"GSU {self.global_split_u}")
        for value in (self.device, self.shape, self.profile):
            if value is not None:
                parts.append(value)
        return " ".join(parts)


def _date(value: object, field: str, where: str) -> datetime.date:
    if isinstance(value, datetime.date):
        return value
    try:
        return datetime.date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"{where}: {field} must be a date, YYYY-MM-DD, not {value!r}") from None


def parse(data: object, source: str = "known bugs") -> list[Entry]:
    """Entries from the parsed YAML document. Raises ValueError naming the first bad entry."""
    if data is None:
        return []
    if not isinstance(data, dict) or not isinstance(data.get("entries", []), list):
        raise ValueError(f"{source}: expected a mapping with an 'entries' list")
    entries: list[Entry] = []
    seen = set()
    for i, raw in enumerate(data.get("entries") or []):
        where = f"{source}, entry {i + 1}"
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: must be a mapping")
        missing = [k for k in _REQUIRED if raw.get(k) in (None, "")]
        if missing:
            raise ValueError(f"{where}: missing {', '.join(missing)}")
        unknown = sorted(set(raw) - set(_REQUIRED) - set(_OPTIONAL))
        if unknown:
            raise ValueError(f"{where}: unknown field {', '.join(unknown)}")
        ticket = str(raw["ticket"])
        if not _TICKET.match(ticket):
            raise ValueError(f"{where}: ticket {ticket!r} is not a tracker key such as ROCM-12345")
        entry = Entry(
            arch=str(raw["arch"]),
            solution_name=str(raw["solution_name"]),
            ticket=ticket,
            added=_date(raw["added"], "added", where),
            review_by=_date(raw["review_by"], "review_by", where),
            reason=str(raw["reason"]),
            global_split_u=(
                int(raw["global_split_u"]) if raw.get("global_split_u") is not None else None
            ),
            device=raw.get("device"),
            shape=raw.get("shape"),
            profile=raw.get("profile"),
        )
        if entry.review_by < entry.added:
            raise ValueError(f"{where}: review_by is before added")
        key = (entry.arch, entry.solution_name, entry.global_split_u)
        key += (entry.device, entry.shape, entry.profile)
        if key in seen:
            raise ValueError(f"{where}: duplicates an earlier entry")
        seen.add(key)
        entries.append(entry)
    return entries


def load(path: Optional[Path]) -> list[Entry]:
    """Entries from a YAML file; no path means no list."""
    if path is None:
        return []
    import yaml

    return parse(yaml.safe_load(Path(path).read_text()), str(path))


@dataclasses.dataclass
class Verdict:
    """What one run says about one entry."""

    entry: Entry
    known_fail: int = 0
    passed: int = 0
    status: str = "not seen"

    def as_dict(self, today: datetime.date) -> dict:
        e = self.entry
        return {
            "ticket": e.ticket,
            "arch": e.arch,
            "solution_name": e.solution_name,
            "label": e.label(),
            "reason": e.reason,
            "added": e.added.isoformat(),
            "age_days": (today - e.added).days,
            "review_by": e.review_by.isoformat(),
            "known_fail_cases": self.known_fail,
            "passing_cases": self.passed,
            "status": self.status,
        }


# Entry statuses. The last two make the run fail.
KNOWN = "known failure"
NOT_SEEN = "not seen"
FIXED = "fixed: every matching case passed; remove the entry"
EXPIRED = "review date passed; renew it with a reason or remove it"
FAILING_STATUSES = (FIXED, EXPIRED)


def apply(
    entries: list[Entry],
    results: dict[str, dict],
    records: dict[int, dict],
    failing: tuple[str, ...],
    today: datetime.date,
) -> list[Verdict]:
    """Marks failing rows that match an entry as known_fail, in place, and judges every entry
    for the architectures these results cover."""
    archs = {r.get("arch") for r in results.values()}
    verdicts = [Verdict(e) for e in entries if e.arch in archs]
    for row in results.values():
        if row.get("shape") is None:
            continue
        record = records.get(row.get("index"))
        for verdict in verdicts:
            if not verdict.entry.matches(row, record):
                continue
            if row["status"] in failing:
                row["status"] = "known_fail"
                row["ticket"] = verdict.entry.ticket
                verdict.known_fail += 1
            elif row["status"] == "pass":
                verdict.passed += 1
            break
    for verdict in verdicts:
        if verdict.entry.review_by < today:
            verdict.status = EXPIRED
        elif verdict.known_fail:
            verdict.status = KNOWN
        elif verdict.passed:
            verdict.status = FIXED
        else:
            verdict.status = NOT_SEEN
    return verdicts
