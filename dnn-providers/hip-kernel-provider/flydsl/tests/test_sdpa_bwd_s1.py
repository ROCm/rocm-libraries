# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The S1 composer: membership precedence, the invariants it rejects, and its object handling.

A synthetic sweep repository (manifests committed in a throwaway git repo, objects in
sibling worktree dirs) stands in for the real sweep. Rows are named by a one-letter
hash; the Spec round trip is off because the specs are not real keys.
Pure Python (git only).
"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL / "tools"))
import sdpa_bwd_s1 as s1  # noqa: E402

COLUMNS = ("hash", "spec", "status", "attempts", "sha256", *s1.RESOURCE_COLUMNS)
GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def h(letter: str) -> str:
    return letter * 12


def row(letter: str, content: str, kernel: str = "dkdv", vgpr: int = 100) -> dict:
    """A manifest row for hash ``h(letter)`` whose object bytes are ``content``."""
    out = dict.fromkeys(COLUMNS, "1")
    out.update(
        hash=h(letter),
        spec=f"{kernel},bf16,64,{letter},bk,0,0,tail,0,1",
        status="ok",
        sha256=digest(content),
        vgpr=str(vgpr),
    )
    return out


class Fixture:
    """``<tmp>/results/sweep`` (git repo) + ``<tmp>/results/sweep-w`` (objects) + evidence files."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.repo = tmp / "results" / "sweep"
        self.repo.mkdir(parents=True)
        self.objects = tmp / "results" / "sweep-w" / "objects-a" / "objs"
        self.out = tmp / "out"
        self.rows: dict[str, list[dict]] = {}
        self.bytes: dict[str, str] = {}  # sha256 -> object content

    def manifest(self, name: str, rows: list[dict], contents: dict[str, str]) -> None:
        """Commit-to-be manifest ``name``; ``contents`` maps letter -> object bytes."""
        self.rows[name] = rows
        path = self.repo / name / "manifest.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        self.bytes.update({digest(c): c for c in contents.values()})

    def evidence(self, name: str, text: str) -> None:
        (self.tmp / name).write_text(text, encoding="utf-8")

    def object(self, letter: str, content: str) -> None:
        path = self.objects / h(letter)
        path.mkdir(parents=True, exist_ok=True)
        (path / "obj.hsaco").write_bytes(content.encode())

    def objects_for_all(self, *, skip: str = "") -> None:
        for rows in self.rows.values():
            for r in rows:
                if r["hash"][0] not in skip:
                    self.object(r["hash"][0], self.bytes[r["sha256"]])

    def commit(self) -> None:
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
            subprocess.run(
                ["git", "-C", str(self.repo), "-c", "commit.gpgsign=false", *args],
                check=True,
                env=GIT_ENV,
            )

    def run(self, selection: str, *, extra=(), reference: Path | None = None) -> int:
        path = self.tmp / "selection.toml"
        path.write_text(textwrap.dedent(selection), encoding="utf-8")
        return s1.run(
            path,
            self.repo,
            "HEAD",
            self.out,
            list(extra),
            reference,
            False,
            False,
            None,
            None,
        )

    def manifest_rows(self) -> dict[str, dict]:
        with (self.out / "manifest.csv").open(newline="", encoding="utf-8") as fh:
            return {r["hash"]: r for r in csv.DictReader(fh)}


BASE = """
    [[step]]
    name = "s0"
    kind = "base"
    manifests = ["git:objs-s0/manifest.csv"]
"""
STEP = '[[step]]\nname = "{name}"\nkind = "{kind}"\n'


@pytest.fixture
def fx(tmp_path: Path) -> Fixture:
    f = Fixture(tmp_path)
    f.manifest(
        "objs-s0",
        [row("a", "a0"), row("b", "b0", "dq"), row("c", "c0", "merged")],
        {"a": "a0", "b": "b0", "c": "c0"},
    )
    return f


def test_repin_replaces_the_sha_and_keeps_the_hash(fx: Fixture) -> None:
    fx.manifest("objs-new", [row("a", "a1", vgpr=90)], {"a": "a1"})
    fx.evidence(
        "delta.csv",
        f"hash,old_sha256,new_sha256,decision\n{h('a')},{digest('a0')},{digest('a1')},repin\n",
    )
    fx.objects_for_all()
    fx.commit()
    rc = fx.run(
        BASE
        + STEP.format(name="pin", kind="repin")
        + 'manifests = ["git:objs-new/manifest.csv"]\ndelta = "file:delta.csv"\nexpect = 1\n'
    )
    assert rc == 0
    rows = fx.manifest_rows()
    assert set(rows) == {h("a"), h("b"), h("c")}
    assert rows[h("a")]["sha256"] == digest("a1")
    assert rows[h("a")]["old_sha256"] == digest("a0")
    assert rows[h("a")]["vgpr"] == "90"
    assert rows[h("a")]["spec"] == fx.rows["objs-s0"][0]["spec"]
    assert rows[h("a")]["object_state"] == "found"
    assert rows[h("b")]["old_sha256"] == ""


def test_repin_against_a_different_old_sha_is_rejected(fx: Fixture) -> None:
    fx.manifest("objs-new", [row("a", "a1")], {"a": "a1"})
    fx.evidence(
        "delta.csv",
        f"hash,old_sha256,new_sha256,decision\n{h('a')},{digest('zz')},{digest('a1')},repin\n",
    )
    fx.commit()
    with pytest.raises(s1.S1Error, match="replaces"):
        fx.run(
            BASE
            + STEP.format(name="pin", kind="repin")
            + 'manifests = ["git:objs-new/manifest.csv"]\ndelta = "file:delta.csv"\n'
        )


def test_repin_of_a_non_member_needs_an_ignore_entry(fx: Fixture) -> None:
    fx.manifest("objs-new", [row("a", "a1"), row("e", "e1")], {"a": "a1", "e": "e1"})
    fx.objects_for_all()
    fx.commit()
    repin = (
        BASE
        + STEP.format(name="pin", kind="repin")
        + 'manifests = ["git:objs-new/manifest.csv"]\n'
    )
    with pytest.raises(s1.S1Error, match="not a member"):
        fx.run(repin)
    ignore = f'ignore = {{ "{h("e")}" = "superseded" }}\nexpect = 1\n'
    assert fx.run(repin + ignore) == 0
    assert "superseded" in (fx.out / "SNAPSHOT.md").read_text()


def removal(name: str, listing: str) -> str:
    return (
        STEP.format(name=name, kind="remove")
        + f'list = "file:{listing}"\nhashes_column = "object"\n'
        + 'spec_column = "spec"\ncovering_column = "covering_object"\n'
    )


def rm_row(fx: Fixture, victim: str, covering: str) -> str:
    spec = next(r["spec"] for r in fx.rows["objs-s0"] if r["hash"] == h(victim))
    return f'{h(victim)},"{spec}",{h(covering)}\n'


def test_removal_requires_a_surviving_covering_object(fx: Fixture) -> None:
    fx.objects_for_all()
    fx.commit()
    head = "object,spec,covering_object\n"
    fx.evidence("rm.csv", head + rm_row(fx, "b", "c"))
    assert fx.run(BASE + removal("rm", "rm.csv") + "expect = 1\n") == 0
    assert set(fx.manifest_rows()) == {h("a"), h("c")}
    # a is the covering object of b, and a later step removes a
    fx.evidence("rm.csv", head + rm_row(fx, "b", "a"))
    fx.evidence("rm2.csv", head + rm_row(fx, "a", "c"))
    with pytest.raises(s1.S1Error, match="covering object"):
        fx.run(BASE + removal("rm", "rm.csv") + removal("rm2", "rm2.csv"))


def test_removal_of_a_non_member_or_a_changed_spec_is_rejected(fx: Fixture) -> None:
    fx.commit()
    head = "object,spec,covering_object\n"
    fx.evidence("rm.csv", head + f'{h("z")},"dq,x",{h("a")}\n')
    with pytest.raises(s1.S1Error, match="not a member"):
        fx.run(BASE + removal("rm", "rm.csv"))
    fx.evidence("rm.csv", head + f'{h("b")},"dq,other",{h("a")}\n')
    with pytest.raises(s1.S1Error, match="not 'dq,other'"):
        fx.run(BASE + removal("rm", "rm.csv"))


def add(name: str, *manifests: str) -> str:
    listed = ", ".join(f'"git:{m}/manifest.csv"' for m in manifests)
    return STEP.format(name=name, kind="add") + f"manifests = [{listed}]\n"


def test_an_add_that_collides_with_a_member_is_rejected(fx: Fixture) -> None:
    fx.manifest("objs-x", [row("b", "b0", "dq")], {"b": "b0"})
    fx.commit()
    with pytest.raises(s1.S1Error, match="collision"):
        fx.run(BASE + add("x", "objs-x"))


def test_a_hash_added_by_two_steps_is_rejected(fx: Fixture) -> None:
    fx.manifest("objs-x", [row("e", "e0")], {"e": "e0"})
    fx.manifest("objs-y", [row("e", "e0")], {"e": "e0"})
    fx.commit()
    with pytest.raises(s1.S1Error, match="double add: e{12} is a member"):
        fx.run(BASE + add("x", "objs-x") + add("y", "objs-y"))


def test_a_step_unions_overlapping_manifests_but_rejects_conflicting_rows(
    fx: Fixture,
) -> None:
    fx.manifest("objs-x", [row("e", "e0"), row("f", "f0")], {"e": "e0", "f": "f0"})
    fx.manifest("objs-y", [row("e", "e0")], {"e": "e0"})
    fx.objects_for_all()
    fx.commit()
    union = BASE + add("u", "objs-x", "objs-y") + "expect = 2\n"
    assert fx.run(union) == 0
    fx.manifest("objs-y", [row("e", "e1")], {"e": "e1"})
    fx.commit()
    with pytest.raises(s1.S1Error, match="disagrees on sha256"):
        fx.run(union)


def test_an_excluded_hash_cannot_be_added(fx: Fixture) -> None:
    fx.manifest("objs-x", [row("e", "e0")], {"e": "e0"})
    fx.commit()
    exclude = f'[[exclude]]\nreason = "comparator"\nhashes = ["{h("e")}"]\n'
    with pytest.raises(s1.S1Error, match="excluded"):
        fx.run(BASE + add("x", "objs-x") + exclude)


def test_expected_counts_are_asserted(fx: Fixture) -> None:
    fx.objects_for_all()
    fx.commit()
    with pytest.raises(s1.S1Error, match="expected 4 rows, got 3"):
        fx.run(BASE + "expect = 4\n")
    with pytest.raises(s1.S1Error, match="per-kernel counts"):
        fx.run(BASE + "[expect.kernels]\ndkdv = 1\ndq = 1\nmerged = 2\n")
    fx.run(
        BASE + "[expect]\ntotal = 3\n[expect.kernels]\ndkdv = 1\ndq = 1\nmerged = 1\n"
    )


def test_members_without_a_binary_are_listed_for_rebuild_and_block_the_reference_dir(
    fx: Fixture,
) -> None:
    fx.objects_for_all(skip="b")
    fx.commit()
    reference = fx.tmp / "ref"
    assert fx.run(BASE, reference=reference) == s1.EXIT_REBUILD_REQUIRED
    with (fx.out / "rebuild-required.csv").open(newline="", encoding="utf-8") as fh:
        listed = list(csv.DictReader(fh))
    assert [(r["hash"], r["kernel"], r["sha256"]) for r in listed] == [
        (h("b"), "dq", digest("b0"))
    ]
    assert fx.manifest_rows()[h("b")]["object_state"] == "rebuild-required"
    assert not reference.exists()
    assert not list(fx.tmp.glob("ref.tmp-*"))

    # the rebuilt binary, supplied as an object root, completes the run
    rebuilt = fx.tmp / "rebuilt"
    rebuilt.mkdir()
    (rebuilt / f"{h('b')}.hsaco").write_bytes(b"b0")
    assert fx.run(BASE, extra=[rebuilt], reference=reference) == 0
    assert {p.name for p in reference.iterdir()} == {f"{h(x)}.hsaco" for x in "abc"}
    assert (reference / f"{h('b')}.hsaco").read_bytes() == b"b0"
    with pytest.raises(s1.S1Error, match="exists"):
        fx.run(BASE, extra=[rebuilt], reference=reference)


def test_a_binary_with_another_sha_fails_instead_of_being_skipped(fx: Fixture) -> None:
    fx.objects_for_all()
    fx.object("a", "tampered")
    fx.commit()
    with pytest.raises(s1.S1Error, match="none equals the pin"):
        fx.run(BASE)


def test_a_stale_copy_never_shadows_the_binary_with_the_pin_sha(fx: Fixture) -> None:
    fx.objects_for_all()
    stale = fx.tmp / "results" / "sweep-a-old" / "objects-a" / "objs" / h("a")
    stale.mkdir(parents=True)
    (stale / "obj.hsaco").write_bytes(b"old")
    fx.commit()
    assert fx.run(BASE) == 0
    path = f"sweep-w/objects-a/objs/{h('a')}/obj.hsaco"
    assert fx.manifest_rows()[h("a")]["object_path"] == path


def test_flaky_sources_merge_and_hashes_that_are_not_members_fail(fx: Fixture) -> None:
    fx.objects_for_all()
    fx.commit()
    fx.evidence("flaky.csv", f"hash,spec,source\n{h('a')},x,bimodal+attempts\n")
    flaky = (
        '[[flaky]]\nkind = "csv"\npath = "file:flaky.csv"\n'
        f'[[flaky]]\nkind = "hashes"\nhashes = ["{h("a")}", "{h("b")}"]\nsource = "t2"\n'
    )
    assert fx.run(BASE + flaky) == 0
    with (fx.out / "known-flaky.csv").open(newline="", encoding="utf-8") as fh:
        got = {r["hash"]: r["source"] for r in csv.DictReader(fh)}
    assert got == {h("a"): "bimodal+attempts+t2", h("b"): "t2"}
    stray = f'[[flaky]]\nkind = "hashes"\nhashes = ["{h("z")}"]\nsource = "t2"\n'
    with pytest.raises(s1.S1Error, match="not a member"):
        fx.run(BASE + stray)


def test_a_rerun_writes_identical_files(fx: Fixture) -> None:
    fx.objects_for_all()
    fx.commit()
    fx.run(BASE)
    first = {p.name: p.read_bytes() for p in fx.out.iterdir()}
    fx.run(BASE)
    assert {p.name: p.read_bytes() for p in fx.out.iterdir()} == first
