# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The audit set never benches a kernel the runs it is built for leave out
of their pools: `run_orchestrator --mode audit` on the fake bench with the
blocklist of its config (a list) and of a source run (its gate's, named by
the run's newest config snapshot), which grows while the set is benched and
shrinks once it is complete; the set's own blocklist keeps what it left out
once. The manifest names every blocklist by its sha256 and the kernels it
left out by digest, the completeness check of any run never waits for them,
and an evaluation treats them as outside the pool. A second set (`grown`) is
stopped once after a part's stage03 and once in an increment before the
blocklist names more kernels: the finished stage03 stands, and an increment
of blocked kernels alone is planned again. A third (`shrunk`) benches parts
after the run's blocklist no longer names a kernel the set left out."""

import hashlib
import json
import signal
from pathlib import Path

import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
import run_orchestrator as ro
import test_audit as ta
from lib import audit as al
from lib import correctness as cx
from lib import hazards as hz
from lib.bank_pool import config_pool
from lib.bench_yaml import line_solution_list, read_bench_lines

# The fake grid names a kernel by its macro tile, so a name blocks the two
# kernels of a macro tile: those the config's list blocks, those the source
# run's gate blocks from the start, and those it blocks from the second
# invocation on.
NAMES = {int(k["index"]): k["name"] for k in fh.kernels_grid()}
LISTED, GATE, LATER = {4, 5}, {2, 3}, {8, 9}
BLOCKED = LISTED | GATE | LATER


def name_of(sids):
    (name,) = {NAMES[s] for s in sids}
    return name


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    pytest.importorskip("tilewright")
    root = tmp_path_factory.mktemp("audit_blocklist")
    mp = pytest.MonkeyPatch()
    try:
        yield _build(root, mp)
    finally:
        mp.undo()


def _build(root: Path, mp: pytest.MonkeyPatch) -> dict:
    fh.unhide_devices(mp)
    bench_root = root / "bench"
    build = bench_root / "projects" / "hipblaslt" / "build" / "release"
    fh.write_fake_build(build, ta.ARCH, bf.LIBRARY_STEM, fh.kernels_grid())
    libdir = build / "Tensile" / "library" / ta.ARCH
    mp.setenv("FAKE_LIBRARY_DIR", str(libdir))
    mp.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    benched = root / "benched"
    benched.mkdir()
    mp.setenv("FAKE_BENCH_YAML_DIR", str(benched))
    audit = root / "audit"
    known = root / "known.tsv"
    known.write_text(f"X\t{name_of(LISTED)}\n")
    cfg = ta.base_config(bench_root, audit)
    cfg["contexts"] = {"list": [{"cu_budget": 0}]}
    cfg["correctness"] = {"lists": [str(known)]}
    assert ro.validate_config(cfg) == ([], [])
    cfg_path = root / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    source = root / "source_run"

    def gate_blocks(stamp, groups):
        """A snapshot of the source run whose gate blocks `groups`."""
        gate = source / "correctness" / "blocklists" / f"{stamp}_full.json"
        entries = [cx.Entry(name_of(g), source="correctness gate") for g in groups]
        cx.write_blocklist(gate, cx.Blocklist(entries))
        snap = source / "configs" / f"{stamp}_full.yaml"
        snap.parent.mkdir(parents=True, exist_ok=True)
        sec = {"enabled": True, "lists": [str(known)], "blocklist": str(gate)}
        snap.write_text(yaml.safe_dump(dict(cfg, correctness=sec)))

    cmd = ["--config", str(cfg_path), "--mode", "audit", "--run-dir", str(audit)]
    cmd += ["--source-run-dir", str(source)]
    real = ro.Pipeline.run_stage

    def interrupting(self, owner_dir, key, args, round_idx, out_dir=None):
        if Path(owner_dir).name == "part_0001" and key == "stage03":
            raise ro.Interrupted(signal.SIGTERM)
        return real(self, owner_dir, key, args, round_idx, out_dir)

    gate_blocks("20261019_000001", [GATE])
    mp.setattr(ro.Pipeline, "run_stage", interrupting)
    first_rc = ro.main(cmd)
    mp.setattr(ro.Pipeline, "run_stage", real)
    first_benched = set(benched.glob("*.yaml"))
    first = al.manifest(audit)
    part1 = audit / "bench" / al.EXCLUSIVE / "part_0001"
    part1_lines = (part1 / "shapes.yaml").read_text()
    gate_blocks("20261019_000002", [GATE, LATER])
    second_rc = ro.main(cmd)
    second = al.manifest(audit)
    second_benched = set(benched.glob("*.yaml")) - first_benched
    gate_blocks("20261019_000003", [GATE])
    third_rc = ro.main(cmd)
    return {
        "root": root,
        "audit": audit,
        "cfg": cfg,
        "libdir": libdir,
        "first_rc": first_rc,
        "first": first,
        "first_benched": first_benched,
        "part1": part1,
        "part1_lines": part1_lines,
        "second_rc": second_rc,
        "second": second,
        "second_benched": second_benched,
        "benched": benched,
        "third_rc": third_rc,
        "third": al.manifest(audit),
    }


def _listed(path):
    out = set()
    for ln in read_bench_lines(path):
        out |= set(line_solution_list(ln) or [])
    return out


# The pool's later kernels: NEW_X joins it first, NEW_Y later, each a macro
# tile of its own (so a name of its own). The source run's hazard canary
# keeps CUT out unchecked and quarantines HUNG.
NEW_X, NEW_Y = {10, 11}, {12, 13}
CUT, HUNG = {4, 5}, {6, 7}
MI = ((16, 16, 128, 1), (32, 32, 64, 1))


def _new(sids, mt):
    return [bf.solution(s, s, mt=mt, mi=mi) for s, mi in zip(sorted(sids), MI)]


@pytest.fixture(scope="module")
def grown(tmp_path_factory):
    pytest.importorskip("tilewright")
    root = tmp_path_factory.mktemp("audit_blocklist_grown")
    mp = pytest.MonkeyPatch()
    try:
        yield _build_grown(root, mp)
    finally:
        mp.undo()


def _build_grown(root: Path, mp: pytest.MonkeyPatch) -> dict:
    """A set whose source run's gate blocks GATE, and whose hazard canary
    keeps CUT out unchecked (a deploy staged a sidecar naming it meanwhile)
    and quarantines HUNG. Its first invocation stops in the near-winner
    re-reads of exclusive part 0, after the part's stage03; the gate then
    blocks LATER, which that part's lines list, and the second invocation
    completes the set. The pool gains NEW_X, and the third invocation stops
    in the stage03 of the increment of NEW_X; the gate then blocks NEW_X,
    the run's blocklist no longer names CUT or HUNG, the pool gains NEW_Y,
    and the fourth invocation completes the set."""
    fh.unhide_devices(mp)
    bench_root = root / "bench"
    build = bench_root / "projects" / "hipblaslt" / "build" / "release"
    grid = fh.kernels_grid()
    fh.write_fake_build(build, ta.ARCH, bf.LIBRARY_STEM, grid)
    libdir = build / "Tensile" / "library" / ta.ARCH
    mp.setenv("FAKE_LIBRARY_DIR", str(libdir))
    mp.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    benched = root / "benched"
    benched.mkdir()
    mp.setenv("FAKE_BENCH_YAML_DIR", str(benched))
    audit = root / "audit"
    cfg = ta.base_config(bench_root, audit)
    cfg["contexts"] = {"list": [{"cu_budget": 0}]}
    assert ro.validate_config(cfg) == ([], [])
    cfg_path = root / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    source = root / "source_run"
    x, y = _new(NEW_X, (64, 128, 64)), _new(NEW_Y, (128, 64, 64))
    names = {**NAMES, **{int(k["index"]): k["name"] for k in x + y}}

    def gate_blocks(stamp, groups, canary=False):
        gate = source / "correctness" / "blocklists" / f"{stamp}_full.json"
        entries = [cx.Entry(names[min(g)], source="correctness gate") for g in groups]
        if canary:
            entries += [_cut(names[min(CUT)]), _hung(names[min(HUNG)])]
        cx.write_blocklist(gate, cx.Blocklist(entries))
        snap = source / "configs" / f"{stamp}_full.yaml"
        snap.parent.mkdir(parents=True, exist_ok=True)
        sec = {"enabled": True, "blocklist": str(gate)}
        snap.write_text(yaml.safe_dump(dict(cfg, correctness=sec)))

    cmd = ["--config", str(cfg_path), "--mode", "audit", "--run-dir", str(audit)]
    cmd += ["--source-run-dir", str(source)]
    real = ro.Pipeline.run_stage
    calls: list = []

    def stopping_at(part, key):
        def run_stage(self, owner_dir, key_, args, round_idx, out_dir=None):
            calls.append((Path(owner_dir), key_))
            if Path(owner_dir) == part and key_ == key:
                raise ro.Interrupted(signal.SIGTERM)
            return real(self, owner_dir, key_, args, round_idx, out_dir)

        return run_stage

    part0 = audit / "bench" / al.EXCLUSIVE / "part_0000"
    inc_part0 = audit / "increments" / "inc_001" / "bench" / al.EXCLUSIVE
    inc_part0 = inc_part0 / "part_0000"
    out: dict = {"root": root, "audit": audit, "benched": benched, "part0": part0}
    out.update(cfg=cfg, libdir=libdir)
    staged = source / "round_1" / "stage06" / "staged"
    staged.mkdir(parents=True)
    side = cx.sidecar_name(bf.LIBRARY_STEM)
    (staged / side).write_text(
        cx.sidecar_text(bf.LIBRARY_STEM, [names[s] for s in GATE | CUT])
    )
    target = {"library_stem": bf.LIBRARY_STEM, "blocklist": side}
    target["staged"] = str(staged / "model.tilewright.bin")
    (staged.parent / "deploy_record.json").write_text(
        json.dumps({"blocklist": {}, "targets": [target]})
    )
    gate_blocks("20261019_000001", [GATE], canary=True)
    mp.setattr(ro.Pipeline, "run_stage", stopping_at(part0, "stage03b"))
    out["first_rc"] = ro.main(cmd)
    out["first"] = al.manifest(audit)
    out["part0_lines"] = (part0 / "shapes.yaml").read_text()
    out["part0_stage03"] = ro.stage_complete(part0 / "stage03")
    gate_blocks("20261019_000002", [GATE, LATER], canary=True)
    calls.clear()
    mp.setattr(ro.Pipeline, "run_stage", stopping_at(None, None))
    out["second_rc"] = ro.main(cmd)
    out["second_calls"] = list(calls)
    out["second"] = al.manifest(audit)
    bf.write_logic(libdir / f"{bf.LIBRARY_STEM}.dat", grid + x)
    mp.setattr(ro.Pipeline, "run_stage", stopping_at(inc_part0, "stage03"))
    out["third_rc"] = ro.main(cmd)
    inc = audit / "increments" / "inc_001" / "increment.json"
    out["third_increment"] = json.loads(inc.read_text())
    mp.setattr(ro.Pipeline, "run_stage", real)
    gate_blocks("20261019_000003", [GATE, LATER, NEW_X])
    bf.write_logic(libdir / f"{bf.LIBRARY_STEM}.dat", grid + x + y)
    out["fourth_rc"] = ro.main(cmd)
    out["fourth"] = al.manifest(audit)
    out["fourth_increment"] = json.loads(inc.read_text())
    return out


def test_a_part_whose_stage03_finished_keeps_its_lines_when_a_kernel_is_blocked(
    grown,
):
    """A part whose stage03 benched a kernel before the blocklist named it
    is not benched again: its near-winner re-reads go on from its stage03,
    which leave out what the blocklist names, and its lines stay as they
    were benched; the parts benched later leave the kernel out, and no
    generation records it."""
    assert grown["first_rc"] == 128 + signal.SIGTERM
    assert grown["part0_stage03"]
    assert grown["second_rc"] == 0
    part0 = grown["part0"]
    assert not (part0 / "stale").exists()
    assert (part0 / "shapes.yaml").read_text() == grown["part0_lines"]
    assert LATER <= _listed(part0 / "shapes.yaml")
    assert [k for d, k in grown["second_calls"] if d == part0] == ["stage03b"]
    later = sorted(part0.parent.glob("part_*"))[1:]
    assert later
    for part in later:
        assert not LATER & _listed(part / "shapes.yaml"), part.name
    (gen0,) = grown["second"]["generations"]
    assert {int(s) for s in gen0["kernels"]} == set(NAMES) - GATE - LATER - HUNG


def test_an_increment_of_blocked_kernels_alone_is_moved_aside_and_planned_again(
    grown,
):
    """An increment planned for kernels the blocklist leaves out before it
    was benched is moved aside: the increment of that index is planned
    again with the kernels still pending, and no line ever lists the
    blocked ones."""
    assert grown["third_rc"] == 128 + signal.SIGTERM
    assert {int(s) for s in grown["third_increment"]["kernels"]} == NEW_X
    assert grown["fourth_rc"] == 0
    audit = grown["audit"]
    m = grown["fourth"]
    assert m["status"] == "complete"
    assert [g["index"] for g in m["generations"]] == [0, 1]
    assert {int(s) for s in m["generations"][1]["kernels"]} == NEW_Y
    assert {int(s) for s in grown["fourth_increment"]["kernels"]} == NEW_Y
    assert not (audit / "increments" / "inc_002").exists()
    (stale,) = (audit / "increments" / "stale").iterdir()
    moved = json.loads((stale / "inc_001" / "increment.json").read_text())
    assert {int(s) for s in moved["kernels"]} == NEW_X
    for path in grown["benched"].glob("*.yaml"):
        assert not _listed(path) & NEW_X, path.name
    inc = audit / "increments" / "inc_001" / "bench" / al.EXCLUSIVE
    assert {k for p in inc.glob("part_*") for k in _listed(p / "shapes.yaml")} == NEW_Y


def test_a_kernel_the_canary_only_kept_out_is_benched_and_a_quarantined_one_not(
    grown,
):
    """A kernel the source run's hazard canary only kept out unchecked is no
    finding: the set's blocklist does not take it from the run's blocklist
    nor from the sidecar a deploy staged meanwhile, so the set benches it,
    and an evaluation ranks it. The canary's quarantine is a finding: the
    set leaves the kernel out, and once a blocklist it applied named it the
    kernel stays out of the set and of every evaluation on it, also after
    the run's blocklist no longer names it."""
    audit = grown["audit"]
    first = grown["first"]
    taken = cx.read_list(audit / first["blocklist"]["file"])
    assert taken.names == {NAMES[min(GATE)], NAMES[min(HUNG)]}
    doc = json.loads((audit / first["blocklist"]["file"]).read_text())
    snapshot, sidecar = doc["sources"]
    assert snapshot["canary_unchecked"] == 1 and snapshot["kernels"] == 3
    assert sidecar["canary_unchecked"] == 1 and sidecar["kernels"] == 2
    plan = audit / "stage02" / "shapes.yaml"
    assert _listed(plan) == set(NAMES) - GATE - HUNG
    m = grown["fourth"]
    assert {int(s) for s in m["generations"][0]["kernels"]} >= CUT
    latest = cx.read_list(audit / m["blocklist"]["file"])
    assert NAMES[min(HUNG)] in latest.names and NAMES[min(CUT)] not in latest.names
    doc = json.loads((audit / m["blocklist"]["file"]).read_text())
    assert doc["sources"][-1] == {
        "from": "the blocklists the audit set applied before",
        "digest": cx.names_digest(cx.Blocklist([cx.Entry(NAMES[min(HUNG)])])),
        "kernels": 1,
    }
    left = sorted(m["left_out"].values())
    names = {
        **NAMES,
        **{int(k["index"]): k["name"] for k in _new(NEW_X, (64, 128, 64))},
    }
    assert left == sorted(names[s] for s in GATE | LATER | HUNG | NEW_X)
    pool = config_pool(grown["cfg"], grown["libdir"], bf.LIBRARY_STEM, 0)
    assert al.unbenched(m, al.pool_digests(pool.kernels)) == []
    spec = al.Spec(
        audit_dir=audit, library_dir=grown["libdir"], library_stem=bf.LIBRARY_STEM
    )
    kept, _ = al.evaluation_pool(spec, grown["cfg"])
    sids = {int(k["sol_idx_global"]) for k in kept.kernels}
    assert sids == {0, 1} | CUT | NEW_Y


@pytest.fixture(scope="module")
def shrunk(tmp_path_factory):
    pytest.importorskip("tilewright")
    root = tmp_path_factory.mktemp("audit_blocklist_shrunk")
    mp = pytest.MonkeyPatch()
    try:
        yield _build_shrunk(root, mp)
    finally:
        mp.undo()


def _build_shrunk(root: Path, mp: pytest.MonkeyPatch) -> dict:
    """A set whose source run's gate blocks GATE. Its first invocation stops
    before the stage03 of exclusive part 0, whose lines list LATER; the gate
    then blocks LATER too, which the set's blocklist names, and the second
    invocation stops in that part's near-winner re-reads; the run's
    blocklist then no longer names LATER, and the third invocation benches
    the parts left."""
    fh.unhide_devices(mp)
    bench_root = root / "bench"
    build = bench_root / "projects" / "hipblaslt" / "build" / "release"
    fh.write_fake_build(build, ta.ARCH, bf.LIBRARY_STEM, fh.kernels_grid())
    libdir = build / "Tensile" / "library" / ta.ARCH
    mp.setenv("FAKE_LIBRARY_DIR", str(libdir))
    mp.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    benched = root / "benched"
    benched.mkdir()
    mp.setenv("FAKE_BENCH_YAML_DIR", str(benched))
    audit = root / "audit"
    cfg = ta.base_config(bench_root, audit)
    cfg["contexts"] = {"list": [{"cu_budget": 0}]}
    cfg_path = root / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    source = root / "source_run"

    def gate_blocks(stamp, groups):
        gate = source / "correctness" / "blocklists" / f"{stamp}_full.json"
        entries = [cx.Entry(name_of(g), source="correctness gate") for g in groups]
        cx.write_blocklist(gate, cx.Blocklist(entries))
        snap = source / "configs" / f"{stamp}_full.yaml"
        snap.parent.mkdir(parents=True, exist_ok=True)
        sec = {"enabled": True, "blocklist": str(gate)}
        snap.write_text(yaml.safe_dump(dict(cfg, correctness=sec)))

    cmd = ["--config", str(cfg_path), "--mode", "audit", "--run-dir", str(audit)]
    cmd += ["--source-run-dir", str(source)]
    real = ro.Pipeline.run_stage
    part0 = audit / "bench" / al.EXCLUSIVE / "part_0000"

    def stopping_at(key):
        def run_stage(self, owner_dir, key_, args, round_idx, out_dir=None):
            if Path(owner_dir) == part0 and key_ == key:
                raise ro.Interrupted(signal.SIGTERM)
            return real(self, owner_dir, key_, args, round_idx, out_dir)

        return run_stage

    out: dict = {"audit": audit}
    gate_blocks("20261019_000001", [GATE])
    mp.setattr(ro.Pipeline, "run_stage", stopping_at("stage03"))
    out["first_rc"] = ro.main(cmd)
    out["plan"] = _listed(audit / "stage02" / "shapes.yaml")
    gate_blocks("20261019_000002", [GATE, LATER])
    mp.setattr(ro.Pipeline, "run_stage", stopping_at("stage03b"))
    out["second_rc"] = ro.main(cmd)
    out["second"] = al.manifest(audit)
    gate_blocks("20261019_000003", [GATE])
    before = set(benched.glob("*.yaml"))
    mp.setattr(ro.Pipeline, "run_stage", real)
    out["third_rc"] = ro.main(cmd)
    out["third_benched"] = set(benched.glob("*.yaml")) - before
    out["third"] = al.manifest(audit)
    return out


def test_a_part_benched_after_the_runs_blocklist_shrinks_leaves_out_what_the_set_did(
    shrunk,
):
    """A kernel a blocklist the set applied named stays out of every part
    the set benches later, also once the run's blocklist no longer names
    it: each invocation applies the union of the runs' blocklist and of
    those the set applied before."""
    assert shrunk["first_rc"] == 128 + signal.SIGTERM
    assert shrunk["second_rc"] == 128 + signal.SIGTERM
    assert shrunk["third_rc"] == 0
    assert LATER <= shrunk["plan"], "planned before the gate blocked it"
    second, third = shrunk["second"], shrunk["third"]
    assert name_of(LATER) in second["left_out"].values()
    assert shrunk["third_benched"]
    for path in shrunk["third_benched"]:
        assert not _listed(path) & LATER, path.name
    latest = cx.read_list(shrunk["audit"] / third["blocklist"]["file"])
    assert latest.names == {name_of(GATE), name_of(LATER)}
    (gen0,) = third["generations"]
    assert not {int(s) for s in gen0["kernels"]} & (GATE | LATER)
    assert third["status"] == "complete"


def test_the_set_never_benches_a_kernel_the_runs_leave_out(world):
    assert world["first_rc"] == 128 + signal.SIGTERM
    assert world["second_rc"] == 0 and world["third_rc"] == 0
    benched = sorted(world["benched"].glob("*.yaml"))
    assert benched
    for path in benched:
        assert not _listed(path) & (LISTED | GATE), path.name
    assert world["second_benched"]
    for path in world["second_benched"]:
        assert not _listed(path) & LATER, path.name
    plan = world["audit"] / "stage02"
    assert _listed(plan / "shapes.yaml") == set(NAMES) - LISTED - GATE
    marker = json.loads((plan / ro.STAGE_MARKER).read_text())
    assert "--listed-only" in json.dumps(marker)


def test_a_part_planned_before_its_kernel_was_blocked_is_benched_again(world):
    part1 = world["part1"]
    (stale,) = (part1 / "stale").iterdir()
    assert (stale / "shapes.yaml").read_text() == world["part1_lines"]
    assert LATER <= _listed(stale / "shapes.yaml")
    assert not LATER & _listed(part1 / "shapes.yaml")
    assert (part1 / ro.STAGE_MARKER).is_file()
    part0 = world["audit"] / "bench" / al.EXCLUSIVE / "part_0000"
    assert LATER <= _listed(part0 / "shapes.yaml"), "benched before they were blocked"


def test_the_manifest_names_every_blocklist_by_sha256(world):
    audit = world["audit"]
    first, second, third = world["first"], world["second"], world["third"]
    assert first["blocklist"]["kernels"] == 2
    assert [r["kernels"] for r in third["blocklists"]] == [2, 3, 3]
    assert len({r["sha256"] for r in third["blocklists"]}) == 3
    for rec in third["blocklists"]:
        data = (audit / rec["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == rec["sha256"]
        assert rec["file"].startswith("blocklists/source_")
    latest = cx.read_list(audit / third["blocklist"]["file"])
    assert latest.names == {name_of(LISTED), name_of(GATE), name_of(LATER)}
    assert third["blocklist"]["digest"] == cx.names_digest(latest)
    doc = json.loads((audit / third["blocklist"]["file"]).read_text())
    assert [s["from"] for s in doc["sources"]][-1] == (
        "the blocklists the audit set applied before"
    )
    assert doc["sources"][-1]["kernels"] == 1, "LATER, which the run no longer names"
    (gen0,) = second["generations"]
    assert gen0["blocklist"] == second["blocklist"]
    assert gen0["blocklist"]["kernels"] == 3
    left = sorted(third["left_out"].values())
    assert left == sorted(NAMES[s] for s in BLOCKED), "by digest, every kernel"


def test_a_generation_records_only_the_kernels_every_part_benched(world):
    (gen0,) = world["third"]["generations"]
    assert {int(s) for s in gen0["kernels"]} == set(NAMES) - BLOCKED
    assert not (world["audit"] / "increments").exists(), "no increment for LATER"


def _bare(world):
    """The training config without its blocklist: a run that blocks
    nothing."""
    return {k: v for k, v in world["cfg"].items() if k != "correctness"}


def test_no_run_waits_for_a_kernel_the_set_left_out(world):
    audit = world["audit"]
    pool = config_pool(_bare(world), world["libdir"], bf.LIBRARY_STEM, 0)
    everything = al.pool_digests(pool.kernels)
    assert sorted(everything) == sorted(NAMES)
    assert al.status(audit, everything) == ("ok", None)
    assert al.unbenched(world["third"], everything) == []
    m = dict(world["third"], left_out={})
    assert al.unbenched(m, everything) == sorted(BLOCKED)


def test_an_evaluation_treats_the_left_out_kernels_as_outside_the_pool(world):
    spec = al.Spec(
        audit_dir=world["audit"],
        library_dir=world["libdir"],
        library_stem=bf.LIBRARY_STEM,
    )
    bare = _bare(world)
    pool, applied = al.evaluation_pool(spec, bare)
    sids = {int(k["sol_idx_global"]) for k in pool.kernels}
    assert sids == set(NAMES) - BLOCKED
    assert applied["config"] == cx.applied_record(None)
    assert applied["audit_kernels"] == 3, "names"
    assert [r["sha256"] for r in applied["audit"]] == [
        r["sha256"] for r in world["third"]["blocklists"]
    ]
    gemms = al.evaluable_gemms(spec, bare)
    assert gemms
    for g in gemms.values():
        assert not {int(c["sol_idx_global"]) for c in g["candidates"]} & BLOCKED
    path = world["audit"] / world["third"]["blocklists"][0]["file"]
    kept = path.read_bytes()
    path.write_text("{}")
    try:
        with pytest.raises(ValueError, match="not the one it recorded"):
            al.evaluation_pool(spec, bare)
        null, prep = al._prepare(b"", spec, bare, None)
        assert prep is None and null["status"] == "error"
    finally:
        path.write_bytes(kept)


def test_the_source_blocklist_is_the_configs_and_each_source_runs(tmp_path):
    """The config's lists and the gate blocklist its snapshot names, and
    per source run the blocklist of its newest snapshot whose files can be
    read and the sidecars its deploys staged; each part named with its
    file's sha256."""
    lst = tmp_path / "l.tsv"
    lst.write_text("X\tCijk_listed\n")
    gate = tmp_path / "gate.json"
    cx.write_blocklist(gate, cx.Blocklist([cx.Entry("Cijk_gate")]))
    cfg = {"correctness": {"lists": [str(lst)], "blocklist": str(gate)}}
    run = tmp_path / "run"
    (run / "configs").mkdir(parents=True)
    old = tmp_path / "old.json"
    cx.write_blocklist(old, cx.Blocklist([cx.Entry("Cijk_old")]))
    for stamp, path in (("20261019_000001", old), ("20261019_000002", tmp_path / "x")):
        snap = run / "configs" / f"{stamp}_full.yaml"
        snap.write_text(yaml.safe_dump({"correctness": {"blocklist": str(path)}}))
        snap.with_name(snap.stem + ".source.yaml").write_text("not: a snapshot\n")
    blocked, parts = al.source_blocklist(cfg, [run])
    assert blocked.names == {"Cijk_listed", "Cijk_gate", "Cijk_old"}
    assert [p["from"] for p in parts] == ["the config"] * 2 + [f"source run {run}"]
    assert parts[-1]["snapshot"].endswith("20261019_000001_full.yaml")
    assert parts[-1]["sha256"] == hashlib.sha256(old.read_bytes()).hexdigest()
    # The sidecars the run's deploys staged join it; one that cannot be read
    # is noted.
    staged = run / "round_1" / "stage06" / "staged"
    staged.mkdir(parents=True)
    (staged / "s.tilewright.blocklist").write_text("# sidecar\nCijk_served\n")
    rec = {
        "blocklist": True,
        "targets": [
            {
                "library_stem": "s",
                "blocklist": "s.tilewright.blocklist",
                "staged": str(staged / "s.tilewright.bin"),
            }
        ],
    }
    (staged.parent / "deploy_record.json").write_text(json.dumps(rec))
    gone = run / "final_retrain" / "stage06"
    gone.mkdir(parents=True)
    rec["targets"][0]["staged"] = str(gone / "staged" / "s.tilewright.bin")
    (gone / "deploy_record.json").write_text(json.dumps(rec))
    cfg["hipblaslt"] = {"library_stem": "s"}
    blocked, parts = al.source_blocklist(cfg, [run])
    assert blocked.names == {"Cijk_listed", "Cijk_gate", "Cijk_old", "Cijk_served"}
    sidecars = [p for p in parts if "deploy" in p]
    assert [p["deploy"] for p in sidecars] == [
        "round_1/stage06/deploy_record.json",
        "final_retrain/stage06/deploy_record.json",
    ]
    assert sidecars[0]["kernels"] == 1 and "error" in sidecars[1]
    found, snap, _ = al.run_blocklist(tmp_path / "empty")
    assert (found, snap) == (None, None)
    old.unlink()
    with pytest.raises(ValueError, match="no config snapshot"):
        al.source_blocklist(cfg, [run])
    with pytest.raises(ValueError):
        al.source_blocklist({"correctness": {"lists": [str(tmp_path / "gone")]}})
    rec = al.write_blocklist_file(tmp_path / "a", "source", blocked, parts)
    again = al.write_blocklist_file(tmp_path / "a", "source", blocked, parts)
    assert rec == again and rec["kernels"] == 4
    assert rec["file"] == f"blocklists/source_{rec['sha256'][:16]}.json"
    assert cx.read_list(tmp_path / "a" / rec["file"]).names == blocked.names


def _cut(name):
    """The hazard canary's entry of a kernel it kept out unchecked."""
    why = "its bench was killed by SIGKILL from outside"
    return cx.Entry(
        name, hz.unchecked_reason(["prefetch_gl2_2"], why), source=hz.SOURCE
    )


def _hung(name):
    """The hazard canary's quarantine of a kernel whose run alone hung."""
    q = {"verdict": "hang", "classes": ["prefetch_gl2_2"], "reason": "no output"}
    return cx.Entry(name, hz.quarantine_reason(q), source=hz.SOURCE)


def test_the_source_blocklist_takes_no_kernel_the_canary_only_kept_out(tmp_path):
    """A kernel a run's hazard canary only kept out unchecked (its run alone
    cut short or never run), which no snapshot of the run names for a
    finding, is no finding: neither the config's blocklist, nor a source
    run's, nor a sidecar staged while the canary kept it out, puts it in the
    set's blocklist. The gate's findings and the canary's quarantine are,
    and so is a kernel the newest snapshot only keeps out unchecked that an
    older one names for a finding (the gate lists a kernel the canary keeps
    out with the canary's entry alone), also from a sidecar. The config's
    keep-out of a kernel no source run shows unchecked alone is taken, and
    without a source run every one is."""
    gate = tmp_path / "gate.json"
    found = cx.Entry("Cijk_failed", "wrong results", source="correctness gate")
    entries = [found, _cut("Cijk_cut"), _hung("Cijk_hung"), _cut("Cijk_once")]
    cx.write_blocklist(gate, cx.Blocklist(entries))
    assert hz.kept_out_unchecked(_cut("Cijk_cut"))
    assert not hz.kept_out_unchecked(_hung("Cijk_hung"))
    assert not hz.kept_out_unchecked(cx.Entry("Cijk_x", "hazard canary: unchecked"))
    listed = cx.Entry("Cijk_x", _cut("Cijk_x").reason, source="lists/known.tsv")
    assert not hz.kept_out_unchecked(listed), "a list's line keeps nothing out"
    cfg = {
        "correctness": {"blocklist": str(gate)},
        "hipblaslt": {"library_stem": "s"},
    }
    run = tmp_path / "run"
    (run / "configs").mkdir(parents=True)
    failed = cx.Entry("Cijk_failed_before", "wrong results", source="correctness gate")
    snaps = {
        "20261019_000001": [_cut("Cijk_once"), _hung("Cijk_hung_before"), failed],
        "20261019_000002": [
            _cut("Cijk_once"),
            _cut("Cijk_hung_before"),
            _cut("Cijk_failed_before"),
        ],
    }
    for stamp, entries in snaps.items():
        path = tmp_path / f"{stamp}.json"
        cx.write_blocklist(path, cx.Blocklist(entries))
        snap = run / "configs" / f"{stamp}_full.yaml"
        snap.write_text(yaml.safe_dump({"correctness": {"blocklist": str(path)}}))
    staged = run / "round_1" / "stage06" / "staged"
    staged.mkdir(parents=True)
    names = ["Cijk_once", "Cijk_failed_before", "Cijk_served"]
    (staged / "s.tilewright.blocklist").write_text(cx.sidecar_text("s", names))
    rec = {
        "blocklist": True,
        "targets": [
            {
                "library_stem": "s",
                "blocklist": "s.tilewright.blocklist",
                "staged": str(staged / "s.tilewright.bin"),
            }
        ],
    }
    (staged.parent / "deploy_record.json").write_text(json.dumps(rec))
    blocked, parts = al.source_blocklist(cfg, [run])
    assert blocked.names == {
        "Cijk_failed",
        "Cijk_cut",
        "Cijk_hung",
        "Cijk_hung_before",
        "Cijk_failed_before",
        "Cijk_served",
    }
    config, newest, sidecar = parts
    assert config["from"] == "the config" and config["canary_unchecked"] == 1
    assert newest["snapshot"].endswith("20261019_000002_full.yaml")
    assert newest["canary_unchecked"] == 1 and newest["kernels"] == 3
    assert sidecar["kernels"] == 3 and sidecar["canary_unchecked"] == 1
    assert al.run_unchecked(run).names == {"Cijk_once"}
    found, _, _ = al.run_blocklist(run)
    assert found.names == {"Cijk_hung_before", "Cijk_failed_before"}
    kept, n = al.verified_blocks(cx.read_list(gate), al.run_unchecked(run))
    assert kept.names == {"Cijk_failed", "Cijk_cut", "Cijk_hung"} and n == 1
    blocked, parts = al.source_blocklist(cfg)
    assert blocked.names == {"Cijk_failed", "Cijk_cut", "Cijk_hung", "Cijk_once"}
    assert "canary_unchecked" not in parts[0]


def test_the_set_runs_the_hazard_canary_of_its_config(tmp_path):
    """The set's stages run with the training config's hazard canary on
    (`bench_config` keeps its `hazards` section): a kernel the runs' canary
    only kept out unchecked, which the set takes, runs alone before any of
    the set's benches does."""
    cfg = ta.base_config(tmp_path, tmp_path / "a")
    for hazards in (None, {"enabled": True, "timeout_s": 60.0}):
        if hazards is not None:
            cfg["hazards"] = hazards
        built = al.bench_config(cfg, [5], tmp_path / "source.json")
        assert built.get("hazards") == cfg.get("hazards")
        assert hz.Settings.of(built).enabled
