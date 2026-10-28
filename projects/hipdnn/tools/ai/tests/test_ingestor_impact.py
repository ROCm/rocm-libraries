# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for hipdnn-ingestor-impact's impact.py (kernel ABI and dispatch facts of a diff).

Every case builds a throwaway git repository under tmp_path with paths the script classifies, so no
test depends on this clone's history or on rocKE. Git runs for real with the user's configuration
isolated.
"""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "hipdnn-ingestor-impact"
    / "scripts"
    / "impact.py"
)

HKP = "dnn-providers/hip-kernel-provider/"
BUILDER = HKP + "rocke/library/kernels/gfx950/attn.py"
KERNELS_INIT = HKP + "rocke/library/kernels/__init__.py"
OTHER_KERNEL = HKP + "rocke/library/kernels/gfx942/other.py"
PIN = HKP + "descriptor-packaging/tests/test_pin.py"
PACK = HKP + "src/engines/kernel_ingestor_engine/packs/Gfx950Native.cpp"
KDP = (
    HKP
    + "src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx950_attn/gfx950_attn.kdp.json"
)
MANIFEST = "projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attn.engine.yaml"
HEADER = "projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/ingestor/Plan.hpp"
UTIL = HKP + "descriptor-packaging/python/hkp_pack/util.py"
READER = HKP + "descriptor-packaging/tests/test_use.py"
GENERIC = "projects/hipdnn/plugin_sdk/data/gpu_generic_targets.json"

BUILD = """\
def build_attn(b, spec):
    b.param("q_ptr", PTR)
    b.param("k_ptr", PTR)
    b.param("scale", F32)
    if spec.sinks:
        b.param("sink_ptr", PTR)
    return b
"""
BUILD_REORDERED = BUILD.replace('    b.param("q_ptr", PTR)\n    b.param("k_ptr", PTR)\n',
                                '    b.param("k_ptr", PTR)\n    b.param("q_ptr", PTR)\n')  # fmt: skip
PIN_TEST = (
    'def test_sig(signature):\n    assert signature == ["q_ptr", "k_ptr", "scale"]\n'
)
PIN_TEST_NEW = PIN_TEST.replace('"q_ptr", "k_ptr"', '"k_ptr", "q_ptr"')
PACK_STATIC = """\
std::vector<KernelArgument> attnSignature()
{
    return {KernelArgument{"global_buffer", PTR, 0, "q_ptr"},
            KernelArgument{"global_buffer", PTR, 8, "k_ptr"},
            KernelArgument{"by_value", F32, 16, "scale"}};
}
"""
PACK_DYNAMIC = """\
std::vector<KernelArgument> attnSignature(const KernelDefinition& kernel)
{
    std::vector<KernelArgument> sig;
    sig.emplace_back(KernelArgument{"global_buffer", 8, 0, "q_ptr"});
    return sig;
}
"""
MANIFEST_YAML = f"""\
arch: gfx950
provider_root: {HKP.rstrip("/")}
launch_surface:
  - name: kernargs
    python_source: kernels/gfx950/attn.py:build_attn
    cpp_mirror: {PACK}
    kmd_fields: []
    guard: none
    test: none
"""
KDP_JSON = json.dumps(
    {
        "kernelDescriptors": [
            {"name": "v", "kernel_source": {"source": "kernels/gfx950/attn.py"}}
        ]
    }
)


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hipdnn_ingestor_impact", SCRIPT)
    assert spec and spec.loader, f"could not load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def isolated_git(tmp_path_factory, monkeypatch):
    empty = tmp_path_factory.mktemp("gitcfg") / "gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.com")


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return done.stdout.strip()


_COUNTER = {}


def commit(
    repo: Path, files: dict, message: str = "c", parent: str | None = None
) -> str:
    """One commit on top of `parent` (default: the last commit made in `repo`), written with a
    single fast-import so a test spawns few git processes. A None value deletes the file.
    """
    parent = parent or _COUNTER.get(repo)
    n = _COUNTER[str(repo) + "#n"] = _COUNTER.get(str(repo) + "#n", 0) + 1
    msg = message.encode()
    stream = [f"commit refs/heads/b{n}\nmark :1\ncommitter T <t@example.com> {n} +0000\n".encode(),
              b"data %d\n%s\n" % (len(msg), msg)]  # fmt: skip
    if parent:
        stream.append(f"from {parent}\n".encode())
    for path, content in files.items():
        if content is None:
            stream.append(f"D {path}\n".encode())
            continue
        data = content if isinstance(content, bytes) else content.encode("utf-8")
        stream.append(
            f"M 100644 inline {path}\ndata {len(data)}\n".encode() + data + b"\n"
        )
    marks = repo.parent / "marks"
    subprocess.run(["git", "-C", str(repo), "fast-import", "--quiet", f"--export-marks={marks}"],
                   input=b"".join(stream), check=True, capture_output=True)  # fmt: skip
    sha = marks.read_text().split()[1]
    _COUNTER[repo] = sha
    return sha


def make_repo(tmp_path: Path, base: dict) -> tuple:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    return repo, commit(repo, base, "base")


def branch_commit(repo: Path, start: str, files: dict, name: str) -> str:
    return commit(repo, files, name, parent=start)


def set_config(repo: Path, **values) -> None:
    with (repo / ".git" / "config").open("a", encoding="utf-8") as cfg:
        for key, value in values.items():
            section, name = key.split("__")
            cfg.write(f"[{section}]\n\t{name} = {value}\n")


def report(mod, capsys, repo, base, head, *extra, cmd="report") -> dict:
    rc = mod.main(
        [cmd, "--repo", str(repo), "--base", base, "--head", head, "--json", *extra]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    return json.loads(out)


def rows_of(data, kind=None, status=None):
    return [
        r
        for r in data["rows"]
        if (kind is None or r["kind"] == kind)
        and (status is None or r["status"] == status)
    ]


def file_class(data, path):
    return next(f["class"] for f in data["files"] if f["path"] == path)


def abi_repo(tmp_path, extra=None):
    """base: builder + literal pin; returns (repo, base)."""
    return make_repo(tmp_path, {BUILDER: BUILD, PIN: PIN_TEST, **(extra or {})})


def test_reordered_builder_with_unmoved_pin_is_mismatch(mod, capsys, tmp_path):
    # catches: copy search that misses literal lists in test files
    repo, base = abi_repo(tmp_path)
    head = commit(repo, {BUILDER: BUILD_REORDERED})
    data = report(mod, capsys, repo, base, head)
    (row,) = rows_of(data, "kernargs", "mismatch")
    assert row["path"] == BUILDER
    assert [(c["path"], c["moved"]) for c in row["counterparts"]] == [(PIN, False)]
    assert data["rollup"] == "Runtime impact: BREAKS"


def test_builder_and_copies_updated_together_is_changed(mod, capsys, tmp_path):
    # catches: any list edit reported as a break
    repo, base = abi_repo(tmp_path)
    head = commit(repo, {BUILDER: BUILD_REORDERED, PIN: PIN_TEST_NEW})
    data = report(mod, capsys, repo, base, head)
    assert not rows_of(data, status="mismatch")
    (row,) = rows_of(data, "kernargs")
    assert row["status"] == "changed" and row["counterparts"][0]["moved"] is True


def test_comment_only_touch_of_pin_is_not_moved(mod, capsys, tmp_path):
    # catches: a comment edit of a counterpart counted as the counterpart following the producer
    repo, base = abi_repo(tmp_path)
    head = commit(repo, {BUILDER: BUILD_REORDERED, PIN: "# touched\n" + PIN_TEST})
    data = report(mod, capsys, repo, base, head)
    assert [r["path"] for r in rows_of(data, "kernargs", "mismatch")] == [BUILDER]


def test_pin_on_target_tip_only_is_mismatch_in_merged_view_only(mod, capsys, tmp_path):
    # catches: merged view not computed, or computed against the merge base
    repo, base = make_repo(tmp_path, {BUILDER: BUILD})
    head = branch_commit(repo, base, {BUILDER: BUILD_REORDERED}, "feature")
    tip = branch_commit(repo, base, {PIN: PIN_TEST}, "target")
    data = report(mod, capsys, repo, base, head, "--target-tip", tip)
    (bad,) = rows_of(data, "kernargs", "mismatch")
    assert bad["merged_only"] and bad["view"] == "merged"
    assert [r["status"] for r in rows_of(data, "kernargs") if not r["merged_only"]] == [
        "changed"
    ]
    assert data["rollup"] == "Runtime impact: BREAKS"


def test_merge_conflict_on_a_surface_file_marks_the_merged_view_changed(
    mod, capsys, tmp_path
):
    # catches: a crash on conflict, a clean NONE when the race cannot be excluded, or rows
    # read from a file full of conflict markers
    repo, base = make_repo(tmp_path, {BUILDER: BUILD})
    head = branch_commit(
        repo, base, {BUILDER: BUILD.replace("scale", "scale_a")}, "feature"
    )
    tip = branch_commit(
        repo, base, {BUILDER: BUILD.replace("scale", "scale_b")}, "target"
    )
    data = report(mod, capsys, repo, base, head, "--target-tip", tip)
    merged = next(v for v in data["views"] if v["label"] == "merged")
    assert (
        merged["status"] == "conflicted"
        and "conflict" in merged["reason"]
        and BUILDER in merged["reason"]
    )
    assert data["rollup"] != "Runtime impact: NONE (1 files inspected)"
    merged_rows = [r for r in data["rows"] if r["view"] == "merged"]
    assert [(r["path"], r["status"]) for r in merged_rows] == [
        ("(merged view)", "changed")
    ]


def test_conflict_elsewhere_still_finds_the_merged_view_race(mod, capsys, tmp_path):
    # catches: one conflicted file, even one no runtime reads, discarding the whole merged
    # view and with it a pin race on files that merged cleanly
    doc = "AGENTS.md"
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, doc: "a\n"})
    head = branch_commit(repo, base, {BUILDER: BUILD_REORDERED, doc: "b\n"}, "feature")
    tip = branch_commit(repo, base, {PIN: PIN_TEST, doc: "c\n"}, "target")
    data = report(mod, capsys, repo, base, head, "--target-tip", tip)
    merged = next(v for v in data["views"] if v["label"] == "merged")
    assert merged["status"] == "conflicted" and doc in merged["reason"]
    (bad,) = rows_of(data, "kernargs", "mismatch")
    assert bad["merged_only"] and bad["view"] == "merged"
    (stand_in,) = [r for r in data["rows"] if r["path"] == "(merged view)"]
    assert stand_in["status"] == "outside"
    assert data["rollup"] == "Runtime impact: BREAKS"


def test_static_cpp_list_disagreeing_with_builder_is_mismatch(mod, capsys, tmp_path):
    # catches: static KernelArgument lists not parsed
    repo, base = abi_repo(tmp_path, {PACK: PACK_STATIC})
    head = commit(repo, {BUILDER: BUILD_REORDERED})
    data = report(mod, capsys, repo, base, head)
    paths = {
        c["path"]
        for r in rows_of(data, "kernargs", "mismatch")
        for c in r["counterparts"]
    }
    assert paths == {PACK, PIN}


def test_declared_pair_checks_static_list_and_flags_dynamic(mod, capsys, tmp_path):
    # catches: declared launch_surface pairs unused; a dynamic signature read as agreeing
    repo, base = abi_repo(tmp_path, {PACK: PACK_STATIC, MANIFEST: MANIFEST_YAML})
    head = commit(
        repo,
        {PACK: PACK_STATIC.replace("PTR, 0", "PTR, 0 ").replace('"k_ptr"', '"k_ptr2"')},
    )
    data = report(mod, capsys, repo, base, head)
    (row,) = rows_of(data, "kernargs", "mismatch")
    assert row["path"] == PACK and "disagrees" in row["reason"]
    repo2, base2 = abi_repo(
        tmp_path / "dyn", {PACK: PACK_DYNAMIC, MANIFEST: MANIFEST_YAML}
    )
    head2 = commit(repo2, {PACK: PACK_DYNAMIC + "// edit\nint x = 1;\n"})
    data2 = report(mod, capsys, repo2, base2, head2)
    dynamic = [r for r in rows_of(data2, "kernargs") if "dynamic" in r["reason"]]
    assert dynamic and not rows_of(data2, status="mismatch")
    assert data2["rollup"].startswith("Runtime impact: NEEDS-DEVICE gfx950")


def test_cpp_string_literal_edit_is_code_not_whitespace(mod, capsys, tmp_path):
    # catches: whitespace runs inside a literal collapsed like code ("attn  fwd" == "attn fwd")
    repo, base = make_repo(
        tmp_path, {PACK: 'const char* kName = "attn  fwd";\nint f() { return 1; }\n'}
    )
    head = commit(
        repo, {PACK: 'const char* kName = "attn fwd";\nint f() { return 1; }\n'}
    )
    assert (
        file_class(report(mod, capsys, repo, base, head, cmd="files"), PACK) == "code"
    )
    spaced = commit(
        repo, {PACK: 'const char* kName = "attn fwd";\nint   f()  {  return 1;   }\n'}
    )
    assert (
        file_class(report(mod, capsys, repo, head, spaced, cmd="files"), PACK)
        == "whitespace-only"
    )


def test_json_type_change_is_code_and_key_reorder_is_data_equal(mod, capsys, tmp_path):
    # catches: JSON compared without types (3 == 3.0); key order read as change
    repo, base = make_repo(tmp_path, {KDP: '{"a": 3, "b": [1, 2]}\n'})
    typed = commit(repo, {KDP: '{"a": 3.0, "b": [1, 2]}\n'})
    data = report(mod, capsys, repo, base, typed, cmd="data")
    assert (
        file_class(report(mod, capsys, repo, base, typed, cmd="files"), KDP) == "code"
    )
    assert [r["status"] for r in data["rows"]] == ["changed"]
    reordered = commit(repo, {KDP: '{"b": [1, 2], "a": 3.0}\n'})
    assert (
        file_class(report(mod, capsys, repo, typed, reordered, cmd="files"), KDP)
        == "data-equal"
    )
    assert [
        r["status"]
        for r in report(mod, capsys, repo, typed, reordered, cmd="data")["rows"]
    ] == ["inert"]


def test_removed_function_with_reader_is_mismatch_docstring_mention_is_not(
    mod, capsys, tmp_path
):
    # catches: word-match readers (docstrings) and removals with a live importer missed
    util = "def helper():\n    return 1\n\n\ndef keep():\n    return 2\n"
    used = "from hkp_pack.util import helper\n\n\ndef test_it():\n    assert helper() == 1\n"
    repo, base = make_repo(tmp_path, {UTIL: util, READER: used})
    head = commit(repo, {UTIL: "def keep():\n    return 2\n"})
    data = report(mod, capsys, repo, base, head)
    (row,) = rows_of(data, "symbols", "mismatch")
    assert row["name"] == "helper" and {c["path"] for c in row["counterparts"]} == {
        READER
    }
    docs_only = 'def test_it():\n    """No longer calls helper()."""\n'
    repo2, base2 = make_repo(tmp_path / "doc", {UTIL: util, READER: docs_only})
    head2 = commit(repo2, {UTIL: "def keep():\n    return 2\n"})
    assert not rows_of(report(mod, capsys, repo2, base2, head2), "symbols")


def test_removed_override_is_not_removed_while_a_base_defines_it(mod, capsys, tmp_path):
    # catches: a method dropped from a subclass reported removed although a base still provides it
    pkg = HKP + "descriptor-packaging/python/hkp_pack/"
    base_py = "class Base:\n    def helper(self):\n        return 1\n"
    sub_py = "from .base import Base\n\n\nclass Sub(Base):\n    def helper(self):\n        return 2\n"
    bare_sub = "from .base import Base\n\n\nclass Sub(Base):\n    pass\n"
    used = (
        "from hkp_pack.sub import Sub\n\n\ndef test_it():\n    assert Sub().helper()\n"
    )
    files = {pkg + "base.py": base_py, pkg + "sub.py": sub_py, READER: used}
    repo, base = make_repo(tmp_path / "a", files)
    head = commit(repo, {pkg + "sub.py": bare_sub})
    assert not rows_of(report(mod, capsys, repo, base, head), "symbols")
    repo, base = make_repo(tmp_path / "b", files)
    head = commit(
        repo, {pkg + "sub.py": bare_sub, pkg + "base.py": "class Base:\n    pass\n"}
    )
    mismatched = rows_of(report(mod, capsys, repo, base, head), "symbols", "mismatch")
    assert "Sub.helper" in {r["name"] for r in mismatched}
    unknown = sub_py.replace(
        "from .base import Base", "from vendor import Mystery"
    ).replace("(Base)", "(Mystery)")
    repo, base = make_repo(tmp_path / "c", {**files, pkg + "sub.py": unknown})
    head = commit(
        repo,
        {
            pkg
            + "sub.py": unknown.replace(
                "    def helper(self):\n        return 2\n", "    pass\n"
            )
        },
    )
    (row,) = rows_of(report(mod, capsys, repo, base, head), "symbols")
    assert row["status"] == "changed" and row["name"] == "Sub.helper"


def test_generic_target_member_removed_names_target_and_member(mod, capsys, tmp_path):
    # catches: membership diff absent; a data-equal call on a changed table
    table = {"schemaVersion": 1, "generics": {"gfx11-generic": ["gfx1100", "gfx1151"]}}
    repo, base = make_repo(tmp_path, {GENERIC: json.dumps(table)})
    table["generics"]["gfx11-generic"] = ["gfx1100"]
    head = commit(repo, {GENERIC: json.dumps(table)})
    (row,) = rows_of(report(mod, capsys, repo, base, head, cmd="data"))
    assert (
        row["name"] == "gfx11-generic:gfx1151"
        and "gfx11-generic" in row["reason"]
        and "gfx1151" in row["reason"]
    )
    assert row["status"] == "changed"


def test_empty_diff_is_an_error_never_none(mod, capsys, tmp_path):
    # catches: an empty diff reported as NONE
    repo, base = abi_repo(tmp_path)
    head = commit(repo, {}, "empty")
    assert (
        mod.main(["report", "--repo", str(repo), "--base", base, "--head", head]) == 1
    )
    captured = capsys.readouterr()
    assert captured.err.startswith("impact: ") and len(captured.err.splitlines()) == 1
    assert "NONE" not in captured.out


def test_user_git_config_does_not_change_results(mod, capsys, tmp_path):
    # catches: parsing that breaks under diff.noprefix, mnemonicPrefix or grep.column
    repo, base = abi_repo(tmp_path)
    set_config(
        repo,
        diff__noprefix="true",
        diff__mnemonicPrefix="true",
        grep__column="true",
        grep__lineNumber="true",
    )
    head = commit(repo, {BUILDER: BUILD_REORDERED})
    data = report(mod, capsys, repo, base, head)
    assert [r["path"] for r in rows_of(data, "kernargs", "mismatch")] == [BUILDER]
    assert rows_of(data, "kernargs", "mismatch")[0]["line"] == 1


def test_surface_code_not_understood_is_changed_and_comment_edits_are_inert(
    mod, capsys, tmp_path
):
    # catches: path-only inertness; a surface file that cannot be read reported inert
    repo, base = make_repo(tmp_path, {HEADER: "// a\nint f();\n", PACK: "int g();\n"})
    head = commit(repo, {HEADER: "// b\nint f();\n", PACK: b"\xff\xfe\x00bin"})
    data = report(mod, capsys, repo, base, head)
    assert {r["path"]: r["status"] for r in data["rows"]} == {
        HEADER: "inert",
        PACK: "changed",
    }
    comment = commit(repo, {HEADER: "// c\nint f();\n"})
    data = report(mod, capsys, repo, head, comment)
    assert data["rollup"] == "Runtime impact: NONE (1 files inspected)"


def test_no_surface_file_is_na(mod, capsys, tmp_path):
    # catches: docs and test-only diffs reported as a runtime impact
    repo, base = make_repo(tmp_path, {"projects/hipdnn/docs/x.md": "a\n", PIN: "# a\n"})
    head = commit(repo, {"projects/hipdnn/docs/x.md": "b\n", PIN: "# b\n"})
    data = report(mod, capsys, repo, base, head)
    assert {r["status"] for r in data["rows"]} == {"outside"}
    assert data["rollup"] == "Runtime impact: N/A (no surface file)"


def test_ancestor_init_of_a_consumed_module_is_consumed(mod, capsys, tmp_path):
    # catches: import_module running a package __init__ that no descriptor names
    base_files = {
        BUILDER: BUILD,
        KDP: KDP_JSON,
        KERNELS_INIT: "VERSION = 1\n",
        OTHER_KERNEL: "def f():\n    return 1\n",
    }
    repo, base = make_repo(tmp_path, base_files)
    head = commit(
        repo, {KERNELS_INIT: "VERSION = 2\n", OTHER_KERNEL: "def f():\n    return 2\n"}
    )
    data = report(mod, capsys, repo, base, head)
    status = {r["path"]: r["status"] for r in rows_of(data, "file")}
    assert status[KERNELS_INIT] == "changed"
    assert status[OTHER_KERNEL] == "inert"


def tree_of(repo, rev) -> str:
    return git(repo, "rev-parse", f"{rev}^{{tree}}")


def family(**fields) -> dict:
    """One emitted family of the builder, every variant equal unless `fields` say otherwise;
    `source` is relative to the rocKE library, as descriptors name it."""
    fam = dict(arch="gfx950", source="kernels/gfx950/attn.py", builder="build_attn", kdp=KDP,
               spec_origin="kdp-in-git", variants={"base": 1, "head": 1}, only_in_base=[],
               only_in_head=[], params_changed=[], ir_changed=[], comgr_changed=[],
               params={"base": None, "head": None}, companions={"base": None, "head": None},
               failures={"base": {}, "head": {}})  # fmt: skip
    fam.update(fields)
    return fam


def write_emit(path, repo, base, head, families=None, **fields):
    """An impact_emit document made for base..head, holding `families` (default: one family
    built from `fields`)."""
    doc = {"schema": 4, "tool": "impact_emit", "flavor": "llvm23", "base": base, "head": head,
           "base_tree": tree_of(repo, base), "head_tree": tree_of(repo, head),
           "families": [family(**fields)] if families is None else families,
           "not_inspected": [], "seconds": 1.0}  # fmt: skip
    Path(path).write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def test_emit_fold_equal_variants_make_shipped_module_edits_inert(
    mod, capsys, tmp_path
):
    # catches: a body edit of a shipped module always NEEDS-DEVICE even with proof of equal IR
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, KDP: KDP_JSON})
    head = commit(repo, {BUILDER: BUILD.replace("return b", "x = 1\n    return b")})
    plain = report(mod, capsys, repo, base, head)
    assert plain["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"
    equal = report(
        mod,
        capsys,
        repo,
        base,
        head,
        "--emit",
        write_emit(tmp_path / "eq.json", repo, base, head),
    )
    assert equal["rollup"] == "Runtime impact: NONE (1 files inspected)"
    changed = report(
        mod,
        capsys,
        repo,
        base,
        head,
        "--emit",
        write_emit(tmp_path / "ir.json", repo, base, head, ir_changed=["v"]),
    )
    assert changed["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"


def test_emit_params_changed_vs_unchanged_static_list_is_mismatch(
    mod, capsys, tmp_path
):
    # catches: emitted params never compared with the native pack's static list; mismatch downgraded
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, KDP: KDP_JSON, PACK: PACK_STATIC})
    head = commit(repo, {BUILDER: BUILD.replace("return b", "x = 1\n    return b")})
    wider = [["q_ptr", "ptr"], ["k_ptr", "ptr"], ["batch", "i32"], ["scale", "float"]]
    data = report(mod, capsys, repo, base, head, "--emit",
                  write_emit(tmp_path / "p.json", repo, base, head, params_changed=["v"], params={"base": None, "head": wider}))  # fmt: skip
    (row,) = rows_of(data, "kernargs", "mismatch")
    assert row["path"] == PACK and "batch" in row["reason"]
    assert data["rollup"] == "Runtime impact: BREAKS"
    # the same params as the static list: consistent, so changed rather than mismatch
    same = [["q_ptr", "ptr"], ["k_ptr", "ptr"], ["scale", "float"]]
    data = report(mod, capsys, repo, base, head, "--emit",
                  write_emit(tmp_path / "s.json", repo, base, head, params_changed=["v"], params={"base": None, "head": same}))  # fmt: skip
    assert not rows_of(data, status="mismatch")
    assert data["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"
    # a static mismatch is never folded away by an otherwise equal emit
    repo2, base2 = make_repo(
        tmp_path / "keep", {BUILDER: BUILD, PIN: PIN_TEST, KDP: KDP_JSON}
    )
    head2 = commit(repo2, {BUILDER: BUILD_REORDERED})
    data = report(
        mod,
        capsys,
        repo2,
        base2,
        head2,
        "--emit",
        write_emit(tmp_path / "e2.json", repo2, base2, head2),
    )
    assert data["rollup"] == "Runtime impact: BREAKS"


BODY_EDIT = BUILD.replace("return b", "x = 1\n    return b")
GT_HEADER = "projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/GpuGenericTargets.hpp"
HKP_PACK = HKP + "descriptor-packaging/python/hkp_pack/"


def sig(name, names):
    """A static `KernelArgument` list function holding `names`, all 4-byte by-value."""
    items = ",\n            ".join(f'KernelArgument{{"by_value", F32, {8 * i}, "{n}"}}' for i, n in enumerate(names))  # fmt: skip
    return f"std::vector<KernelArgument> {name}()\n{{\n    return {{{items}}};\n}}\n"


def test_removed_function_keeps_its_importer_when_another_file_defines_the_name(
    mod, capsys, tmp_path
):
    # catches: any same-named def elsewhere read as "moved", dropping every live importer
    util = "def helper():\n    return 1\n\n\ndef keep():\n    return 2\n"
    used = "from hkp_pack.util import helper\n\n\ndef test_it():\n    assert helper() == 1\n"
    other = HKP_PACK + "other_util.py"
    files = {UTIL: util, READER: used, other: "def helper():\n    return 3\n"}
    repo, base = make_repo(tmp_path / "a", files)
    head = commit(repo, {UTIL: "def keep():\n    return 2\n"})
    data = report(mod, capsys, repo, base, head)
    (row,) = rows_of(data, "symbols", "mismatch")
    assert row["name"] == "helper" and {c["path"] for c in row["counterparts"]} == {
        READER
    }
    # a real move: the reader now imports the new home, so nothing reads the old one
    repo, base = make_repo(tmp_path / "b", {UTIL: util, READER: used})
    moved = {UTIL: "def keep():\n    return 2\n", other: "def helper():\n    return 1\n",
             READER: used.replace("hkp_pack.util", "hkp_pack.other_util")}  # fmt: skip
    head = commit(repo, moved)
    assert not rows_of(report(mod, capsys, repo, base, head), "symbols", "mismatch")


def test_generic_targets_header_code_edit_is_changed_not_na(mod, capsys, tmp_path):
    # catches: the header tagged a surface but dropped as if it were the JSON table
    text = "inline bool isGeneric(int a) { return a > 3; }\n"
    repo, base = make_repo(tmp_path, {GT_HEADER: text})
    head = commit(repo, {GT_HEADER: text.replace("> 3", "> 4")})
    data = report(mod, capsys, repo, base, head)
    assert [r["status"] for r in data["rows"] if r["path"] == GT_HEADER] == ["changed"]
    assert data["rollup"].startswith("Runtime impact: NEEDS-DEVICE")


def test_emit_made_for_another_base_or_head_is_refused(mod, capsys, tmp_path):
    # catches: a previous round's emit.json folded into this round's report
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, KDP: KDP_JSON})
    old = commit(repo, {BUILDER: BODY_EDIT})
    head = commit(repo, {BUILDER: BODY_EDIT.replace("x = 1", "x = 2")})
    stale = write_emit(tmp_path / "stale.json", repo, base, old)
    argv = [
        "report",
        "--repo",
        str(repo),
        "--base",
        base,
        "--head",
        head,
        "--emit",
        stale,
    ]
    assert mod.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("impact: ") and "Runtime impact" not in captured.out


@pytest.mark.parametrize(
    "spoil",
    [
        lambda d: d.update(schema=3),
        lambda d: d.update(not_inspected={}),
        lambda d: d["families"][0].update(failures=[]),
        lambda d: d["families"][0].pop("variants"),
        lambda d: d["families"][0].update(params={"base": None, "head": "q_ptr"}),
        lambda d: d["families"][0].pop("companions"),
        lambda d: d["families"][0].update(
            companions={"base": None, "head": {"reduce": None}}
        ),
    ],
    ids=[
        "schema-3-without-companions",
        "not-inspected-dict",
        "failures-list",
        "no-variants",
        "params-str",
        "no-companions",
        "companion-without-params",
    ],
)
def test_malformed_emit_exits_one_instead_of_folding_or_crashing(
    mod, capsys, tmp_path, spoil
):
    # catches: a malformed emit crashing with a traceback, or folding on missing fields
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, KDP: KDP_JSON})
    head = commit(repo, {BUILDER: BODY_EDIT})
    path = Path(write_emit(tmp_path / "e.json", repo, base, head))
    doc = json.loads(path.read_text(encoding="utf-8"))
    spoil(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    argv = [
        "report",
        "--repo",
        str(repo),
        "--base",
        base,
        "--head",
        head,
        "--emit",
        str(path),
    ]
    assert mod.main(argv) == 1
    assert capsys.readouterr().err.startswith("impact: ")


def test_emit_with_no_families_proves_nothing(mod, capsys, tmp_path):
    # catches: an empty family list read as "every shipped variant equal"
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, KDP: KDP_JSON})
    head = commit(repo, {BUILDER: BODY_EDIT})
    emit = write_emit(tmp_path / "none.json", repo, base, head, families=[])
    data = report(mod, capsys, repo, base, head, "--emit", emit)
    assert data["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"


@pytest.mark.parametrize("key", ["only_in_base", "only_in_head"])
def test_emit_with_an_added_or_removed_variant_does_not_fold(
    mod, capsys, tmp_path, key
):
    # catches: a variant-set change ignored when deciding every shipped variant is equal
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, KDP: KDP_JSON})
    head = commit(repo, {BUILDER: BODY_EDIT})
    emit = write_emit(tmp_path / "e.json", repo, base, head, **{key: ["v2"]})
    data = report(mod, capsys, repo, base, head, "--emit", emit)
    assert data["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"


@pytest.mark.parametrize(
    "fam",
    [
        family(arch="gfx942", source="kernels/gfx942/other.py", builder="build_other"),
        family(variants={"base": 0, "head": 0}),
    ],
    ids=["unrelated-source", "no-variants-built"],
)
def test_emit_folds_only_rows_a_built_family_imports(mod, capsys, tmp_path, fam):
    # catches: one quiet family, or an empty one, folding every shipped-module row
    other = "def build_other(b, spec):\n    return b\n"
    repo, base = make_repo(
        tmp_path, {BUILDER: BUILD, KDP: KDP_JSON, OTHER_KERNEL: other}
    )
    head = commit(repo, {BUILDER: BODY_EDIT})
    emit = write_emit(tmp_path / "e.json", repo, base, head, families=[fam])
    data = report(mod, capsys, repo, base, head, "--emit", emit)
    assert data["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"


def test_declared_static_list_reordered_against_builder_is_mismatch(
    mod, capsys, tmp_path
):
    # catches: the declared-pair check comparing names as a set rather than in order
    repo, base = make_repo(
        tmp_path, {BUILDER: BUILD, PACK: PACK_STATIC, MANIFEST: MANIFEST_YAML}
    )
    swapped = PACK_STATIC.replace('"q_ptr"', "@@").replace('"k_ptr"', '"q_ptr"').replace("@@", '"k_ptr"')  # fmt: skip
    head = commit(repo, {PACK: swapped})
    data = report(mod, capsys, repo, base, head)
    assert [r["path"] for r in rows_of(data, "kernargs", "mismatch")] == [PACK]


@pytest.mark.parametrize(
    "names", [["scale", "stride"], ["q_ptr", "k_ptr", "a", "b", "c", "d"]]
)
def test_unrelated_static_list_in_the_same_pack_is_not_a_mismatch(
    mod, capsys, tmp_path, names
):
    # catches: `related` loosened to one shared name, or without its half-overlap rule
    pack = PACK_STATIC + "\n" + sig("otherSignature", names)
    repo, base = make_repo(
        tmp_path, {BUILDER: BUILD, PACK: pack, MANIFEST: MANIFEST_YAML}
    )
    head = commit(repo, {BUILDER: BODY_EDIT})
    data = report(mod, capsys, repo, base, head)
    assert not rows_of(data, status="mismatch")
    assert data["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"


def test_type_only_builder_change_against_declared_static_list_is_mismatch(
    mod, capsys, tmp_path
):
    # catches: a declared pair compared by names only and reported as agreeing
    repo, base = abi_repo(tmp_path, {PACK: PACK_STATIC, MANIFEST: MANIFEST_YAML})
    head = commit(repo, {BUILDER: BUILD.replace('"scale", F32', '"scale", I64')})
    data = report(mod, capsys, repo, base, head)
    (row,) = rows_of(data, "kernargs", "mismatch")
    assert row["path"] == PACK and "scale" in row["reason"]
    assert data["rollup"] == "Runtime impact: BREAKS"


@pytest.mark.parametrize(
    "before, after",
    [
        ("SC = F32\n\n\n" + BUILD.replace('"scale", F32', '"scale", SC'),
         "SC = I64\n\n\n" + BUILD.replace('"scale", F32', '"scale", SC')),
        (BUILD, BUILD.replace('    b.param("scale", F32)\n',
                              '    if spec.scaled:\n        b.param("scale", F32)\n')),
    ],
    ids=["type-in-a-variable", "parameter-made-conditional"],
)  # fmt: skip
def test_builder_list_change_with_the_same_names_is_a_kernargs_row(
    mod, capsys, tmp_path, before, after
):
    # catches: a type held in a variable, or a change in which params are conditional, unseen
    repo, base = make_repo(tmp_path, {BUILDER: before})
    head = commit(repo, {BUILDER: after})
    rows = rows_of(report(mod, capsys, repo, base, head), "kernargs")
    assert [(r["name"], r["status"]) for r in rows] == [("build_attn", "changed")]


def test_declared_mirror_with_no_comparable_list_stays_visible(mod, capsys, tmp_path):
    # catches: a declared C++ list sharing fewer than two names skipped without a row
    pack = sig("staleSignature", ["scale", "stride"])
    repo, base = make_repo(
        tmp_path, {BUILDER: BUILD, PACK: pack, MANIFEST: MANIFEST_YAML}
    )
    head = commit(repo, {BUILDER: BODY_EDIT})
    rows = [r for r in rows_of(report(mod, capsys, repo, base, head), "kernargs") if r["path"] == PACK]  # fmt: skip
    assert [r["status"] for r in rows] == ["changed"]


def manifest(*surfaces, source=None) -> str:
    """An engine manifest with `(name, python_source, cpp_mirror)` surfaces and, when given, the
    `(source, builder)` the packer compiles."""
    lines = ["arch: gfx950", f"provider_root: {HKP.rstrip('/')}"]
    if source:
        lines += [f"source: {source[0]}", f"builder: {source[1]}"]
    lines.append("launch_surface:")
    for name, locator, mirror in surfaces:
        lines += [f"  - name: {name}", f"    python_source: {locator}", f"    cpp_mirror: {mirror}",
                  "    kmd_fields: []", "    guard: none", "    test: none"]  # fmt: skip
    return "\n".join(lines) + "\n"


def ptr_sig(name, names):
    """A static `KernelArgument` list function holding `names`, all pointers."""
    items = ",\n            ".join(f'KernelArgument{{"global_buffer", PTR, {8 * i}, "{n}"}}' for i, n in enumerate(names))  # fmt: skip
    return f"std::vector<KernelArgument> {name}()\n{{\n    return {{{items}}};\n}}\n"


GEOM = HKP + "src/engines/kernel_ingestor_engine/packs/Gfx950Geometry.hpp"
WITH_METHOD = (
    BUILD
    + "\n\nclass Tuning:\n    def launch_block(self):\n        return (64, 1, 1)\n"
)
TWO_METHODS = (
    WITH_METHOD
    + "\n\nclass Other:\n    def launch_block(self):\n        return (32, 1, 1)\n"
)
UNDER_TRY = (
    BUILD
    + "\n\ntry:\n    def launch_block():\n        return (64, 1, 1)\nexcept ImportError:\n    pass\n"
)


@pytest.mark.parametrize(
    "before, after, status",
    [(WITH_METHOD, WITH_METHOD.replace("(64, 1, 1)", "(128, 1, 1)"), "changed"),
     (WITH_METHOD, WITH_METHOD.replace("def launch_block", "def launch_blk"), "mismatch"),
     (TWO_METHODS, TWO_METHODS.replace("(32, 1, 1)", "(16, 1, 1)"), "changed"),
     (UNDER_TRY, UNDER_TRY.replace("(64, 1, 1)", "(128, 1, 1)"), "changed")],
    ids=["method-edited", "method-renamed", "second-same-named-method-edited",
         "def-under-module-level-try-edited"],
)  # fmt: skip
def test_launch_surface_naming_a_method_resolves_like_launch_surface_check(
    mod, capsys, tmp_path, before, after, status
):
    # catches: a python_source symbol looked up only at module level or only through def and
    # class bodies, so a method or a def under `try:` reads "missing" or goes unseen; only the
    # first of several same-named defs compared; or a renamed method no longer reported missing
    block = manifest(("block", "kernels/gfx950/attn.py:launch_block", GEOM))
    repo, base = make_repo(
        tmp_path, {BUILDER: before, GEOM: "int block();\n", MANIFEST: block}
    )
    head = commit(repo, {BUILDER: after})
    rows = rows_of(report(mod, capsys, repo, base, head), "surfaces")
    assert [(r["name"], r["status"]) for r in rows] == [("block:launch_block", status)]


SIG_FACTORY = HKP + "rocke/library/kernels/common/sig.py"
FACTORY = 'def attn_signature(dtype):\n    return SignatureBuilder().ptr("q_ptr", dtype).build()\n'
FACTORY_MANIFEST = manifest(
    ("kernargs", "kernels/common/sig.py:attn_signature", PACK),
    source=("kernels/gfx950/attn.py", "build_attn"),
)


@pytest.mark.parametrize(
    "edit, status, says",
    [({PACK: PACK_STATIC + "int unrelated = 1;\n"}, "changed", "the manifest declares attn_signature"),
     ({PACK: PACK_STATIC.replace('"q_ptr"', "@@").replace('"k_ptr"', '"q_ptr"').replace("@@", '"k_ptr"')},
      "mismatch", "disagrees with builder build_attn"),
     ({SIG_FACTORY: FACTORY.replace("def attn_signature", "def attn_sig")},
      "mismatch", "function attn_signature"),
     ({BUILDER: BODY_EDIT}, "changed", "the manifest declares attn_signature"),
     ({BUILDER: BUILD.replace('"scale", F32', '"scale", F64')}, "mismatch", "types do not")],
    ids=["mirror-edited", "mirror-reordered", "factory-renamed", "source-builder-edited",
         "source-builder-type-changed"],
)  # fmt: skip
def test_kernargs_pair_declaring_a_signature_factory_compares_the_manifest_builder(
    mod, capsys, tmp_path, edit, status, says
):
    # catches: a declared function with no `b.param` list reported as a missing builder, so a
    # SignatureBuilder factory reads BREAKS; the fallback hiding a real disagreement; or an edit
    # to the manifest's `source` builder alone not checking the pair
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, SIG_FACTORY: FACTORY, PACK: PACK_STATIC,
                                      MANIFEST: FACTORY_MANIFEST})  # fmt: skip
    head = commit(repo, edit)
    rows = [r for r in rows_of(report(mod, capsys, repo, base, head), "kernargs") if r["path"] == PACK]  # fmt: skip
    assert [r["status"] for r in rows] == [status]
    assert says in rows[0]["reason"]


TWO_LAUNCHES = """\
def build_seg(b, spec):
    b.param("part_ptr", PTR)
    b.param("q_ptr", PTR)
    b.param("lens_ptr", PTR)
    b.param("scale", F32)
    return b


def build_reduce(b, spec):
    b.param("out_ptr", PTR)
    b.param("part_ptr", PTR)
    b.param("lens_ptr", PTR)
    return b
"""
SEG_SIG = ptr_sig("segSignature", ["part_ptr", "q_ptr", "lens_ptr"]).replace(
    '"lens_ptr"}',
    '"lens_ptr"},\n            KernelArgument{"by_value", F32, 24, "scale"}',
)


@pytest.mark.parametrize(
    "reduce_names, status",
    [(["out_ptr", "part_ptr", "lens_ptr"], "changed"), (["out_ptr", "lens_ptr"], "mismatch")],
    ids=["reduce-list-agrees", "reduce-list-drops-a-param"],
)  # fmt: skip
def test_multi_launch_pack_compares_each_list_with_its_own_builder(
    mod, capsys, tmp_path, reduce_names, status
):
    # catches: every static list of a multi-launch pack compared with the one declared builder,
    # so the reduce list reads as disagreeing with the segment builder; or a stale reduce list
    # no longer caught
    seg = manifest(("kernargs", "kernels/gfx950/attn.py:build_seg", PACK))
    pack = SEG_SIG + "\n" + ptr_sig("reduceSignature", reduce_names)
    repo, base = make_repo(tmp_path, {BUILDER: TWO_LAUNCHES, PACK: pack, MANIFEST: seg})
    head = commit(
        repo, {BUILDER: TWO_LAUNCHES.replace("return b", "x = 1\n    return b")}
    )
    rows = rows_of(report(mod, capsys, repo, base, head), "kernargs")
    by_func = {r["name"]: r for r in rows if r["path"] == PACK}
    assert by_func["declared:segSignature"]["status"] == "changed"
    reduce = by_func["declared:reduceSignature"]
    assert reduce["status"] == status and "build_reduce" in reduce["reason"]


@pytest.mark.parametrize(
    "added, status",
    [('    if spec.lse:\n        b.param("lse_ptr", PTR)\n', "changed"),
     ('    b.param("lse_ptr", PTR)\n', "mismatch")],
    ids=["conditional-param-added", "unconditional-param-added"],
)  # fmt: skip
def test_copy_without_a_newly_added_conditional_param_is_not_a_mismatch(
    mod, capsys, tmp_path, added, status
):
    # catches: a copy of the list without an optional param, which variants that do not enable
    # it still pass, read as a break; or a copy missing a required new param read as fine
    repo, base = abi_repo(tmp_path)
    head = commit(
        repo, {BUILDER: BUILD.replace("    return b", added + "    return b")}
    )
    (row,) = rows_of(report(mod, capsys, repo, base, head), "kernargs")
    assert row["status"] == status and row["counterparts"][0]["path"] == PIN


@pytest.mark.parametrize(
    "reduce_params, status",
    [([["out_ptr", "ptr"], ["part_ptr", "ptr"], ["lens_ptr", "ptr"]], None),
     ([["out_ptr", "ptr"], ["part_ptr", "ptr"], ["lens_ptr", "ptr"], ["lse_ptr", "ptr"]], "mismatch")],
    ids=["companion-agrees", "companion-gains-a-param"],
)  # fmt: skip
def test_emit_fold_compares_each_companion_launch_with_its_static_list(
    mod, capsys, tmp_path, reduce_params, status
):
    # catches: only a KernelModule's primary launch compared with the pack, so a companion's
    # changed kernargs read as consistent; or a pack found only by the arch-named fallback,
    # not through the manifest's `source` builder that the family names
    # no arch in the pack's name, so only the manifest's builder can pair it
    pack = HKP + "src/engines/kernel_ingestor_engine/packs/AttnNative.cpp"
    lists = (
        PACK_STATIC
        + "\n"
        + ptr_sig("reduceSignature", ["out_ptr", "part_ptr", "lens_ptr"])
    )
    declared = FACTORY_MANIFEST.replace(PACK, pack)
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, KDP: KDP_JSON, SIG_FACTORY: FACTORY,
                                      pack: lists, MANIFEST: declared})  # fmt: skip
    head = commit(repo, {BUILDER: BODY_EDIT})
    primary = [["q_ptr", "ptr"], ["k_ptr", "ptr"], ["scale", "float"]]
    emit = write_emit(tmp_path / "m.json", repo, base, head, params_changed=["v"],
                      params={"base": None, "head": primary},
                      companions={"base": None, "head": {"reduce": reduce_params}})  # fmt: skip
    data = report(mod, capsys, repo, base, head, "--emit", emit)
    bad = rows_of(data, "kernargs", "mismatch")
    assert [r["status"] for r in bad] == ([status] if status else [])
    if status:
        assert "build_attn[reduce]" in bad[0]["reason"] and bad[0]["path"] == pack


@pytest.mark.parametrize(
    "before, after",
    [
        ("// note\nint x = 1;\n", "// note \\\nint x = 1;\n"),
        ('auto s = u8R"(a"b // c)";\n', 'auto s = u8R"(a"b // d)";\n'),
        ('auto s = R"(a\n  b)";\n', 'auto s = R"(a\n b)";\n'),
    ],
    ids=["comment-splices-next-line", "prefixed-raw-string", "multi-line-raw-string"],
)
def test_cpp_edits_behind_splices_and_raw_strings_are_code(
    mod, capsys, tmp_path, before, after
):
    # catches: a `//` comment ended before a `\`-newline, or raw strings read as plain strings
    repo, base = make_repo(tmp_path, {PACK: before})
    head = commit(repo, {PACK: after})
    data = report(mod, capsys, repo, base, head, cmd="files")
    assert file_class(data, PACK) == "code"


def test_dependency_pin_is_left_to_deps_and_named_in_the_rollup(mod, capsys, tmp_path):
    # catches: requirements.txt read as documentation, so a pin bump reads as plain N/A
    req = HKP + "rocke/platform/requirements.txt"
    repo, base = make_repo(tmp_path, {req: "numpy==1.26\n"})
    head = commit(repo, {req: "numpy==2.0\n"})
    data = report(mod, capsys, repo, base, head)
    (row,) = data["rows"]
    assert row["status"] == "outside" and "DEPS" in row["reason"]
    assert data["rollup"].startswith("Runtime impact: N/A") and "DEPS" in data["rollup"]


def test_method_removed_from_a_base_read_only_through_a_subclass_is_changed(
    mod, capsys, tmp_path
):
    # catches: a removed method whose readers never name its class reported as no impact, or
    # any same-named attribute read anywhere (a dict's `.helper`) taken for a reader
    files = {
        HKP_PACK + "base.py": "class Base:\n    def helper(self):\n        return 1\n",
        HKP_PACK + "sub.py": "from .base import Base\n\n\nclass Sub(Base):\n    pass\n",
        HKP_PACK + "cfg.py": "def pick(d):\n    return d.helper()\n",
        READER: "from hkp_pack.sub import Sub\n\n\ndef test_it():\n    assert Sub().helper()\n",
        HKP
        + "descriptor-packaging/tests/test_dotted.py": (
            "import hkp_pack.sub\n\n\ndef test_it():\n    assert hkp_pack.sub.Sub().helper()\n"
        ),
    }
    repo, base = make_repo(tmp_path, files)
    head = commit(repo, {HKP_PACK + "base.py": "class Base:\n    pass\n"})
    (row,) = rows_of(report(mod, capsys, repo, base, head), "symbols")
    assert (row["name"], row["status"]) == ("Base.helper", "changed")
    assert {c["path"] for c in row["counterparts"]} == {
        READER,
        HKP + "descriptor-packaging/tests/test_dotted.py",
    }


def test_conflict_only_in_a_test_still_marks_the_merged_view_changed(
    mod, capsys, tmp_path
):
    # catches: a conflicted test counted as outside, though it is a searched copy
    repo, base = make_repo(tmp_path / "a", {BUILDER: BUILD, PIN: PIN_TEST})
    head = branch_commit(repo, base, {BUILDER: BODY_EDIT, PIN: PIN_TEST + "# head\n"}, "feature")  # fmt: skip
    tip = branch_commit(repo, base, {PIN: PIN_TEST + "# tip\n"}, "target")
    data = report(mod, capsys, repo, base, head, "--target-tip", tip)
    assert (
        next(v for v in data["views"] if v["label"] == "merged")["status"]
        == "conflicted"
    )
    assert any(r["view"] == "merged" and r["status"] == "changed" for r in data["rows"])
    # a docs-only change conflicting in docs still has no surface file
    doc = "projects/hipdnn/docs/x.md"
    repo, base = make_repo(tmp_path / "b", {doc: "a\n"})
    head = branch_commit(repo, base, {doc: "b\n"}, "feature")
    tip = branch_commit(repo, base, {doc: "c\n"}, "target")
    data = report(mod, capsys, repo, base, head, "--target-tip", tip)
    assert data["rollup"] == "Runtime impact: N/A (no surface file)"


def test_unrelated_target_history_skips_the_merged_view_as_changed(
    mod, capsys, tmp_path
):
    # catches: a merge failure other than a conflict downgraded to outside
    repo, base = make_repo(tmp_path, {BUILDER: BUILD})
    head = commit(repo, {BUILDER: BODY_EDIT})
    _COUNTER.pop(repo, None)
    tip = commit(repo, {"README.md": "orphan\n"}, parent=None)
    data = report(mod, capsys, repo, base, head, "--target-tip", tip)
    assert (
        next(v for v in data["views"] if v["label"] == "merged")["status"] == "skipped"
    )
    assert any(r["view"] == "merged" and r["status"] == "changed" for r in data["rows"])


def test_non_ascii_path_is_reported_under_its_real_name(mod, capsys, tmp_path):
    # catches: git quoting non-ASCII paths (core.quotepath) so rows name a mangled path
    odd = HKP + "rocke/library/kernels/gfx950/attn_\u00e9.py"
    repo, base = make_repo(tmp_path, {odd: "def f():\n    return 1\n", KDP: KDP_JSON})
    head = commit(repo, {odd: "def f():\n    return 2\n"})
    data = report(mod, capsys, repo, base, head)
    assert odd in {r["path"] for r in data["rows"]}


ROCKE_PY = HKP + "rocke/platform/python/rocke/"
PACKER_PY = HKP_PACK + "rocke_compile.py"


def test_module_the_packer_reaches_ships_on_every_arch(mod, capsys, tmp_path):
    # catches: a module the packer compiles every variant through, imported by name from its
    # package (`from rocke.core import K`), reported only for the arch whose builder also
    # imports it, so the device run named misses the others
    shared = ROCKE_PY + "core/ir.py"
    files = {
        BUILDER: "from rocke.core.ir import K\n" + BUILD,
        KDP: KDP_JSON,
        shared: "K = 1\n",
        ROCKE_PY + "core/__init__.py": "from .ir import K\n",
        PACKER_PY: "def load():\n    from rocke.core import K\n    return K\n",
    }
    repo, base = make_repo(tmp_path, files)
    head = commit(repo, {shared: "K = 2\n"})
    data = report(mod, capsys, repo, base, head)
    (row,) = [r for r in rows_of(data, "file") if r["path"] == shared]
    assert (row["status"], row["arch"]) == ("changed", "all-ingestor-archs")
    assert data["rollup"] == "Runtime impact: NEEDS-DEVICE all-ingestor-archs"


def test_body_edit_of_a_helper_a_builder_imports_from_its_package_is_changed(
    mod, capsys, tmp_path
):
    # catches: `from rocke.helpers import f` resolved to the package __init__ only, so a body
    # edit of the module that defines f, which the shipped builder runs, reads inert
    helper = ROCKE_PY + "helpers/x.py"
    files = {
        BUILDER: "from rocke.helpers import f\n" + BUILD,
        KDP: KDP_JSON,
        ROCKE_PY + "helpers/__init__.py": "from .x import f\n",
        helper: "def f():\n    return 1\n",
    }
    repo, base = make_repo(tmp_path, files)
    head = commit(repo, {helper: "def f():\n    return 2\n"})
    data = report(mod, capsys, repo, base, head)
    (row,) = [r for r in rows_of(data, "file") if r["path"] == helper]
    assert (row["status"], row["arch"]) == ("changed", "gfx950")


def test_compile_step_edit_never_folds_on_equal_ir(mod, capsys, tmp_path):
    # catches: a COMGR option change (-O3 to -O2) folded inert because the IR is equal
    step = ROCKE_PY + "helpers/compile.py"
    opts = 'def _comgr_options_for_kernel(k):\n    return ["-O3"]\n'
    files = {
        BUILDER: "from rocke.helpers.compile import _comgr_options_for_kernel\n"
        + BUILD,
        KDP: KDP_JSON,
        step: opts,
    }
    repo, base = make_repo(tmp_path, files)
    head = commit(repo, {step: opts.replace("-O3", "-O2")})
    emit = write_emit(tmp_path / "eq.json", repo, base, head)
    data = report(mod, capsys, repo, base, head, "--emit", emit)
    (row,) = [r for r in rows_of(data, "file") if r["path"] == step]
    assert row["status"] == "changed" and "after lowering" in row["reason"]
    assert data["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"


def test_emit_comgr_inputs_changed_keep_shipped_rows_changed(mod, capsys, tmp_path):
    # catches: a builder attribute that changes only the COMGR options or the fresh-process
    # compile (the IR is equal) folded inert
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, KDP: KDP_JSON})
    head = commit(repo, {BUILDER: BODY_EDIT})
    emit = write_emit(tmp_path / "c.json", repo, base, head, comgr_changed=["v"])
    data = report(mod, capsys, repo, base, head, "--emit", emit)
    assert data["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"
    assert [r["name"] for r in rows_of(data, "kernargs")] == [
        "emit-comgr:gfx950:build_attn"
    ]
    assert [r["status"] for r in rows_of(data, "file")] == ["changed"]


def test_reexport_chain_past_three_packages_still_ships_the_defining_module(
    mod, capsys, tmp_path
):
    # catches: a depth cap or a memoised dead end in the re-export walk, so `from rocke.a
    # import f` stops short of the module defining f and a body edit of it reads inert
    chain = "abcd"
    files = {
        ROCKE_PY + f"{p}/__init__.py": f"from rocke.{q} import f\n"
        for p, q in zip(chain, chain[1:])
    }
    files[ROCKE_PY + "d/__init__.py"] = "from rocke.d.e import f\n"
    defining = ROCKE_PY + "d/e.py"
    files[defining] = "def f():\n    return 1\n"
    files[HKP_PACK + "a1.py"] = "from rocke.a import f\n"
    files[HKP_PACK + "a2.py"] = "from rocke.d import f\n"
    repo, base = make_repo(tmp_path, files)
    head = commit(repo, {defining: "def f():\n    return 2\n"})
    data = report(mod, capsys, repo, base, head)
    assert [(r["path"], r["status"]) for r in rows_of(data, "file")] == [
        (defining, "changed")
    ]


ENGINE_SRC = HKP + "rocke/platform/cpp/core/a.cpp"
PROVIDER_CMAKE = HKP + "src/CMakeLists.txt"


@pytest.mark.parametrize(
    "cmake, pin, status",
    [
        ("add_library(p x.cpp)\n# rocke_core is not linked\n", '_BACKEND = "python"\n', "inert"),
        ("target_link_libraries(p PRIVATE rocke_core)\n", '_BACKEND = "python"\n', "changed"),
        ("add_library(p SHARED $<TARGET_OBJECTS:rocke_core_objects>)\n", '_BACKEND = "python"\n', "changed"),
        ("add_library(p x.cpp)\n", '_BACKEND = "cpp"\n', "changed"),
    ],
    ids=["unlinked-python-pin", "linked", "engine-objects", "cpp-pin"],
)  # fmt: skip
def test_rocke_cpp_engine_source_is_inert_only_while_nothing_ships_it(
    mod, capsys, tmp_path, cmake, pin, status
):
    # catches: every rocKE C++ engine edit read as NEEDS-DEVICE on every arch though no shipped
    # kernel is built by the engine; and the reverse, an engine the provider links or the
    # packer lowers with read as inert
    files = {
        ENGINE_SRC: "int f() { return 1; }\n",
        PROVIDER_CMAKE: cmake,
        PACKER_PY: pin,
    }
    repo, base = make_repo(tmp_path, files)
    head = commit(repo, {ENGINE_SRC: "int f() { return 2; }\n"})
    data = report(mod, capsys, repo, base, head)
    assert {r["path"]: r["status"] for r in rows_of(data, "file")} == {
        ENGINE_SRC: status
    }


def test_ci_test_category_list_is_left_to_build(mod, capsys, tmp_path):
    # catches: a CI test-category list read as provider code, so enrolling a test in a CI
    # tier reads as NEEDS-DEVICE on every arch
    cats = HKP + "ROCKE_ENGINE_test_categories_external.yaml"
    repo, base = make_repo(tmp_path, {cats: "quick:\n  - a\n"})
    head = commit(repo, {cats: "quick:\n  - a\n  - b\n"})
    data = report(mod, capsys, repo, base, head)
    (row,) = data["rows"]
    assert row["status"] == "outside" and "BUILD" in row["reason"]
    assert (
        data["rollup"].startswith("Runtime impact: N/A") and "BUILD" in data["rollup"]
    )


def test_changed_dvc_pointer_is_not_folded_by_a_regenerated_catalog(
    mod, capsys, tmp_path
):
    # catches: a new DVC-held catalog read as equal because the enumerator regenerates the
    # same variant list from source, while the file's other fields went unread
    ptr = KDP.rsplit("/", 1)[0] + "/artifacts.dvc"
    pointer = "outs:\n- md5: {}.dir\n  path: artifacts\n"
    repo, base = make_repo(tmp_path, {BUILDER: BUILD, ptr: pointer.format("a" * 32)})
    head = commit(repo, {ptr: pointer.format("b" * 32)})
    emit = write_emit(
        tmp_path / "e.json", repo, base, head, spec_origin="registry_enumerator"
    )
    data = report(mod, capsys, repo, base, head, "--emit", emit)
    assert [(r["path"], r["status"]) for r in rows_of(data, "data")] == [
        (ptr, "changed")
    ]
    assert data["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"
