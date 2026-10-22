# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the hipdnn-deep-review impact.py helper (kernel ABI and dispatch facts of a diff).

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
    / "hipdnn-deep-review"
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
    spec = importlib.util.spec_from_file_location("hipdnn_deep_review_impact", SCRIPT)
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
    # --round-dir takes the same tip from PIN.md's TARGET_TIP
    rd = tmp_path / "round"
    rd.mkdir()
    (rd / "PIN.md").write_text(
        f"CLONE: {repo}\nBASE: {base}\nHEAD: {head}\nTARGET_TIP: {tip}\n",
        encoding="utf-8",
    )
    assert mod.main(["report", "--round-dir", str(rd), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["rollup"] == "Runtime impact: BREAKS"


def test_merge_conflict_skips_merged_view_with_reason(mod, capsys, tmp_path):
    # catches: a crash on conflict, or a clean NONE when the race cannot be excluded
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
        merged["status"] == "skipped"
        and "conflict" in merged["reason"]
        and BUILDER in merged["reason"]
    )
    assert data["rollup"] != "Runtime impact: NONE (1 files inspected)"
    assert any(r["view"] == "merged" and r["status"] == "changed" for r in data["rows"])


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
    # catches: whitespace deletion instead of collapse ("attn fwd" == "attnfwd")
    repo, base = make_repo(
        tmp_path, {PACK: 'const char* kName = "attn fwd";\nint f() { return 1; }\n'}
    )
    head = commit(
        repo, {PACK: 'const char* kName = "attnfwd";\nint f() { return 1; }\n'}
    )
    assert (
        file_class(report(mod, capsys, repo, base, head, cmd="files"), PACK) == "code"
    )
    spaced = commit(
        repo, {PACK: 'const char* kName = "attnfwd";\nint   f()  {  return 1;   }\n'}
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
    assert (
        row["status"] == "changed"
        and row["reason"] == "override removed; base Mystery not resolved"
    )


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
    assert captured.err.strip() == "impact: empty diff" and "NONE" not in captured.out


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


def write_emit(path, **family):
    base = dict(arch="gfx950", source=BUILDER, builder="build_attn", kdp=KDP, spec_origin="kdp-in-git",
                variants={"base": 1, "head": 1}, only_in_base=[], only_in_head=[], params_changed=[],
                ir_changed=[], params={"base": None, "head": None}, failures={"base": {}, "head": {}})  # fmt: skip
    base.update(family)
    doc = {"schema": 1, "tool": "impact_emit", "flavor": "llvm23", "base": "b", "head": "h",
           "families": [base], "not_inspected": [], "seconds": 1.0}  # fmt: skip
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
        mod, capsys, repo, base, head, "--emit", write_emit(tmp_path / "eq.json")
    )
    assert equal["rollup"] == "Runtime impact: NONE (1 files inspected)"
    changed = report(
        mod,
        capsys,
        repo,
        base,
        head,
        "--emit",
        write_emit(tmp_path / "ir.json", ir_changed=["v"]),
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
                  write_emit(tmp_path / "p.json", params_changed=["v"], params={"base": None, "head": wider}))  # fmt: skip
    (row,) = rows_of(data, "kernargs", "mismatch")
    assert row["path"] == PACK and "batch" in row["reason"]
    assert data["rollup"] == "Runtime impact: BREAKS"
    # the same params as the static list: consistent, so changed rather than mismatch
    same = [["q_ptr", "ptr"], ["k_ptr", "ptr"], ["scale", "float"]]
    data = report(mod, capsys, repo, base, head, "--emit",
                  write_emit(tmp_path / "s.json", params_changed=["v"], params={"base": None, "head": same}))  # fmt: skip
    assert not rows_of(data, status="mismatch")
    assert data["rollup"] == "Runtime impact: NEEDS-DEVICE gfx950"
    # a static mismatch is never folded away by an otherwise equal emit
    repo2, base2 = make_repo(
        tmp_path / "keep", {BUILDER: BUILD, PIN: PIN_TEST, KDP: KDP_JSON}
    )
    head2 = commit(repo2, {BUILDER: BUILD_REORDERED})
    data = report(
        mod, capsys, repo2, base2, head2, "--emit", write_emit(tmp_path / "e2.json")
    )
    assert data["rollup"] == "Runtime impact: BREAKS"
