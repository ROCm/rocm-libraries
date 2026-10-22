# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the hipdnn-deep-review impact_emit.py script.

A tiny fake provider tree (fake `hkp_pack` and `rocke` packages with the helper names
the child resolves, a builder module, a KDP) lives in a temporary git repo; the lowering
returns deterministic IR text, so no rocKE, comgr or GPU is involved.
"""

import importlib.util
import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "hipdnn-deep-review"
    / "scripts"
    / "impact_emit.py"
)

PROVIDER = "dnn-providers/hip-kernel-provider"
DESC = f"{PROVIDER}/src/engines/kernel_ingestor_engine/descriptors/rocKE/fake"
BUILDER = f"{PROVIDER}/rocke/library/kernels/gfx_fake/fake_attn.py"

HELPERS = """
import dataclasses
def _module_from_source(source):
    return source[:-3].replace("/", ".") if source.endswith(".py") else source.replace("/", ".")
def _resolve_spec_class(module, fn):
    return module.Spec
def build_spec(cls, data):
    return cls(**data)
"""

COMPILE = """
import os
def _lower_llvm_via_backend(kd, *, arch, backend, spec):
    assert backend == "python"
    args = ", ".join(f"{t} noalias %{n}" if t.startswith("ptr") else f"{t} %{n}" for n, t in kd["params"])
    return (f"; flavor={os.environ.get('ROCKE_LLVM_FLAVOR')} arch={arch}\\n; body={kd['body']}\\n"
            f"define amdgpu_kernel void @fake({args}) #0 {{\\n  ret void\\n}}\\n")
"""

LOWER_LLVM = """
LLVM_FLAVOR_LLVM20 = "llvm20"
LLVM_FLAVOR_LLVM23 = "llvm23"
_LLVM_FLAVOR_LADDER: tuple = (
    (0, LLVM_FLAVOR_LLVM20),
    (23, LLVM_FLAVOR_LLVM23),
)
"""


def builder_src(params=(), body="v1"):
    ps = "".join(f'    p.append(("{n}", "{t}"))\n' for n, t in params)
    return (
        "import dataclasses\n"
        "@dataclasses.dataclass\n"
        "class Spec:\n    n: int = 1\n    boom: bool = False\n\n"
        "def build_fake(spec, arch):\n"
        "    if spec.boom:\n        raise ValueError('boom')\n"
        '    p = [("q", "ptr addrspace(1)"), ("n", "i32")]\n'
        f"{ps}"
        f'    return {{"params": p, "body": "{body}"}}\n'
    )


KDP_VARIANTS = ({"name": "v_a", "spec": {"n": 1}}, {"name": "v_b", "spec": {"n": 2}})


def kdp_json(variants=KDP_VARIANTS, arch=("gfx950",), target=None):
    return json.dumps(
        {
            "arch": list(arch),
            "kernelDescriptors": [
                {
                    "name": v["name"],
                    **({"arch": v["arch"]} if "arch" in v else {}),
                    "kernel_source": {
                        "kind": "rocke",
                        "source": "kernels/gfx_fake/fake_attn.py",
                        "builder": "build_fake",
                        "spec": v["spec"],
                        **({"target": target} if target else {}),
                    },
                }
                for v in variants
            ],
        }
    )


@pytest.fixture
def emit():
    spec = importlib.util.spec_from_file_location(
        "hipdnn_deep_review_impact_emit", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERIC_HELPERS = (
    HELPERS
    + """
def rocke_compile_arch(arch, target="native"):
    if target == "native":
        return arch
    return {"gfx11": "gfx11-generic", "gfx12": "gfx12-generic"}[arch[:5]]
"""
)


def tree_files(
    params=(), body="v1", variants=KDP_VARIANTS, arch=("gfx950",), target=None
):
    """Every file of the fake provider tree, as {repo path: text}."""
    py = f"{PROVIDER}/rocke/platform/python"
    files = {
        f"{py}/rocke/__init__.py": "",
        f"{py}/rocke/helpers/__init__.py": "",
        f"{py}/rocke/core/__init__.py": "",
        f"{PROVIDER}/rocke/library/kernels/__init__.py": "",
        f"{PROVIDER}/rocke/library/kernels/gfx_fake/__init__.py": "",
        f"{PROVIDER}/descriptor-packaging/python/hkp_pack/__init__.py": "",
        f"{py}/rocke/helpers/compile.py": COMPILE,
        f"{py}/rocke/core/lower_llvm.py": LOWER_LLVM,
        f"{PROVIDER}/descriptor-packaging/python/hkp_pack/rocke_compile.py": HELPERS,
        BUILDER: builder_src(params, body),
        f"{DESC}/fake.kdp.json": kdp_json(variants, arch, target),
    }
    return files


GEN = "projects/hipdnn/tools/IngestorGenerator"
ENUMERATOR = textwrap.dedent(
    """
    import argparse, json, os
    p = argparse.ArgumentParser(); p.add_argument("--manifest"); p.add_argument("--out")
    a = p.parse_args()
    assert os.environ["ROCKE_PROVIDER_ROOT"]
    ks = {"source": "kernels/gfx_fake/fake_attn.py", "builder": "build_fake", "spec": {"n": 1}}
    open(a.out, "w").write(json.dumps({"packs": [{"arch": ["gfx950"], "kernels": [
        {"name": "enum_a", "kernel_source": ks}]}]}))
    """
)


def dvc_files():
    files = tree_files()
    del files[f"{DESC}/fake.kdp.json"]
    files[f"{DESC}/artifacts.dvc"] = "outs:\n- path: artifacts\n"
    return files


def scenarios():
    s = {"base": tree_files()}
    s["params"] = tree_files(params=[("batch", "i32")])
    s["body"] = tree_files(body="v2")
    s["removed"] = tree_files(variants=KDP_VARIANTS[:1])
    s["reordered"] = tree_files(variants=tuple(reversed(KDP_VARIANTS)))
    s["bad"] = tree_files(
        variants=KDP_VARIANTS + ({"name": "v_bad", "spec": {"boom": True}},)
    )
    s["nohelper"] = tree_files()
    s["nohelper"][
        f"{PROVIDER}/descriptor-packaging/python/hkp_pack/rocke_compile.py"
    ] = HELPERS.replace("build_spec", "renamed_build_spec")
    s["dvc"] = dvc_files()
    s["dvc_enum"] = dvc_files()
    s["dvc_enum"][f"{GEN}/configs/fake.engine.yaml"] = "authored_subpath: rocKE/fake\n"
    s["dvc_enum"][f"{GEN}/tools/registry_enumerator.py"] = ENUMERATOR
    narrowed = (KDP_VARIANTS[0], {**KDP_VARIANTS[1], "arch": ["gfx1100"]})
    s["generic"] = tree_files(
        variants=narrowed, arch=("gfx1100", "gfx1101", "gfx1200"), target="generic"
    )
    s["generic"][
        f"{PROVIDER}/descriptor-packaging/python/hkp_pack/rocke_compile.py"
    ] = GENERIC_HELPERS
    s["bare"] = {"README.md": "x"}
    return s


def fast_import_stream(commits):
    """A `git fast-import` stream: one commit per scenario, tagged with its name."""
    out = bytearray()

    def data(b):
        out.extend(b"data %d\n" % len(b) + b + b"\n")

    for name, files in commits.items():
        out.extend(
            f"commit refs/heads/work\ncommitter t <t@example.com> 1700000000 +0000\n".encode()
        )
        data(name.encode())
        out.extend(b"deleteall\n")
        for path, text in files.items():
            out.extend(f"M 100644 inline {path}\n".encode())
            data(text.encode("utf-8"))
        out.extend(f"\nreset refs/tags/{name}\nfrom refs/heads/work\n\n".encode())
    return bytes(out)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """One repo whose tags are the scenarios (a single git process builds all of them)."""
    root = tmp_path_factory.mktemp("impact_emit")
    cfg = root / "gitconfig"
    cfg.write_text("")
    mp = pytest.MonkeyPatch()
    for k, v in {
        "GIT_CONFIG_GLOBAL": str(cfg),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }.items():
        mp.setenv(k, v)
    r = root / "repo"
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(
        ["git", "-C", str(r), "fast-import", "--quiet"],
        input=fast_import_stream(scenarios()),
        check=True,
    )
    yield r
    mp.undo()


def run(emit, repo, base, head, tmp_path, *extra):
    out = tmp_path / "out.json"
    rc = emit.main(
        [
            "run",
            "--repo",
            str(repo),
            "--base",
            base,
            "--head",
            head,
            "--out",
            str(out),
            *extra,
        ]
    )
    return rc, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)


def only_family(doc):
    assert len(doc["families"]) == 1
    return doc["families"][0]


def test_identical_sides_report_no_change(emit, repo, tmp_path):
    rc, doc = run(emit, repo, "base", "base", tmp_path)
    assert rc == 0
    f = only_family(doc)
    assert (f["arch"], f["builder"], f["spec_origin"]) == (
        "gfx950",
        "build_fake",
        "kdp-in-git",
    )
    assert f["kdp"] == f"{DESC}/fake.kdp.json"
    assert f["members"] == ["gfx950"]
    assert f["variants"] == {"base": 2, "head": 2}
    for k in ("only_in_base", "only_in_head", "params_changed", "ir_changed"):
        assert f[k] == [], k
    assert f["failures"] == {"base": {}, "head": {}}
    assert f["params"]["head"] == [["q", "ptr addrspace(1)"], ["n", "i32"]]
    assert doc["flavor"] == "llvm23"  # top rung of the ladder in the head tree
    assert doc["not_inspected"] == []


def test_extra_param_lists_every_variant_and_shows_head_list(emit, repo, tmp_path):
    rc, doc = run(emit, repo, "base", "params", tmp_path)
    assert rc == 0
    f = only_family(doc)
    assert f["params_changed"] == ["v_a", "v_b"]
    assert set(f["ir_changed"]) == {"v_a", "v_b"}
    assert f["params"]["base"] == [["q", "ptr addrspace(1)"], ["n", "i32"]]
    assert f["params"]["head"][-1] == ["batch", "i32"]


def test_body_only_change_is_ir_changed_not_params(emit, repo, tmp_path):
    _, doc = run(emit, repo, "base", "body", tmp_path)
    f = only_family(doc)
    assert f["params_changed"] == []
    assert f["ir_changed"] == ["v_a", "v_b"]


def test_removed_variant_is_only_in_base_and_unchanged_one_is_not_flagged(
    emit, repo, tmp_path
):
    _, doc = run(emit, repo, "base", "removed", tmp_path)
    f = only_family(doc)
    assert f["only_in_base"] == ["v_b"]
    assert f["only_in_head"] == []
    assert f["variants"] == {"base": 2, "head": 1}
    assert f["params_changed"] == f["ir_changed"] == []


def test_variants_are_matched_by_name_not_position(emit, repo, tmp_path):
    _, doc = run(emit, repo, "base", "reordered", tmp_path)
    f = only_family(doc)
    assert f["only_in_base"] == f["only_in_head"] == f["ir_changed"] == []


def test_raising_builder_is_a_recorded_failure_and_exit_zero(emit, repo, tmp_path):
    rc, doc = run(emit, repo, "bad", "bad", tmp_path)
    assert rc == 0
    f = only_family(doc)
    assert f["variants"] == {"base": 3, "head": 3}
    assert f["failures"]["head"] == {"v_bad": "ValueError: boom"}
    assert f["failures"]["base"] == {"v_bad": "ValueError: boom"}
    assert f["params_changed"] == f["ir_changed"] == []


def test_missing_helper_fails_every_variant_instead_of_skipping(emit, repo, tmp_path):
    rc, doc = run(emit, repo, "nohelper", "nohelper", tmp_path)
    assert rc == 0
    f = only_family(doc)
    assert set(f["failures"]["head"]) == {"v_a", "v_b"}
    assert "build_spec" in f["failures"]["head"]["v_a"]


def test_dvc_held_kdp_without_enumerator_is_not_inspected(emit, repo, tmp_path):
    rc, doc = run(emit, repo, "dvc", "dvc", tmp_path)
    assert rc == 0
    assert doc["families"] == []
    assert sorted(n["side"] for n in doc["not_inspected"]) == ["base", "head"]
    n = doc["not_inspected"][0]
    assert n["kdp"] == f"{DESC}/fake.kdp.json"
    assert "registry_enumerator" in n["reason"]


def test_dvc_held_kdp_is_regenerated_by_the_enumerator_of_that_side(
    emit, repo, tmp_path
):
    _, doc = run(emit, repo, "dvc_enum", "dvc_enum", tmp_path)
    f = only_family(doc)
    assert f["spec_origin"] == "registry_enumerator"
    assert f["variants"] == {"base": 1, "head": 1}
    assert f["kdp"] == f"{DESC}/fake.kdp.json"
    assert doc["not_inspected"] == []


def test_generic_members_collapse_and_descriptor_arch_narrows_the_kdp_arch(
    emit, repo, tmp_path
):
    rc, doc = run(emit, repo, "generic", "generic", tmp_path)
    assert rc == 0
    fams = {f["arch"]: f for f in doc["families"]}
    assert set(fams) == {"gfx11-generic", "gfx12-generic"}
    g11, g12 = fams["gfx11-generic"], fams["gfx12-generic"]
    assert g11["members"] == ["gfx1100", "gfx1101"]
    assert g11["variants"] == {"base": 2, "head": 2}  # v_a once, not once per member
    assert g12["members"] == ["gfx1200"]
    assert g12["variants"] == {"base": 1, "head": 1}  # v_b is gfx1100-only
    assert g11["failures"] == g12["failures"] == {"base": {}, "head": {}}


def test_flavor_is_pinned_in_the_child_and_overridable(
    emit, repo, tmp_path, monkeypatch
):
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")  # caller env must not leak
    _, doc = run(emit, repo, "base", "base^{tree}", tmp_path, "--flavor", "llvm22")
    assert doc["flavor"] == "llvm22"
    assert doc["head"] == "base^{tree}"  # a tree id is a valid rev
    assert only_family(doc)["ir_changed"] == []
    _, doc = run(emit, repo, "base", "base", tmp_path)
    assert doc["flavor"] == "llvm23"


def test_unreadable_rev_exits_one_with_one_line(emit, repo, tmp_path, capsys):
    rc, doc = run(emit, repo, "base", "no-such-rev", tmp_path)
    assert rc == 1 and doc is None
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("impact_emit: ")


def test_tree_without_rocke_exits_one(emit, repo, tmp_path, capsys):
    rc, _ = run(emit, repo, "bare", "bare", tmp_path, "--flavor", "llvm23")
    assert rc == 1
    assert "lacks" in capsys.readouterr().err


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            "define amdgpu_kernel void @attn(ptr addrspace(1) noalias nocapture readonly align 16 %q, "
            "ptr addrspace(1) noalias %o, i32 noundef %batch, float %scale, <4 x float> %v, "
            "ptr %flat) #1 {",
            [
                ["q", "ptr addrspace(1)"],
                ["o", "ptr addrspace(1)"],
                ["batch", "i32"],
                ["scale", "float"],
                ["v", "<4 x float>"],
                ["flat", "ptr"],
            ],
        ),
        ("define amdgpu_kernel void @k() #0 {", []),
    ],
)
def test_parse_kernargs_reads_names_and_types(emit, line, expected):
    ir = f"; header\ndeclare i32 @llvm.foo(i32)\n{line}\n  ret void\n}}\n"
    assert emit.parse_kernargs(ir) == expected


def test_parse_kernargs_prefers_the_kernel_over_helper_functions(emit):
    ir = "define internal void @helper(i32 %a) {\n}\ndefine amdgpu_kernel void @k(i64 %z) #0 {\n}\n"
    assert emit.parse_kernargs(ir) == [["z", "i64"]]
    with pytest.raises(ValueError):
        emit.parse_kernargs("; no functions\n")


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            'A = "llvm20"\nB = "llvm23"\n_LLVM_FLAVOR_LADDER = ((0, A), (23, B))\nLLVM_FLAVORS = (A,)\n',
            "llvm23",
        ),
        ('A = "llvm20"\nB = "llvm22"\nLLVM_FLAVORS: tuple = (A, B)\n', "llvm22"),
    ],
)
def test_default_flavor_is_the_ladder_top_else_the_newest_known_flavor(
    emit, monkeypatch, source, expected
):
    # catches: a tree from before the ladder existed failing instead of using its newest flavor
    monkeypatch.setattr(emit, "git", lambda *args, **kw: source)
    assert emit.default_flavor(".", "rev") == expected
