# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the hipdnn-ingestor-impact impact_emit.py script.

A tiny fake provider tree (fake `hkp_pack` and `rocke` packages with the helper names
the child resolves, a builder module, a KDP) lives in a temporary git repo; the lowering
returns deterministic IR text, so no rocKE, comgr or GPU is involved.
"""

import importlib.util
import json
import os
import sys
import subprocess
import textwrap
import types
import time
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "skills"
    / "hipdnn-ingestor-impact"
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

# The packer's COMGR inputs: options per kernel, a fresh-process attribute, the ISA name.
OPTIONS = """
COMGR_FRESH_PROCESS_ATTR = "comgr_fresh_process"
def _comgr_options_for_kernel(kernel):
    return ["-O3"]
def _resolve_compile_target(arch, isa):
    return arch, "amdgcn-amd-amdhsa--" + arch
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


# rocke.helpers.KernelModule in miniature: a primary kernel plus role-keyed companions.
KERNEL_MODULE = """
class KernelModule:
    def __init__(self, primary, companions):
        self.primary, self.companions = primary, dict(companions)
    @property
    def kernels(self):
        return (self.primary, *self.companions.values())
"""


def module_builder_src(reduce_params=(), reduce_body="r1"):
    """A builder returning a two-launch KernelModule: the primary, then a `reduce` companion."""
    ps = "".join(f', ("{n}", "{t}")' for n, t in reduce_params)
    return (
        "import dataclasses\n"
        "from rocke.helpers import KernelModule\n"
        "@dataclasses.dataclass\n"
        "class Spec:\n    n: int = 1\n    boom: bool = False\n\n"
        "def build_fake(spec, arch):\n"
        '    p = [("q", "ptr addrspace(1)"), ("n", "i32")]\n'
        f'    r = [("out", "ptr addrspace(1)"){ps}]\n'
        '    return KernelModule({"params": p, "body": "v1"},\n'
        f'                        {{"reduce": {{"params": r, "body": "{reduce_body}"}}}})\n'
    )


def module_files(**kw):
    files = tree_files()
    files[f"{PROVIDER}/rocke/platform/python/rocke/helpers/__init__.py"] = KERNEL_MODULE
    files[BUILDER] = module_builder_src(**kw)
    return files


# The packer's predicate lookup in miniature: `supports_<builder minus build_>`.
RESOLVER = """
def _resolve_support_predicate(module, builder):
    name = "supports_" + builder[len("build_"):]
    fn = getattr(module, name, None)
    return (name, fn) if callable(fn) else (None, None)
"""


def table_builder_src(reduce_check="spec.n % 8 == 0"):
    """A KernelModule builder for the predicate tables. Its predicate accepts any positive
    tile, but the builder asserts the tile divides the block. Its `reduce` companion's spec
    shares the field `n` (8 times the primary's), its predicate is REDUCE_CHECK, and the
    primary's predicate refuses n <= 0."""
    return textwrap.dedent(
        f"""
        import dataclasses
        from rocke.helpers import KernelModule

        @dataclasses.dataclass(frozen=True)
        class Spec:
            n: int = 1
            block: int = 32
            tile: int = 16

        @dataclasses.dataclass(frozen=True)
        class RSpec:
            n: int = 8

        def supports_fake(spec, arch="x"):
            return spec.n >= 1 and spec.block >= 1 and spec.tile >= 1, "ok"

        def supports_reduce(spec, arch="x"):
            return {reduce_check}, "ok"

        def build_reduce(spec, arch):
            return {{"params": [("out", "ptr addrspace(1)")], "body": "r"}}

        def build_fake(spec, arch):
            assert spec.block % spec.tile == 0, "tile must divide block"
            p = [("q", "ptr addrspace(1)"), ("n", "i32")]
            return KernelModule({{"params": p, "body": "v1"}},
                                {{"reduce": build_reduce(RSpec(n=8 * spec.n), arch=arch)}})
        """
    )


def table_files(**kw):
    files = module_files()
    files[f"{PROVIDER}/descriptor-packaging/python/hkp_pack/rocke_compile.py"] = (
        HELPERS + RESOLVER
    )
    files[BUILDER] = table_builder_src(**kw)
    return files


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
    spec = importlib.util.spec_from_file_location("hipdnn_ingestor_impact_emit", SCRIPT)
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
    s["dvc_enum_sources"] = dvc_files()
    s["dvc_enum_sources"][
        f"{GEN}/configs/fake.engine.yaml"
    ] = "authored_subpath: rocKE/fake\n"
    s["dvc_enum_sources"][f"{GEN}/tools/registry_enumerator.py"] = ENUMERATOR_SOURCES
    narrowed = (KDP_VARIANTS[0], {**KDP_VARIANTS[1], "arch": ["gfx1100"]})
    s["generic"] = tree_files(
        variants=narrowed, arch=("gfx1100", "gfx1101", "gfx1200"), target="generic"
    )
    s["generic"][
        f"{PROVIDER}/descriptor-packaging/python/hkp_pack/rocke_compile.py"
    ] = GENERIC_HELPERS
    s["dupname"] = tree_files(
        body="v2",
        variants=(
            KDP_VARIANTS[0],
            {"name": "v_a", "spec": {"n": 3}},
            KDP_VARIANTS[1],
        ),
    )
    s["bare"] = {"README.md": "x"}
    compile_py = f"{PROVIDER}/rocke/platform/python/rocke/helpers/compile.py"
    for name, level in (("opts_o3", "-O3"), ("opts_o2", "-O2")):
        s[name] = tree_files()
        s[name][compile_py] = COMPILE + OPTIONS.replace("-O3", level)
    s["cpp_pin"] = tree_files()
    s["cpp_pin"][
        f"{PROVIDER}/descriptor-packaging/python/hkp_pack/rocke_compile.py"
    ] = (HELPERS + '_BACKEND = "cpp"\n')
    s["module"] = module_files()
    s["module_wide"] = module_files(reduce_params=[("lse", "ptr addrspace(1)")])
    s["module_body"] = module_files(reduce_body="r2")
    s["tables"] = table_files()
    s["tables_hang"] = table_files()
    s["tables_hang"][BUILDER] = (
        s["tables_hang"][BUILDER]
        .replace(
            'assert spec.block % spec.tile == 0, "tile must divide block"',
            "while spec.tile < 1:\n        pass",
        )
        .replace(
            'return spec.n >= 1 and spec.block >= 1 and spec.tile >= 1, "ok"',
            'return True, "ok"',
        )
    )
    s["tables_fixed"] = table_files(reduce_check="spec.n >= 1 and spec.n % 8 == 0")
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


def git_out(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


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
    rc, doc = run(emit, repo, "base", "body", tmp_path)
    assert rc == 0
    assert doc["schema"] == 6
    # the recorded ids are the real tree ids, and differ when the sides differ
    assert doc["base_tree"] == git_out(repo, "rev-parse", "base^{tree}")
    assert doc["head_tree"] == git_out(repo, "rev-parse", "body^{tree}")
    assert doc["base_tree"] != doc["head_tree"]
    f = only_family(doc)
    assert f["params_changed"] == []
    assert f["ir_changed"] == ["v_a", "v_b"]


@pytest.mark.parametrize(
    "head, params_changed, ir_changed",
    [("module_wide", ["v_a", "v_b"], ["v_a", "v_b"]), ("module_body", [], ["v_a", "v_b"])],
    ids=["companion-gains-a-param", "companion-body-only"],
)  # fmt: skip
def test_kernel_module_builder_records_and_compares_every_launch(
    emit, repo, tmp_path, head, params_changed, ir_changed
):
    # catches: a KernelModule passed whole to the single-kernel lowering (every variant fails),
    # or only its primary lowered, so a companion launch's kernargs or body change reads equal
    _, doc = run(emit, repo, "module", head, tmp_path)
    f = only_family(doc)
    assert f["failures"] == {"base": {}, "head": {}}
    assert f["params"]["head"] == [["q", "ptr addrspace(1)"], ["n", "i32"]]
    reduce = [["out", "ptr addrspace(1)"]] + (
        [["lse", "ptr addrspace(1)"]] if head == "module_wide" else []
    )
    assert f["companions"]["head"] == {"reduce": reduce}
    assert f["params_changed"] == params_changed
    assert f["ir_changed"] == ir_changed


def findings_of(doc):
    """{(inner builder or "", check, field): finding} over every predicate table."""
    return {
        (t["inner"], f["check"], f["field"]): f
        for t in doc["predicate_tables"]
        for f in t["findings"]
    }


def test_predicate_tables_flag_what_the_builder_rejects_and_a_companion_sign_gap(
    emit, repo, tmp_path
):
    # catches: a predicate that accepts a tile the builder asserts on (#14259 S-1), and a
    # companion predicate that accepts n <= 0 where the module's own predicate refuses it
    # (#14242 r4385341723), reported as nothing; or a field every predicate guards flagged
    _, doc = run(emit, repo, "module", "tables", tmp_path)
    found = findings_of(doc)
    assert set(found) == {("", "builder", "tile"), ("build_reduce", "sign", "n")}
    tile = found["", "builder", "tile"]
    assert 24 in tile["values"] and "AssertionError" in tile["detail"]
    assert set(found["build_reduce", "sign", "n"]["values"]) == {-16, -8, 0}
    # the base's tables could not run (no resolver there): nothing predates the change, and
    # no table counts as tabled at base
    assert not any(f["at_base"] for f in found.values())
    assert not any(t["at_base"] for t in doc["predicate_tables"])
    (main,) = [t for t in doc["predicate_tables"] if not t["inner"]]
    assert (main["predicate"], main["skipped"], main["truncated"]) == (
        "supports_fake",
        "",
        False,
    )


def test_predicate_table_findings_also_at_base_are_pre_existing_and_a_fix_clears_one(
    emit, repo, tmp_path
):
    # catches: a finding the base already had reported as new, or one the head fixed kept
    _, doc = run(emit, repo, "tables", "tables", tmp_path)
    assert all(t["at_base"] for t in doc["predicate_tables"])
    assert all(f["at_base"] for f in findings_of(doc).values())
    _, doc = run(emit, repo, "tables", "tables_fixed", tmp_path)
    found = findings_of(doc)
    assert (
        set(found) == {("", "builder", "tile")}
        and found["", "builder", "tile"]["at_base"]
    )


def test_predicate_tables_say_why_they_did_not_run_and_stop_at_their_budget(
    emit, repo, tmp_path, monkeypatch
):
    # catches: a revision without the packer's resolver read as "no findings", or a table
    # that runs past its budget instead of stopping and saying so
    _, doc = run(emit, repo, "base", "base", tmp_path)
    (t,) = doc["predicate_tables"]
    assert t["skipped"].startswith("hkp_pack has no _resolve_support_predicate")
    monkeypatch.setenv("IMPACT_EMIT_TABLE_SECONDS", "0")
    _, doc = run(emit, repo, "tables", "tables", tmp_path)
    assert all(t["truncated"] and t["rows"] == 0 for t in doc["predicate_tables"])


def test_a_builder_that_hangs_on_a_probed_spec_costs_only_the_tables(
    emit, repo, tmp_path, monkeypatch
):
    # catches: the probe loop sharing the child's --timeout, so one builder that never
    # returns on a spec its predicate accepts (tile=0) kills the child and the whole run
    # exits 1 with no JSON although every variant built and compared
    monkeypatch.setenv("IMPACT_EMIT_TABLE_SECONDS", "0.2")
    t0 = time.time()
    rc, doc = run(
        emit, repo, "tables_hang", "tables_hang", tmp_path, "--timeout", "120"
    )
    assert time.time() - t0 < 100
    assert rc == 0
    f = only_family(doc)
    assert f["variants"] == {"base": 2, "head": 2}
    assert f["failures"] == {"base": {}, "head": {}}
    assert {t["inner"] for t in doc["predicate_tables"]} == {"", "build_reduce"}
    assert all(t["skipped"] == "probe timed out" for t in doc["predicate_tables"])


def probe_src(fields, pred, body="", reduce_pred=None):
    """A builder module for one predicate-table case. FIELDS are the Spec fields, PRED the
    full `supports_fake` def ("" for none), BODY lines run first in `build_fake`. With
    REDUCE_PRED, `build_fake` returns a KernelModule whose companion is built by the
    inner `build_reduce` (spec `RSpec(n=spec.n)`), guarded by that predicate."""
    fields = "\n".join(f"    {f}" for f in fields)
    body = "\n".join(f"    {b}" for b in body)
    inner = (
        "\n@dataclasses.dataclass(frozen=True)\nclass RSpec:\n    n: int = 1\n\n"
        f"def supports_reduce(spec, arch='x'):\n    return {reduce_pred}\n\n"
        "def build_reduce(spec, arch):\n"
        '    return {"params": [("o", "ptr addrspace(1)")], "body": "r"}\n'
        if reduce_pred
        else ""
    )
    tail = (
        '    return KernelModule({"params": p, "body": "v"},'
        ' {"r": build_reduce(RSpec(n=spec.n), arch=arch)})\n'
        if reduce_pred
        else '    return {"params": p, "body": "v"}\n'
    )
    return (
        "import dataclasses\nfrom rocke.helpers import KernelModule\n\n"
        f"@dataclasses.dataclass(frozen=True)\nclass Spec:\n{fields}\n\n{pred}\n{inner}\n"
        f"def build_fake(spec, arch):\n{body}\n"
        '    p = [("q", "ptr addrspace(1)")]\n'
        f"{tail}"
    )


def probe_tables(emit, tmp_path, src, specs):
    """The head tables the child writes for a tree whose builder is SRC and whose KDP ships
    SPECS (one variant each)."""
    variants = tuple({"name": f"v{i}", "spec": s} for i, s in enumerate(specs))
    files = tree_files(variants=variants)
    files[f"{PROVIDER}/rocke/platform/python/rocke/helpers/__init__.py"] = KERNEL_MODULE
    files[f"{PROVIDER}/descriptor-packaging/python/hkp_pack/rocke_compile.py"] = (
        HELPERS + RESOLVER
    )
    files[BUILDER] = src
    root = tmp_path / "tree"
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    out = tmp_path / "side.json"
    r = emit.run_tree(
        [sys.executable, str(SCRIPT), "child", "--root", str(root), "--out", str(out)],
        120,
        cwd=str(root),
        env=emit.child_env(str(root), "llvm23"),
    )
    assert r.returncode == 0, r.stderr
    return json.loads(out.read_text(encoding="utf-8"))["tables"]


ACCEPT = "def supports_fake(spec, arch='x'):\n    return {}, 'ok'\n"
RAISE_AT_3 = ["if spec.n == 3:", "    raise ValueError('three')"]


@pytest.mark.parametrize(
    "src, specs, skipped, found",
    [
        (  # a bare-bool predicate, not a (bool, reason) tuple
            probe_src(
                ["n: int = 1"],
                "def supports_fake(spec, arch='x'):\n    return spec.n >= 1\n",
                RAISE_AT_3,
            ),
            [{"n": 1}, {"n": 2}],
            "",
            {("", "builder", "n"): [3]},
        ),
        (  # a predicate that takes no arch
            probe_src(
                ["n: int = 1"],
                "def supports_fake(spec):\n    return spec.n >= 1, 'ok'\n",
                RAISE_AT_3,
            ),
            [{"n": 1}, {"n": 2}],
            "",
            {("", "builder", "n"): [3]},
        ),
        (  # a bool field shipped True: False is the value only the bool candidates reach
            probe_src(
                ["flag: bool = True"],
                ACCEPT.format("True"),
                ["if not spec.flag:", "    raise ValueError('off')"],
            ),
            [{"flag": True}],
            "",
            {("", "builder", "flag"): [False]},
        ),
        (  # more than six shipped values: only the first four and last two seed the
            # derived values (15 from 10, never 75 from 50)
            probe_src(
                ["n: int = 1"],
                ACCEPT.format("True"),
                ["if spec.n in (15, 75):", "    raise ValueError('x')"],
            ),
            [{"n": 10 * i} for i in range(1, 9)],
            "",
            {("", "builder", "n"): [15]},
        ),
        (  # a finding only the second seed reaches (n=3 with m=2)
            probe_src(
                ["n: int = 1", "m: int = 1"],
                ACCEPT.format("True"),
                ["if spec.n == 3 and spec.m == 2:", "    raise ValueError('x')"],
            ),
            [{"n": 1, "m": 1}, {"n": 2, "m": 2}],
            "",
            {("", "builder", "n"): [3]},
        ),
        (  # a field shipping 0: only the negative values are outside its sign range
            probe_src(["n: int = 1"], ACCEPT.format("True"), reduce_pred="spec.n >= 0"),
            [{"n": 0}, {"n": 1}],
            "",
            {("", "sign", "n"): [-1]},
        ),
        (  # a field shipping a negative value has no sign range to leave
            probe_src(["n: int = 1"], ACCEPT.format("True"), reduce_pred="spec.n >= 0"),
            [{"n": -2}, {"n": 1}],
            "",
            {},
        ),
        (
            probe_src(["n: int = 1"], "", RAISE_AT_3),
            [{"n": 1}, {"n": 2}],
            "no support predicate takes its spec",
            {},
        ),
        (  # a spec class that is not a dataclass
            probe_src(["n: int = 1"], ACCEPT.format("True")).replace(
                "@dataclasses.dataclass(frozen=True)\nclass Spec:",
                "class Spec:\n    def __init__(self, n=1):\n        self.n = n\n",
            ),
            [{"n": 1}],
            "no shipped spec, or a spec that is not a dataclass instance",
            {},
        ),
    ],
    ids=[
        "bare-bool-predicate",
        "predicate-without-arch",
        "bool-field",
        "many-shipped-values",
        "second-seed",
        "ships-zero",
        "ships-negative",
        "no-predicate",
        "non-dataclass-spec",
    ],
)
def test_predicate_table_probes_each_kind_of_field_and_predicate(
    emit, tmp_path, src, specs, skipped, found
):
    # catches: a bare-bool predicate read as res[0]; a predicate without `arch` called with
    # one; a bool field probed with only True; the many-shipped-values cut dropped; one seed
    # instead of several; the sign guard for a field that ships 0 or a negative value; a
    # missing predicate or a non-dataclass spec reported as "no findings" with no reason
    tables = probe_tables(emit, tmp_path, src, specs)
    assert tables[0]["skipped"] == skipped
    got = {
        (t["inner"], f["check"], f["field"]): f["values"]
        for t in tables
        for f in t["findings"]
    }
    assert got == found


def test_a_finding_is_at_base_only_for_the_same_check_on_the_same_field(emit):
    # catches: a finding marked pre-existing because the base tabled the builder, or
    # because it found another check on the field
    def table(*findings):
        return {
            "arch": "gfx950",
            "source": "s.py",
            "builder": "build_x",
            "inner": "",
            "skipped": "",
            "findings": [{"check": c, "field": f} for c, f in findings],
        }

    base = {"tables": [table(("sign", "n"), ("builder", "m"))]}
    head = {"tables": [table(("builder", "n"), ("sign", "n"), ("builder", "m"))]}
    (t,) = emit.compare_tables(base, head)
    assert t["at_base"]
    assert [(f["check"], f["field"], f["at_base"]) for f in t["findings"]] == [
        ("builder", "n", False),
        ("sign", "n", True),
        ("builder", "m", True),
    ]


def test_the_emitted_document_is_accepted_by_impacts_loader(emit, tmp_path):
    # catches: the emitter's schema literal and impact.py's EMIT_SCHEMA bumped one-sided,
    # so every emit file is then rejected; or a document field the loader reads changing
    spec = importlib.util.spec_from_file_location(
        "hipdnn_ingestor_impact_for_emit", SCRIPT.with_name("impact.py")
    )
    impact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impact)
    side = {"not_inspected": [{"kdp": "k.kdp.json", "reason": "r"}]}
    sides = {"base": side, "head": side}
    names = ["v"]
    fam = family(names, {"v": [["q", "ptr"]]})
    args = types.SimpleNamespace(base="b", head="h")
    tree = "0" * 40
    doc = emit.build_doc(
        args, "llvm23", tree, tree, sides["base"], sides["head"],
        emit.compare({"families": [fam]}, {"families": [fam]}), [], 1.0,
    )  # fmt: skip
    path = tmp_path / "emit.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert impact.load_emit(path)["schema"] == doc["schema"]


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
    for side in ("head", "base"):
        assert set(f["failures"][side]) == {"v_bad"}
        assert f["failures"][side]["v_bad"].startswith("ValueError")
    assert f["params_changed"] == f["ir_changed"] == []


def test_comgr_option_change_with_equal_ir_is_comgr_changed(emit, repo, tmp_path):
    # catches: a COMGR option change (the code object differs, the IR does not) reported
    # as every variant equal
    _, doc = run(emit, repo, "opts_o3", "opts_o2", tmp_path)
    f = only_family(doc)
    assert f["ir_changed"] == [] and f["params_changed"] == []
    assert f["comgr_changed"] == ["v_a", "v_b"]
    _, doc = run(emit, repo, "opts_o3", "opts_o3", tmp_path)
    assert only_family(doc)["comgr_changed"] == []


def test_packer_pinned_to_another_backend_fails_every_variant(emit, repo, tmp_path):
    # catches: Python-lowered IR compared as if it were what a packer pinned to the C++
    # backend ships
    _, doc = run(emit, repo, "cpp_pin", "cpp_pin", tmp_path)
    f = only_family(doc)
    assert set(f["failures"]["head"]) == {"v_a", "v_b"}
    assert "'cpp'" in f["failures"]["head"]["v_a"]


def test_missing_helper_fails_every_variant_instead_of_skipping(emit, repo, tmp_path):
    rc, doc = run(emit, repo, "nohelper", "nohelper", tmp_path)
    assert rc == 0
    f = only_family(doc)
    assert set(f["failures"]["head"]) == {"v_a", "v_b"}
    assert f["failures"]["head"]["v_a"]


def test_dvc_held_kdp_without_enumerator_is_not_inspected(emit, repo, tmp_path):
    rc, doc = run(emit, repo, "dvc", "dvc", tmp_path)
    assert rc == 0
    assert doc["families"] == []
    assert sorted(n["side"] for n in doc["not_inspected"]) == ["base", "head"]
    n = doc["not_inspected"][0]
    assert n["kdp"] == f"{DESC}/fake.kdp.json"
    assert n["reason"]


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
    tree = git_out(repo, "rev-parse", "base^{tree}")
    assert doc["head_tree"] == doc["base_tree"] == tree
    assert only_family(doc)["ir_changed"] == []
    _, doc = run(emit, repo, "base", "base", tmp_path)
    assert doc["flavor"] == "llvm23"


def test_dup_name_with_different_specs_is_a_failure_not_silently_one_variant(
    emit, repo, tmp_path
):
    # catches: the second of two same-named variants being dropped unnoticed
    rc, doc = run(emit, repo, "base", "dupname", tmp_path)
    assert rc == 0
    f = only_family(doc)
    assert "v_a" in f["failures"]["head"]
    assert f["ir_changed"] == ["v_b"]  # v_a takes no part in the comparison
    rc, doc = run(emit, repo, "dupname", "dupname", tmp_path)
    assert only_family(doc)["failures"]["base"].keys() == {"v_a"}


def test_unreadable_rev_exits_one_with_one_line(emit, repo, tmp_path, capsys):
    rc, doc = run(emit, repo, "base", "no-such-rev", tmp_path)
    assert rc == 1 and doc is None
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("impact_emit: ")


def test_tree_without_rocke_exits_one(emit, repo, tmp_path, capsys):
    rc, _ = run(emit, repo, "bare", "bare", tmp_path, "--flavor", "llvm23")
    assert rc == 1
    assert capsys.readouterr().err.startswith("impact_emit: ")


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


LADDER_22 = 'A = "llvm20"\nB = "llvm22"\n_LLVM_FLAVOR_LADDER = ((0, A), (22, B))\n'
LADDER_23 = (
    'A = "llvm20"\nB = "llvm22"\nC = "llvm23"\n'
    "_LLVM_FLAVOR_LADDER = ((0, A), (22, B), (23, C))\n"
)


@pytest.mark.parametrize(
    "base_src, head_src, expected",
    [
        (LADDER_22, LADDER_23, "llvm22"),  # base predates the head's top rung
        (LADDER_23, LADDER_23, "llvm23"),
        (LADDER_23, LADDER_22, "llvm22"),
        ('X = "llvm19"\nLLVM_FLAVORS = (X,)\n', LADDER_23, "llvm23"),  # none shared
    ],
)
def test_shared_default_flavor_is_the_heads_newest_that_the_base_knows(
    emit, monkeypatch, base_src, head_src, expected
):
    # catches: a base lacking the head's top flavor failing every variant
    srcs = {"b": base_src, "h": head_src}

    def fake_git(repo, cmd, *rest, **kw):
        return "" if cmd == "ls-tree" else srcs[rest[0].split(":")[0]]

    monkeypatch.setattr(emit, "git", fake_git)
    assert emit.shared_default_flavor(".", "b", "h") == expected


def test_default_flavor_follows_the_ladder_into_llvm_profile(emit, monkeypatch):
    # catches: the ladder read from lower_llvm.py only, which newer trees reduce to an import
    # from rocke/core/llvm_profile.py, so every run fails with "no _LLVM_FLAVOR_LADDER"
    files = {
        emit.LOWER_LLVM: "from .llvm_profile import LLVM_FLAVORS, _LLVM_FLAVOR_LADDER\n",
        emit.LLVM_PROFILE: LADDER_23,
    }

    def fake_git(repo, cmd, *rest, **kw):
        if cmd == "ls-tree":
            return "\n".join(p for p in rest[-1:] if p in files)
        return files[rest[0].split(":", 1)[1]]

    monkeypatch.setattr(emit, "git", fake_git)
    assert emit.default_flavor(".", "rev") == "llvm23"
    assert emit.shared_default_flavor(".", "b", "h") == "llvm23"


def test_child_env_is_an_allow_list_that_pins_the_rocke_variables(
    emit, tmp_path, monkeypatch
):
    # catches: parent credentials and import paths reaching the executed rocKE code
    for k, v in {
        "GH_TOKEN": "t",
        "AWS_SECRET_ACCESS_KEY": "s",
        "ROCKE_FOO": "1",
        "HIP_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "/bogus",
        "ROCKE_LLVM_FLAVOR": "llvm20",
        "LC_ALL": "C",
        "IMPACT_EMIT_ENUM_TIMEOUT": "7",
    }.items():
        monkeypatch.setenv(k, v)
    env = emit.child_env(tmp_path, "llvm23")
    names = {k.upper() for k in env}
    assert not {"GH_TOKEN", "AWS_SECRET_ACCESS_KEY", "ROCKE_FOO"} & names
    assert "HIP_VISIBLE_DEVICES" not in names
    assert "PATH" in names and env["LC_ALL"] == "C"
    assert env["IMPACT_EMIT_ENUM_TIMEOUT"] == "7"
    assert env["ROCKE_LLVM_FLAVOR"] == "llvm23"
    assert "/bogus" not in env["PYTHONPATH"]
    assert str(tmp_path) in env["PYTHONPATH"]
    assert env["ROCKE_PROVIDER_ROOT"].startswith(str(tmp_path))
    assert env["PYTHONDONTWRITEBYTECODE"] == env["PYTHONUTF8"] == "1"


def test_timeout_kills_a_grandchild_holding_the_pipes_and_exits_one(
    emit, repo, tmp_path, monkeypatch, capsys
):
    # catches: --timeout not bounding wall time when a grandchild outlives the child
    fake = tmp_path / "fake_child.py"
    fake.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "time.sleep(120)\n"
    )
    monkeypatch.setattr(emit, "__file__", str(fake))
    t0 = time.time()
    rc, doc = run(emit, repo, "base", "base", tmp_path, "--timeout", "2")
    assert time.time() - t0 < 60
    assert rc == 1 and doc is None
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and err[0].startswith("impact_emit: ")


def sources_kdp(ukds, arch=("gfx950",)):
    return json.dumps({"arch": list(arch), "kernelDescriptors": ukds})


KS = {
    "kind": "rocke",
    "source": "kernels/gfx_fake/fake_attn.py",
    "builder": "build_fake",
    "spec": {"n": 1},
}


def discover(emit, tmp_path, ukds):
    kdp = tmp_path / emit.DESCRIPTORS / "rocKE" / "x" / "x.kdp.json"
    kdp.parent.mkdir(parents=True)
    kdp.write_text(sources_kdp(ukds))
    return emit._discover(tmp_path)


def test_sources_ukd_sharing_source_builder_spec_is_one_variant_named_by_the_ukd(
    emit, tmp_path
):
    # catches: a multi-launch UKD skipped, or one variant per launch (a second build and a
    # name that is not the UKD's)
    ukd = {"name": "attn", "sources": {"decode": KS, "merge": {**KS, "role": "merge"}}}
    variants, skipped = discover(emit, tmp_path, [ukd])
    assert skipped == []
    assert len(variants) == 1
    v = variants[0]
    assert (v["name"], v["source"], v["builder"], v["spec"], v["target"]) == (
        "attn",
        KS["source"],
        "build_fake",
        {"n": 1},
        "native",
    )
    assert v["arches"] == ["gfx950"] and v["origin"] == "kdp-in-git"


def test_sources_ukd_launches_with_different_specs_are_variants_named_by_launch(
    emit, tmp_path
):
    # catches: launches differing in spec collapsed into one variant, or named alike
    ukd = {
        "name": "attn",
        "sources": {
            "merge": {**KS, "spec": {"n": 2}},
            "decode": KS,
            "decode2": {**KS, "spec": {"n": 1}},
        },
    }
    variants, skipped = discover(emit, tmp_path, [ukd])
    assert skipped == []
    assert {v["name"]: v["spec"] for v in variants} == {
        "attn@decode+decode2": {"n": 1},
        "attn@merge": {"n": 2},
    }


def test_sources_spec_key_order_does_not_split_a_variant(emit, tmp_path):
    # catches: a group key that compares spec dicts by identity or insertion order
    ukd = {
        "name": "attn",
        "sources": {
            "a": {**KS, "spec": {"n": 1, "m": 2}},
            "b": {**KS, "spec": {"m": 2, "n": 1}},
        },
    }
    variants, _ = discover(emit, tmp_path, [ukd])
    assert [v["name"] for v in variants] == ["attn"]


@pytest.mark.parametrize(
    "sources",
    [
        {"a": {"kind": "rocke", "source": "s.py"}},
        {"a": {"kind": "rocke", "builder": "build_x"}},
        {"a": KS, "b": {**KS, "kind": "hip"}},
        {"a": "not-an-object"},
        {},
        ["a"],
        {"a": {**KS, "kind": ["rocke"]}},
        {"a": {**KS, "source": ["s.py"]}},
        {"a": {**KS, "builder": {"name": "build_x"}}},
        {"a": {**KS, "target": ["generic"]}},
    ],
    ids=[
        "no-builder",
        "no-source",
        "mixed-kinds",
        "entry-not-object",
        "empty",
        "not-object",
        "kind-list",
        "source-list",
        "builder-object",
        "target-list",
    ],
)
def test_unreadable_sources_ukd_is_not_inspected_and_yields_no_variant(
    emit, tmp_path, sources
):
    # catches: a `sources` UKD or entry the emitter cannot read dropped silently, or one whose
    # kind, source, builder or target is not a string crashing the child (unhashable key)
    # instead of being listed
    good = {"name": "ok", "kernel_source": KS}
    variants, skipped = discover(
        emit, tmp_path, [{"name": "bad", "sources": sources}, good]
    )
    assert [v["name"] for v in variants] == ["ok"]
    assert len(skipped) == 1
    assert skipped[0]["kdp"].endswith("x/x.kdp.json")
    assert skipped[0]["reason"].startswith("bad: ")


def test_ukd_with_both_kernel_source_and_sources_is_not_inspected(emit, tmp_path):
    variants, skipped = discover(
        emit, tmp_path, [{"name": "both", "kernel_source": KS, "sources": {"a": KS}}]
    )
    assert variants == []
    assert [s["reason"].split(":")[0] for s in skipped] == ["both"]


@pytest.mark.parametrize(
    "ukd",
    [{"name": "hip", "kernel_source": {"kind": "hip", "file": "k.hip"}},
     {"name": "plain"}, {"name": "nulled", "kernel_source": None}],
    ids=["hip", "no-kernel-source", "null-kernel-source"],
)  # fmt: skip
def test_single_kernel_source_without_source_and_builder_is_still_skipped_silently(
    emit, tmp_path, ukd
):
    variants, skipped = discover(emit, tmp_path, [ukd])
    assert variants == [] and skipped == []


@pytest.mark.parametrize(
    "ukd",
    [{"name": ["bad"], "kernel_source": KS}, {"name": {"bad": 1}, "sources": {"a": KS}}],
    ids=["single-list-name", "sources-object-name"],
)  # fmt: skip
def test_descriptor_whose_name_is_not_a_string_is_not_inspected(emit, tmp_path, ukd):
    # catches: a list or object `name` passing the launch checks and failing the child, where
    # it keys the specs by variant name (unhashable), instead of being listed
    variants, skipped = discover(
        emit, tmp_path, [ukd, {"name": "ok", "kernel_source": KS}]
    )
    assert [v["name"] for v in variants] == ["ok"]
    assert len(skipped) == 1
    assert skipped[0]["reason"].endswith("the descriptor `name` is not a string")


def test_single_kernel_source_without_a_name_is_named_by_its_id(emit, tmp_path):
    # catches: a nameless descriptor failing discovery (KeyError) where a `sources` UKD of the
    # same shape is named by its id
    variants, skipped = discover(emit, tmp_path, [{"id": "u1", "kernel_source": KS}])
    assert [v["name"] for v in variants] == ["u1"] and skipped == []


@pytest.mark.parametrize(
    "kernel_source",
    [{**KS, "source": ["s.py"]}, {**KS, "builder": {"name": "build_x"}},
     {**KS, "target": ["generic"]}, ["not", "an", "object"]],
    ids=["source-list", "builder-object", "target-list", "not-object"],
)  # fmt: skip
def test_malformed_single_kernel_source_is_not_inspected(emit, tmp_path, kernel_source):
    # catches: a single launch whose source, builder or target is not a string, or a
    # kernel_source that is not an object, crashing the child (the family key is unhashable)
    # instead of being listed under not_inspected
    good = {"name": "ok", "kernel_source": KS}
    variants, skipped = discover(
        emit, tmp_path, [{"name": "bad", "kernel_source": kernel_source}, good]
    )
    assert [v["name"] for v in variants] == ["ok"]
    assert [s["reason"].split(":")[0] for s in skipped] == ["bad"]


@pytest.mark.parametrize("kind", ["hip", "hsaco", "kpack", "embedded_source"])
def test_multi_launch_ukd_of_another_kind_is_skipped_like_a_single_launch(
    emit, tmp_path, kind
):
    # catches: a hip or hsaco `sources` UKD, which the packer accepts and the emitter has
    # nothing to build for, listed as not inspected, so no emit file could ever fold
    launch = {"kind": kind, "file": "k.co", "symbol": "k"}
    variants, skipped = discover(
        emit, tmp_path, [{"name": kind, "sources": {"a": launch, "b": dict(launch)}}]
    )
    assert variants == [] and skipped == []


ENUMERATOR_SOURCES = textwrap.dedent(
    """
    import argparse, json
    p = argparse.ArgumentParser(); p.add_argument("--manifest"); p.add_argument("--out")
    a = p.parse_args()
    ks = {"kind": "rocke", "source": "kernels/gfx_fake/fake_attn.py",
          "builder": "build_fake", "spec": {"n": 1}}
    open(a.out, "w").write(json.dumps({"packs": [{"arch": ["gfx950"], "kernels": [
        {"name": "enum_m", "sources": {"decode": ks, "merge": {**ks, "role": "merge"}}},
        {"name": "enum_bad", "sources": {"x": {"kind": "rocke", "source": "s.py"}}}]}]}))
    """
)


def test_regenerated_sources_ukd_is_one_variant_and_an_unreadable_one_is_not_inspected(
    emit, repo, tmp_path
):
    # catches: the enumerator's `sources` kernels read differently from the git-tracked ones
    _, doc = run(emit, repo, "dvc_enum_sources", "dvc_enum_sources", tmp_path)
    f = only_family(doc)
    assert f["variants"] == {"base": 1, "head": 1}
    assert f["failures"] == {"base": {}, "head": {}}
    assert sorted(
        (n["side"], n["reason"].split(":")[0]) for n in doc["not_inspected"]
    ) == [
        ("base", "enum_bad"),
        ("head", "enum_bad"),
    ]


def family(names, params_by_name, ir="ir"):
    return {
        "arch": "gfx950",
        "source": "s.py",
        "builder": "build_x",
        "kdp": "k.kdp.json",
        "spec_origin": "kdp-in-git",
        "names": list(names),
        "built": {
            n: {"params": params_by_name[n], "ir": ir, "comgr": None} for n in names
        },
        "failures": {},
    }


def test_compare_params_reordered_is_params_changed(emit):
    # catches: params compared as a set or by sorted names, so a reorder (a different
    # kernarg ABI) reads unchanged
    qn = [["q", "ptr"], ["n", "i32"]]
    nq = [["n", "i32"], ["q", "ptr"]]
    (f,) = emit.compare(
        {"families": [family(["v"], {"v": qn})]},
        {"families": [family(["v"], {"v": nq})]},
    )
    assert f["params_changed"] == ["v"]
    assert f["ir_changed"] == []


def test_compare_lists_variants_only_on_each_side(emit):
    # catches: only_in_base and only_in_head swapped or merged, and a family present on
    # one side only losing its variants
    p = [["q", "ptr"]]
    (f,) = emit.compare(
        {"families": [family(["keep", "gone"], {"keep": p, "gone": p})]},
        {"families": [family(["keep", "new"], {"keep": p, "new": p})]},
    )
    assert f["only_in_base"] == ["gone"]
    assert f["only_in_head"] == ["new"]
    (g,) = emit.compare({"families": []}, {"families": [family(["new"], {"new": p})]})
    assert g["only_in_head"] == ["new"] and g["only_in_base"] == []
    assert g["variants"] == {"base": 0, "head": 1}
    (g,) = emit.compare({"families": [family(["old"], {"old": p})]}, {"families": []})
    assert g["only_in_base"] == ["old"] and g["only_in_head"] == []


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")
def test_run_tree_own_group_false_keeps_the_child_in_the_callers_group(emit):
    # catches: an enumerator launched through run_tree landing in its own session, out of
    # reach of the killpg that bounds the child
    code = [emit.sys.executable, "-c", "import os; print(os.getpgrp())"]
    inherited = emit.run_tree(code, 30, own_group=False)
    own = emit.run_tree(code, 30)
    assert int(inherited.stdout) == os.getpgrp()
    assert int(own.stdout) != os.getpgrp()


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")
def test_run_tree_own_group_false_still_times_out_and_kills_the_child(emit):
    code = [emit.sys.executable, "-c", "import time; time.sleep(60)"]
    t0 = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        emit.run_tree(code, 1, own_group=False)
    assert time.time() - t0 < 30
