import dataclasses
import hashlib
import json
import shutil
import sys
import textwrap
from typing import Literal, Optional

import pytest

from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError
from hkp_pack.pipeline import run_pipeline
from hkp_pack.rocke_compile import (
    build_spec,
    compile_rocke_variant,
    rocke_variant_key,
)

ARCH = "gfx950"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_fixture(tmp_path, fixture):
    dst = tmp_path / "src"
    shutil.copytree(fixture, dst)
    return dst


def _run(source_root, tmp_path, hipcc, rocm_kpack_dir, arches=(ARCH,)):
    return run_pipeline(
        source_root=source_root,
        arches=list(arches),
        out_root=tmp_path / "out",
        hipcc=hipcc,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )


def _load_kpack(rocm_kpack_dir):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    return kpack


# --- Local stub dataclasses for build_spec recursion (comgr-free) -----------
@dataclasses.dataclass
class _Leaf:
    x: int
    tag: str = "leaf"


@dataclasses.dataclass
class _Nested:
    name: str
    leaf: _Leaf
    opt: Optional[_Leaf] = None
    mode: Literal["a", "b"] = "a"


@dataclasses.dataclass
class _Flat:
    a: int
    b: str
    c: bool = False


@dataclasses.dataclass
class _WithList:
    items: list


@dataclasses.dataclass
class _PostInit:
    v: int

    def __post_init__(self):
        if self.v < 0:
            raise ValueError("v must be non-negative")


# --- A. build_spec recursion (quick, comgr-free) ----------------------------
@pytest.mark.quick
def test_build_spec_flat_base_case():
    obj = build_spec(_Flat, {"a": 1, "b": "hi", "c": True})
    assert isinstance(obj, _Flat)
    assert (obj.a, obj.b, obj.c) == (1, "hi", True)


@pytest.mark.quick
def test_build_spec_nested_optional_literal():
    obj = build_spec(
        _Nested,
        {"name": "n", "leaf": {"x": 5}, "opt": {"x": 9, "tag": "t"}, "mode": "b"},
    )
    assert isinstance(obj, _Nested)
    assert isinstance(obj.leaf, _Leaf) and obj.leaf.x == 5 and obj.leaf.tag == "leaf"
    assert isinstance(obj.opt, _Leaf) and obj.opt.x == 9 and obj.opt.tag == "t"
    assert obj.mode == "b"


@pytest.mark.quick
def test_build_spec_optional_none():
    obj = build_spec(_Nested, {"name": "n", "leaf": {"x": 1}, "opt": None})
    assert obj.opt is None
    assert isinstance(obj.leaf, _Leaf)


@pytest.mark.quick
def test_build_spec_unknown_key_rejected():
    with pytest.raises(HkpPackError, match="unexpected spec field"):
        build_spec(_Flat, {"a": 1, "b": "x", "typo": 3})


@pytest.mark.quick
def test_build_spec_unknown_nested_key_rejected():
    with pytest.raises(HkpPackError, match="unexpected spec field"):
        build_spec(_Nested, {"name": "n", "leaf": {"x": 1, "bad": 2}})


@pytest.mark.quick
def test_build_spec_list_field_rejected():
    with pytest.raises(HkpPackError, match="unsupported spec field type"):
        build_spec(_WithList, {"items": [1, 2, 3]})


@pytest.mark.quick
def test_build_spec_missing_field_propagates():
    # build_spec does not catch construction errors: a missing required field
    # surfaces as TypeError for compile_rocke_variant to wrap as "invalid spec".
    with pytest.raises(TypeError):
        build_spec(_Flat, {"b": "x"})


@pytest.mark.quick
def test_build_spec_post_init_rejection_propagates():
    with pytest.raises(ValueError, match="non-negative"):
        build_spec(_PostInit, {"v": -1})


# --- B. adapter contract via a stub builder module (quick, comgr-free) ------
def _write_stub_pkg(tmp_path, body, pkg="stubpkg", sub="sub", mod="mod"):
    """Place an importable stub package on sys.path and return its dotted source
    path ('pkg/sub/mod.py'). The stub monkeypatches nothing itself; each test
    patches rocke_compile._load_compiler to a fake so no comgr is touched."""
    base = tmp_path / pkg
    (base / sub).mkdir(parents=True)
    (base / "__init__.py").write_text("", encoding="utf-8")
    (base / sub / "__init__.py").write_text("", encoding="utf-8")
    (base / sub / f"{mod}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    return f"{pkg}/{sub}/{mod}.py"


class _FakeArtifact:
    def __init__(self, name, data):
        self.kernel_name = name
        self.hsaco = data


class _FakeComgrError(Exception):
    pass


def _patch_compiler(monkeypatch, name="stub_symbol", data=b"ELF\x00stub_symbol\x00"):
    """Stub the comgr entry, recording the backend the producer requested.

    The recorder exists so a test can assert the producer PINS the backend
    rather than merely tolerating one. Without it, dropping the pin would leave
    every one of these tests green.
    """
    from hkp_pack import rocke_compile

    seen = {}

    def _fake_compile(kernel, *, arch, capture_ir_text=False, backend=None):
        seen["backend"] = backend
        return _FakeArtifact(name, data)

    monkeypatch.setattr(
        rocke_compile, "_load_compiler", lambda: (_fake_compile, _FakeComgrError)
    )
    return name, data, seen


_GOOD_STUB = """
    import dataclasses

    @dataclasses.dataclass
    class StubSpec:
        n: int
        label: str = "x"

    def build_stub(spec: StubSpec, *, arch="gfx950"):
        return ("kernel", spec, arch)
"""


@pytest.mark.quick
def test_adapter_import_build_capture(tmp_path, monkeypatch):
    src = _write_stub_pkg(tmp_path, _GOOD_STUB)
    name, data, seen = _patch_compiler(monkeypatch)
    co, symbol = compile_rocke_variant(
        src, "build_stub", {"n": 3, "label": "y"}, ARCH, tmp_path / "co"
    )
    assert symbol == name
    assert co.read_bytes() == data
    assert co.name.startswith("mod_") and co.suffix == ".co"
    # The producer must PIN the lowering backend rather than inherit rocke's
    # default: the wheel has no C++ engine, so an unpinned request degrades to
    # Python via a fallback and the artifact misreports which engine built it.
    assert seen["backend"] == "python"


@pytest.mark.quick
def test_adapter_source_dotted_derivation(tmp_path, monkeypatch):
    # source path with a nested folder resolves via the derived dotted module.
    src = _write_stub_pkg(tmp_path, _GOOD_STUB, pkg="stubpkg2", sub="deep", mod="k")
    _patch_compiler(monkeypatch)
    co, symbol = compile_rocke_variant(
        src, "build_stub", {"n": 1}, ARCH, tmp_path / "co"
    )
    assert symbol == "stub_symbol"


@pytest.mark.quick
def test_adapter_module_not_importable(tmp_path, monkeypatch):
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="module not importable"):
        compile_rocke_variant(
            "does/not/exist.py", "build_stub", {}, ARCH, tmp_path / "co"
        )


@pytest.mark.quick
def test_adapter_builder_not_found(tmp_path, monkeypatch):
    src = _write_stub_pkg(tmp_path, _GOOD_STUB, pkg="stubpkg_bnf")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="builder not found"):
        compile_rocke_variant(src, "no_such_builder", {"n": 1}, ARCH, tmp_path / "co")


@pytest.mark.quick
def test_adapter_spec_not_introspectable(tmp_path, monkeypatch):
    body = """
        def build_stub(spec, *, arch="gfx950"):
            return ("kernel", spec, arch)
    """
    src = _write_stub_pkg(tmp_path, body, pkg="stubpkg_noann")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="spec type not introspectable"):
        compile_rocke_variant(src, "build_stub", {"n": 1}, ARCH, tmp_path / "co")


@pytest.mark.quick
def test_adapter_non_dataclass_spec_hint(tmp_path, monkeypatch):
    body = """
        def build_stub(spec: int, *, arch="gfx950"):
            return ("kernel", spec, arch)
    """
    src = _write_stub_pkg(tmp_path, body, pkg="stubpkg_int")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="spec type not introspectable"):
        compile_rocke_variant(src, "build_stub", {"n": 1}, ARCH, tmp_path / "co")


@pytest.mark.quick
def test_adapter_bad_signature_no_arch(tmp_path, monkeypatch):
    body = """
        import dataclasses

        @dataclasses.dataclass
        class StubSpec:
            n: int

        def build_stub(spec: StubSpec):
            return ("kernel", spec)
    """
    src = _write_stub_pkg(tmp_path, body, pkg="stubpkg_noarch")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="builder signature must be"):
        compile_rocke_variant(src, "build_stub", {"n": 1}, ARCH, tmp_path / "co")


@pytest.mark.quick
def test_adapter_bad_signature_two_positional(tmp_path, monkeypatch):
    body = """
        import dataclasses

        @dataclasses.dataclass
        class StubSpec:
            n: int

        def build_stub(spec: StubSpec, extra, *, arch="gfx950"):
            return ("kernel", spec, extra, arch)
    """
    src = _write_stub_pkg(tmp_path, body, pkg="stubpkg_two")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="builder signature must be"):
        compile_rocke_variant(src, "build_stub", {"n": 1}, ARCH, tmp_path / "co")


@pytest.mark.quick
def test_adapter_unexpected_spec_field(tmp_path, monkeypatch):
    src = _write_stub_pkg(tmp_path, _GOOD_STUB, pkg="stubpkg_unexp")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="unexpected spec field"):
        compile_rocke_variant(
            src, "build_stub", {"n": 1, "bogus": 2}, ARCH, tmp_path / "co"
        )


@pytest.mark.quick
def test_adapter_invalid_spec_missing_field(tmp_path, monkeypatch):
    src = _write_stub_pkg(tmp_path, _GOOD_STUB, pkg="stubpkg_miss")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="invalid spec for StubSpec"):
        compile_rocke_variant(src, "build_stub", {"label": "y"}, ARCH, tmp_path / "co")


@pytest.mark.quick
def test_adapter_invalid_spec_post_init(tmp_path, monkeypatch):
    body = """
        import dataclasses

        @dataclasses.dataclass
        class StubSpec:
            n: int

            def __post_init__(self):
                if self.n < 0:
                    raise ValueError("n must be non-negative")

        def build_stub(spec: StubSpec, *, arch="gfx950"):
            return ("kernel", spec, arch)
    """
    src = _write_stub_pkg(tmp_path, body, pkg="stubpkg_pi")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="invalid spec for StubSpec"):
        compile_rocke_variant(src, "build_stub", {"n": -1}, ARCH, tmp_path / "co")


@pytest.mark.quick
def test_adapter_arch_not_supported(tmp_path, monkeypatch):
    body = """
        import dataclasses

        @dataclasses.dataclass
        class StubSpec:
            n: int

        def build_stub(spec: StubSpec, *, arch="gfx950"):
            if arch != "gfx950":
                raise NotImplementedError("stub is gfx950-only")
            return ("kernel", spec, arch)
    """
    src = _write_stub_pkg(tmp_path, body, pkg="stubpkg_arch")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="arch not supported by builder"):
        compile_rocke_variant(src, "build_stub", {"n": 1}, "gfx942", tmp_path / "co")


@pytest.mark.quick
def test_adapter_comgr_failure_wrapped(tmp_path, monkeypatch):
    src = _write_stub_pkg(tmp_path, _GOOD_STUB, pkg="stubpkg_comgr")
    from hkp_pack import rocke_compile

    def _boom(kernel, *, arch, capture_ir_text=False, backend=None):
        raise _FakeComgrError("codegen exploded")

    monkeypatch.setattr(
        rocke_compile, "_load_compiler", lambda: (_boom, _FakeComgrError)
    )
    with pytest.raises(HkpPackError, match="comgr compile failed"):
        compile_rocke_variant(src, "build_stub", {"n": 1}, ARCH, tmp_path / "co")


@pytest.mark.quick
def test_adapter_builder_call_failure_wrapped(tmp_path, monkeypatch):
    body = """
        import dataclasses

        @dataclasses.dataclass
        class StubSpec:
            n: int

        def build_stub(spec: StubSpec, *, arch="gfx950"):
            raise ValueError("builder blew up")
    """
    src = _write_stub_pkg(tmp_path, body, pkg="stubpkg_bcall")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="builder call failed") as excinfo:
        compile_rocke_variant(src, "build_stub", {"n": 1}, ARCH, tmp_path / "co")
    assert "ValueError" in str(excinfo.value)


@pytest.mark.quick
def test_adapter_module_import_raises_wrapped(tmp_path, monkeypatch):
    body = """
        raise ValueError("module blew up at import")
    """
    src = _write_stub_pkg(tmp_path, body, pkg="stubpkg_imperr")
    _patch_compiler(monkeypatch)
    with pytest.raises(HkpPackError, match="module not importable"):
        compile_rocke_variant(src, "build_stub", {"n": 1}, ARCH, tmp_path / "co")


# --- C. rocke variant identity (quick, comgr-free) --------------------------
@pytest.mark.quick
def test_variant_key_builder_distinguishes():
    spec = {"batch": 1, "seqlen_q": 256}
    k1 = rocke_variant_key("kernels/gfx950/attention_dense.py", "build_a", spec)
    k2 = rocke_variant_key("kernels/gfx950/attention_dense.py", "build_b", spec)
    assert k1 != k2


@pytest.mark.quick
def test_variant_key_insertion_order_independent():
    a = rocke_variant_key("s.py", "b", {"x": 1, "y": 2})
    b = rocke_variant_key("s.py", "b", {"y": 2, "x": 1})
    assert a == b


@pytest.mark.quick
def test_variant_key_identical_inputs_same_key():
    args = ("kernels/gfx950/attention_dense.py", "build_attention_dense", {"a": 1})
    assert rocke_variant_key(*args) == rocke_variant_key(*args)


@pytest.mark.quick
def test_variant_key_stem_is_source_stem():
    key = rocke_variant_key("kernels/gfx950/attention_dense.py", "b", {})
    assert key.startswith("attention_dense_")


# --- D. rocke validation accept/reject (quick, comgr-free) ------------------
@pytest.mark.quick
def test_validation_accepts_rocke_ukd(tmp_path, rocke_fixture, rocke_ukd):
    flat = load_flat_input(_copy_fixture(tmp_path, rocke_fixture))
    kdp = next(k for k in flat.kdps() if k.id == "kdp-attention")
    ks = kdp.doc["kernelDescriptors"][0]["kernel_source"]
    assert ks["kind"] == "rocke"
    assert ks["builder"] == rocke_ukd.builder


@pytest.mark.quick
def test_validation_rejects_missing_builder(tmp_path, rocke_fixture):
    src = _copy_fixture(tmp_path, rocke_fixture)
    p = src / "attention.kdp.json"
    doc = _read(p)
    del doc["kernelDescriptors"][0]["kernel_source"]["builder"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="missing required field 'builder'"):
        load_flat_input(src)


@pytest.mark.quick
def test_validation_rejects_missing_spec(tmp_path, rocke_fixture):
    src = _copy_fixture(tmp_path, rocke_fixture)
    p = src / "attention.kdp.json"
    doc = _read(p)
    del doc["kernelDescriptors"][0]["kernel_source"]["spec"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="missing required field 'spec'"):
        load_flat_input(src)


@pytest.mark.quick
def test_validation_rejects_non_object_spec(tmp_path, rocke_fixture):
    src = _copy_fixture(tmp_path, rocke_fixture)
    p = src / "attention.kdp.json"
    doc = _read(p)
    doc["kernelDescriptors"][0]["kernel_source"]["spec"] = "not-an-object"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(HkpPackError, match="invalid spec"):
        load_flat_input(src)


@pytest.mark.quick
def test_validation_standalone_rocke_admitted(tmp_path, rocke_fixture):
    # A standalone rocke UKD file loads (no hip-form force in the standalone path).
    src = _copy_fixture(tmp_path, rocke_fixture)
    kdp_path = src / "attention.kdp.json"
    doc = _read(kdp_path)
    inline = doc["kernelDescriptors"][0]
    standalone_id = inline["id"]
    doc["kernelDescriptors"] = [standalone_id]
    kdp_path.write_text(json.dumps(doc), encoding="utf-8")
    (src / "attention.ukd.json").write_text(json.dumps(inline), encoding="utf-8")
    flat = load_flat_input(src)
    assert standalone_id in flat.ukd_by_id()


# --- E. comgr-gated: real compile, pack, coexistence, arch, symbol ----------
def _kpack_archive(rocm_kpack_dir, out_dir, arch):
    kpack = _load_kpack(rocm_kpack_dir)
    return kpack.PackedKernelArchive.read(
        out_dir / arch / "kpack" / f"hip_kernel_provider_{arch}.kpack"
    )


def test_rocke_compile_variant_real(tmp_path, rocke_available, rocke_ukd):
    from kernels.gfx950.attention_dense import (
        AttentionDenseSpec,
        build_attention_dense,
    )

    co, symbol = compile_rocke_variant(
        rocke_ukd.source,
        rocke_ukd.builder,
        dict(rocke_ukd.spec),
        ARCH,
        tmp_path / "co",
    )
    assert co.is_file() and co.stat().st_size > 0
    expected = build_attention_dense(
        AttentionDenseSpec(**rocke_ukd.spec), arch=ARCH
    ).name
    assert symbol == expected


def test_rocke_compiles_and_packs(
    tmp_path, rocke_fixture, hipcc, rocm_kpack_dir, rocke_available, rocke_ukd
):
    _run(rocke_fixture, tmp_path, hipcc, rocm_kpack_dir)
    ukd = _read(tmp_path / "out" / ARCH / "attention.kdp.json")["kernelDescriptors"][0]
    ks = ukd["kernel_source"]
    assert ks["kind"] == "kpack"
    assert ks["library"] == f"kpack/hip_kernel_provider_{ARCH}.kpack"
    assert "file" not in ks and "build" not in ks
    prov = ukd["provenance"]
    assert prov["origin_kind"] == "rocke"
    assert prov["source"] == rocke_ukd.source
    assert prov["builder"] == rocke_ukd.builder
    assert prov["spec"] == rocke_ukd.spec
    archive = _kpack_archive(rocm_kpack_dir, tmp_path / "out", ARCH)
    blob = archive.get_kernel(ks["toc_key"], ARCH)
    assert blob is not None
    assert hashlib.sha256(blob).hexdigest() == ks["sha256"]
    assert ks["symbol"].encode("ascii") in blob


def test_rocke_symbol_capture_differs_from_builder(
    tmp_path, rocke_fixture, hipcc, rocm_kpack_dir, rocke_available, rocke_ukd
):
    from kernels.gfx950.attention_dense import (
        AttentionDenseSpec,
        build_attention_dense,
    )

    _run(rocke_fixture, tmp_path, hipcc, rocm_kpack_dir)
    ukd = _read(tmp_path / "out" / ARCH / "attention.kdp.json")["kernelDescriptors"][0]
    symbol = ukd["kernel_source"]["symbol"]
    expected = build_attention_dense(
        AttentionDenseSpec(**rocke_ukd.spec), arch=ARCH
    ).name
    assert symbol == expected
    assert symbol != rocke_ukd.builder


def test_rocke_arch_scoping(
    tmp_path, rocke_fixture, hipcc, rocm_kpack_dir, rocke_available
):
    """A gfx950-scoped rocke UKD under a gfx942+gfx950 KDP packs only for gfx950;
    targeting gfx942 excludes the UKD before compile, so the gfx950-only builder
    is never invoked and no gfx942 shard is produced."""
    _run(rocke_fixture, tmp_path, hipcc, rocm_kpack_dir, arches=["gfx942", ARCH])
    # gfx950 shard has the packed rocke UKD.
    assert (tmp_path / "out" / ARCH / "attention.kdp.json").exists()
    ukd = _read(tmp_path / "out" / ARCH / "attention.kdp.json")["kernelDescriptors"][0]
    assert ukd["provenance"]["origin_kind"] == "rocke"
    # gfx942 has no applicable UKD: the shard is skipped entirely.
    assert not (tmp_path / "out" / "gfx942").exists()


# --- comgr self-diagnosing error (quick, comgr-free) ------------------------
@pytest.mark.quick
def test_comgr_error_names_loaded_lib(tmp_path, monkeypatch):
    from hkp_pack import rocke_compile

    base = tmp_path / "diagpkg"
    (base / "sub").mkdir(parents=True)
    (base / "__init__.py").write_text("", encoding="utf-8")
    (base / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (base / "sub" / "mod.py").write_text(
        textwrap.dedent(
            """
            import dataclasses

            @dataclasses.dataclass
            class StubSpec:
                n: int

            def build_stub(spec: StubSpec, *, arch="gfx950"):
                return ("kernel", spec, arch)
            """
        ),
        encoding="utf-8",
    )
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))

    class _FakeComgrError(Exception):
        pass

    def _boom(kernel, *, arch, capture_ir_text=False, backend=None):
        raise _FakeComgrError("codegen exploded")

    monkeypatch.setattr(
        rocke_compile, "_load_compiler", lambda: (_boom, _FakeComgrError)
    )
    monkeypatch.setattr(
        rocke_compile, "_resolved_comgr_path", lambda: "/fake/path/amd_comgr.dll"
    )
    with pytest.raises(HkpPackError, match="comgr loaded from /fake/path") as excinfo:
        rocke_compile.compile_rocke_variant(
            "diagpkg/sub/mod.py",
            "build_stub",
            {"n": 1},
            "gfx950",
            tmp_path / "co",
        )
    assert "ROCKE_COMGR_LIB" in str(excinfo.value)


# --- real-corpus guards (rocke importable, no comgr needed) -----------------
@pytest.mark.quick
def test_real_gfx942_attention_dense_is_refused(rocke_importable):
    """The corpus's one genuine unsuppliable-parameter case must be refused.

    gfx942's build_attention_dense takes a keyword-only
    `tuning: Gfx942DenseTuning = _DEFAULT_TUNING` that no descriptor can set, so
    packing it would silently freeze a performance knob. This asserts against
    the real builder rather than a stub, so the guard cannot rot away from the
    thing it protects.
    """
    from kernels.gfx942 import attention_dense as m

    from hkp_pack.rocke_compile import _require_spec_arch_signature

    with pytest.raises(HkpPackError, match="tuning"):
        _require_spec_arch_signature(m.build_attention_dense, "build_attention_dense")


@pytest.mark.quick
def test_real_gfx942_tiled_2d_is_accepted(rocke_importable):
    """The builder the example descriptor tree uses must pass the gate.

    Pairs with the refusal above: the gate has to be narrow enough that real
    kernels remain packageable, not just strict.
    """
    from kernels.gfx942 import attention_tiled_2d as m

    from hkp_pack.rocke_compile import _require_spec_arch_signature

    _require_spec_arch_signature(
        m.build_unified_attention_2d_tiled, "build_unified_attention_2d_tiled"
    )


@pytest.mark.quick
def test_rocke_toc_key_collision_is_detected(tmp_path, monkeypatch, rocm_kpack_dir):
    """The collision guard must see rocke's real inputs, not (source, build).

    rocke UKDs always have build=None, so a signature of (source, build) is
    identical for every UKD sharing a source module. Two rocke UKDs differing in
    builder or spec would then collide undetected and one would ship the other's
    bytes -- the same silent-substitution class the layout work removed.
    """
    from hkp_pack import pipeline

    src = _write_stub_pkg(tmp_path, _GOOD_STUB, pkg="collidepkg")
    _patch_compiler(monkeypatch)
    # Force both variants onto one key, which is what a real hash collision or a
    # regressed key function would do.
    monkeypatch.setattr(pipeline, "rocke_variant_key", lambda *a, **k: "COLLIDE")

    root = tmp_path / "root"
    root.mkdir()
    kdp = {
        "version": "0.1",
        "id": "kdp-collide",
        "name": "Collide",
        "arch": [ARCH],
        "matchers": [],
        "engine": None,
        "dispatch": None,
        "kernelDescriptors": [
            {
                "version": "0.1",
                "id": f"ukd-collide-{n}",
                "name": f"Collide {n}",
                "kernel_source": {
                    "kind": "rocke",
                    "source": src,
                    "builder": "build_stub",
                    "spec": {"n": n},
                },
                "metadata": {},
                "priority": 0,
            }
            for n in (1, 2)
        ],
    }
    (root / "collide.kdp.json").write_text(json.dumps(kdp), encoding="utf-8")

    with pytest.raises(HkpPackError, match="toc_key collision"):
        pipeline.run_pipeline(
            source_root=root,
            arches=[ARCH],
            out_root=tmp_path / "out",
            hipcc="hipcc-not-invoked",
            rocm_kpack_dir=rocm_kpack_dir,
            inter_root=tmp_path / "inter",
        )
