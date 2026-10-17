"""Companion kernels: a rocke builder emitting several kernels into ONE code object.

The builder returns a `rocke.helpers.KernelModule`; the UKD declares the roles in
`kernel_source.companion_kernels`; the packer writes `<role>_symbol` and
`<role>_signature` metadata read off the compiled object. Section A stubs the
compiler (no comgr); section B compiles two real kernels and reads them back out
of the packed archive.
"""

import copy
import hashlib
import json
import sys
import textwrap

import pytest

from synthesised_objects import _arg, _kernel, _object, requires_msgpack
from hkp_pack import agreement, companions
from hkp_pack.errors import HkpPackError
from hkp_pack.kernel_signature import kernel_signature
from hkp_pack.rocke_compile import compile_rocke_variant
from pack_helpers import read_shipped, run_pipeline

ARCH = "gfx1100"

PRIMARY_ARGS = [
    _arg("global_buffer", 8, 0, "X"),
    _arg("global_buffer", 8, 8, "Y"),
    _arg("by_value", 4, 16, "N"),
    _arg("by_value", 4, 20, "alpha"),
]
MERGE_ARGS = [
    _arg("global_buffer", 8, 0, "A"),
    _arg("global_buffer", 8, 8, "B"),
    _arg("global_buffer", 8, 16, "Out"),
    _arg("by_value", 4, 24, "N"),
]
#: `MERGE_ARGS` as the packer records it in `merge_signature`.
MERGE_SIGNATURE = (
    '[{"kind":"global_buffer","size":8,"offset":0,"name":"A"},'
    '{"kind":"global_buffer","size":8,"offset":8,"name":"B"},'
    '{"kind":"global_buffer","size":8,"offset":16,"name":"Out"},'
    '{"kind":"by_value","size":4,"offset":24,"name":"N"}]'
)


def _write_module(tmp_path, pkg, body):
    """An importable `pkg/toy.py` holding `body`; returns its UKD `source`."""
    base = tmp_path / pkg
    base.mkdir()
    (base / "__init__.py").write_text("", encoding="utf-8")
    (base / "toy.py").write_text(textwrap.dedent(body), encoding="utf-8")
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    return f"{pkg}/toy.py"


# A module of empty kernels: enough for the producer, whose compile is stubbed.
_STUB_MODULE = """
    import dataclasses

    from rocke.core.ir import KernelDef, Region
    from rocke.helpers import KernelModule

    @dataclasses.dataclass
    class ToySpec:
        tag: str = "a"

    # The support predicate the packer resolves and calls before building.
    def is_valid_spec(spec: ToySpec, *, arch: str):
        return True, ""

    def build_toy(spec: ToySpec, *, arch="gfx1100"):
        return KernelModule(
            primary=KernelDef(f"toy_scale_{spec.tag}", [], Region("entry")),
            companions={"merge": KernelDef(f"toy_merge_{spec.tag}", [], Region("entry"))},
        )

    def build_single(spec: ToySpec, *, arch="gfx1100"):
        return KernelDef(f"toy_scale_{spec.tag}", [], Region("entry"))
"""


class _FakeArtifact:
    def __init__(self, name, companion_names, data):
        self.kernel_name = name
        self.companion_names = companion_names
        self.hsaco = data


class _FakeComgrError(Exception):
    pass


def _patch_compilers(monkeypatch, data=None):
    """Stub both compile entrypoints, recording which one the producer chose."""
    from rocke.helpers import KernelModule

    from hkp_pack import rocke_compile

    seen = []

    def _object_for(names):
        requires_msgpack()
        args = [PRIMARY_ARGS, MERGE_ARGS]
        return _object([_kernel(n, a) for n, a in zip(names, args)])

    def _fake_kernel(kernel, *, arch, capture_ir_text=False, backend=None):
        seen.append(("kernel", backend))
        return _FakeArtifact(kernel.name, {}, data or _object_for([kernel.name]))

    def _fake_module(module, *, arch, capture_ir_text=False, backend=None):
        seen.append(("module", backend))
        names = {role: k.name for role, k in module.companions.items()}
        payload = data or _object_for([module.primary.name, *names.values()])
        return _FakeArtifact(module.primary.name, names, payload)

    monkeypatch.setattr(
        rocke_compile, "_load_compiler", lambda: (_fake_kernel, _FakeComgrError)
    )
    monkeypatch.setattr(
        rocke_compile, "_load_module_compiler", lambda: (KernelModule, _fake_module)
    )
    return seen


# --- Authored descriptors: one engine whose KMD declares the merge fields ------
def _root(tmp_path, source, *, builder="build_toy", roles=("merge",), mutate=None):
    """A source root with one rocke UKD consumed by one engine.

    `mutate(docs)` may edit the documents, keyed by suffix, before they are
    written.
    """
    ks = {"kind": "rocke", "source": source, "builder": builder, "spec": {"tag": "a"}}
    if roles:
        ks["companion_kernels"] = list(roles)
    produced = [n for r in roles for n in companions.metadata_fields(r)]
    docs = {
        "kmd": {
            "version": "0.1",
            "id": "kmd-toy",
            "name": "Toy metadata",
            "fields": [{"name": "family", "type": "string"}]
            + [{"name": n, "type": "string"} for n in produced],
        },
        "ued": {
            "version": "0.1",
            "id": "ued-toy",
            "name": "test_fixture:toy",
            "heuristic": "uhd-toy",
            "metadata": "kmd-toy",
        },
        "uhd": {
            "version": "0.1",
            "id": "uhd-toy",
            "name": "Toy heuristic",
            "kind": "native",
            "payload": "hipkernel.toy.score",
        },
        "umd": {
            "version": "0.1",
            "id": "umd-toy",
            "name": "Toy matcher",
            "scope": "kernel",
            "match_symbol": "hipkernel.toy.match",
        },
        "udd": {
            "version": "0.1",
            "id": "udd-toy",
            "name": "Toy dispatch",
            "dispatch_symbol": "hipkernel.toy.dispatch",
        },
        "kdp": {
            "version": "0.1",
            "id": "kdp-toy",
            "name": "Toy pack",
            "arch": [ARCH],
            "matchers": ["umd-toy"],
            "engine": "ued-toy",
            "dispatch": "udd-toy",
            "kernelDescriptors": [
                {
                    "version": "0.1",
                    "id": "ukd-toy",
                    "name": "Toy decode with merge",
                    "kernel_source": ks,
                    "metadata": {"family": "toy"},
                    "priority": 0,
                    "provenance": {
                        "specialization_contract": {
                            "schema_version": 1,
                            "consumers": [
                                {
                                    "engine_id": "ued-toy",
                                    "kmd_id": "kmd-toy",
                                    "metadata_fields": [],
                                    "matcher_only_fields": ["family", *produced],
                                    "bindings": {},
                                    "vocabulary": {},
                                }
                            ],
                        }
                    },
                }
            ],
        },
    }
    if mutate is not None:
        mutate(docs)
    root = tmp_path / "root"
    root.mkdir()
    for suffix, doc in docs.items():
        (root / f"toy.{suffix}.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


def _pack(root, tmp_path, rocm_kpack_dir):
    return run_pipeline(
        source_root=root,
        arches=[ARCH],
        out_root=tmp_path / "out",
        hipcc="hipcc-not-invoked",
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
    )


def _shipped(tmp_path):
    shipped = read_shipped(tmp_path / "out" / ARCH / "toy.kdp.json")
    return shipped, shipped["kernelDescriptors"][0]


def _blob(tmp_path, rocm_kpack_dir, ks):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(
        tmp_path / "out" / ARCH / "kpack" / f"hip_kernel_provider_{ARCH}.kpack"
    )
    return archive.get_kernel(ks["toc_key"], ARCH)


def _reader_records(tmp_path, shipped, ukd):
    """The consumer records a reader rebuilds from the shipped documents alone."""
    root = tmp_path / "root"
    kmd = json.loads((root / "toy.kmd.json").read_text(encoding="utf-8"))
    engine = json.loads((root / "toy.ued.json").read_text(encoding="utf-8"))
    header = {k: v for k, v in shipped.items() if k != "kernelDescriptors"}
    declaration = agreement.select_declaration(
        ukd, engine, kmd, {kmd["id"]: kmd}, shipped
    )
    return agreement.canonical_records(
        [agreement.consumer_record(ukd, engine, kmd, header, ARCH, declaration)]
    )


# --- A. producer and pipeline, compiler stubbed (quick, comgr-free) ------------
@pytest.mark.quick
def test_module_builder_compiles_as_one_object(tmp_path, monkeypatch, rocke_importable):
    src = _write_module(tmp_path, "toymod_a", _STUB_MODULE)
    seen = _patch_compilers(monkeypatch)
    co, symbol, observations = compile_rocke_variant(
        src, "build_toy", {"tag": "a"}, ARCH, tmp_path / "co"
    )
    assert symbol == "toy_scale_a"
    assert observations["symbol"] == "toy_scale_a"
    assert observations["companions"] == {"merge": "toy_merge_a"}
    # One compile, through the module entrypoint, still pinned to Python lowering.
    assert seen == [("module", "python")]
    assert kernel_signature(co.read_bytes(), "toy_merge_a", "t") == json.loads(
        MERGE_SIGNATURE
    )


@pytest.mark.quick
def test_single_kernel_builder_records_no_companions(
    tmp_path, monkeypatch, rocke_importable
):
    """A single-kernel compile's evidence keeps its historical key set."""
    src = _write_module(tmp_path, "toymod_b", _STUB_MODULE)
    seen = _patch_compilers(monkeypatch)
    _co, symbol, observations = compile_rocke_variant(
        src, "build_single", {"tag": "a"}, ARCH, tmp_path / "co"
    )
    assert symbol == "toy_scale_a"
    assert "companions" not in observations
    assert seen == [("kernel", "python")]


@pytest.mark.quick
def test_a_module_compile_error_is_reported_as_raised(
    tmp_path, monkeypatch, rocke_importable
):
    """A ValueError from compile_module before any link is not called a link error."""
    from rocke.helpers import KernelModule

    from hkp_pack import rocke_compile

    src = _write_module(tmp_path, "toymod_f", _STUB_MODULE)
    _patch_compilers(monkeypatch)

    def _unresolvable(module, *, arch, capture_ir_text=False, backend=None):
        raise ValueError(f"no compile target for {arch!r}")

    monkeypatch.setattr(
        rocke_compile, "_load_module_compiler", lambda: (KernelModule, _unresolvable)
    )
    with pytest.raises(HkpPackError) as info:
        compile_rocke_variant(src, "build_toy", {"tag": "a"}, ARCH, tmp_path / "co")
    assert str(info.value) == (
        f"compiling the kernels of 'build_toy' @ {ARCH} into one code object "
        f"failed: no compile target for '{ARCH}'"
    )


@pytest.mark.quick
def test_companions_ship_as_metadata(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    src = _write_module(tmp_path, "toymod_c", _STUB_MODULE)
    _patch_compilers(monkeypatch)
    _pack(_root(tmp_path, src), tmp_path, rocm_kpack_dir)

    shipped, ukd = _shipped(tmp_path)
    ks = ukd["kernel_source"]
    assert ks["symbol"] == "toy_scale_a"
    assert "companion_kernels" not in ks
    assert ukd["metadata"] == {
        "family": "toy",
        "merge_symbol": "toy_merge_a",
        "merge_signature": MERGE_SIGNATURE,
    }
    # The primary's own signature is untouched by its companion.
    assert [a["name"] for a in ks["signature"]] == ["X", "Y", "N", "alpha"]

    blob = _blob(tmp_path, rocm_kpack_dir, ks)
    assert hashlib.sha256(blob).hexdigest() == ks["sha256"]
    evidence = ukd["provenance"]["effective_spec"]
    assert evidence["observations"]["companions"] == {"merge": "toy_merge_a"}
    # The evidence binds the metadata that shipped, as a reader rebuilds it.
    agreement.verify(ukd, _reader_records(tmp_path, shipped, ukd), blob)


def _forge_signature(ukd, observations):
    ukd["metadata"]["merge_signature"] = "[]"


def _forge_symbol(ukd, observations):
    # A symbol the payload does export, under the wrong role.
    ukd["metadata"]["merge_symbol"] = "toy_scale_a"


def _forge_role_map(ukd, observations):
    observations["companions"] = [["merge", "toy_merge_a"]]


@pytest.mark.quick
@pytest.mark.parametrize(
    "forge, diagnostic",
    [
        (_forge_signature, "companion binding mismatch"),
        (_forge_symbol, "companion binding mismatch"),
        (_forge_role_map, "companions are not a role map"),
    ],
    ids=["signature", "symbol", "role-map"],
)
def test_a_forged_companion_fails_verification(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir, forge, diagnostic
):
    src = _write_module(tmp_path, "toymod_d", _STUB_MODULE)
    _patch_compilers(monkeypatch)
    _pack(_root(tmp_path, src), tmp_path, rocm_kpack_dir)
    shipped, ukd = _shipped(tmp_path)
    blob = _blob(tmp_path, rocm_kpack_dir, ukd["kernel_source"])

    # Forge consistently: re-publish the evidence over the forged descriptor, so
    # every binding but the companion one still holds and only that check can
    # refuse it.
    forged = copy.deepcopy(ukd)
    observations = forged["provenance"]["effective_spec"]["observations"]
    forge(forged, observations)
    records = _reader_records(tmp_path, shipped, forged)
    agreement.publish(forged, observations, records)
    with pytest.raises(HkpPackError, match=diagnostic):
        agreement.verify(forged, records, blob)


@pytest.mark.quick
@pytest.mark.parametrize(
    "builder, roles, diagnostic",
    [
        ("build_toy", ("merge", "reduce"), r"declares \['merge', 'reduce'\]"),
        ("build_single", ("merge",), r"emitted companions \[\]"),
        ("build_toy", (), r"declares \[\], but builder"),
    ],
    ids=["undelivered-role", "single-kernel-builder", "undeclared-companion"],
)
def test_declared_roles_must_equal_the_compiled_companions(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir, builder, roles, diagnostic
):
    src = _write_module(tmp_path, f"toymod_e_{builder}_{len(roles)}", _STUB_MODULE)
    _patch_compilers(monkeypatch)
    root = _root(tmp_path, src, builder=builder, roles=roles)
    with pytest.raises(HkpPackError, match=diagnostic):
        _pack(root, tmp_path, rocm_kpack_dir)


def _authored_merge_symbol(docs):
    docs["kdp"]["kernelDescriptors"][0]["metadata"]["merge_symbol"] = "forged"


def _undeclared_signature(docs):
    fields = docs["kmd"]["fields"]
    docs["kmd"]["fields"] = [f for f in fields if f["name"] != "merge_signature"]
    consumer = docs["kdp"]["kernelDescriptors"][0]["provenance"][
        "specialization_contract"
    ]["consumers"][0]
    consumer["matcher_only_fields"].remove("merge_signature")


def _int_symbol(docs):
    for f in docs["kmd"]["fields"]:
        if f["name"] == "merge_symbol":
            f["type"] = "int"


def _bound_symbol(docs):
    consumer = docs["kdp"]["kernelDescriptors"][0]["provenance"][
        "specialization_contract"
    ]["consumers"][0]
    consumer["matcher_only_fields"].remove("merge_symbol")
    consumer["metadata_fields"].append("merge_symbol")
    consumer["bindings"]["merge_symbol"] = {"field": "tag"}


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutate, diagnostic",
    [
        (_authored_merge_symbol, r"authored metadata supplies \['merge_symbol'\]"),
        (
            _undeclared_signature,
            r"declare string metadata field\(s\) \['merge_signature'\]",
        ),
        (_int_symbol, r"declare string metadata field\(s\) \['merge_symbol'\]"),
        (_bound_symbol, r"declare them matcher-only"),
    ],
    ids=["authored", "undeclared", "wrong-type", "spec-bound"],
)
def test_companion_fields_are_the_producers_to_write(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir, mutate, diagnostic
):
    src = _write_module(tmp_path, "toymod_f", _STUB_MODULE)
    seen = _patch_compilers(monkeypatch)
    root = _root(tmp_path, src, mutate=mutate)
    with pytest.raises(HkpPackError, match=diagnostic):
        _pack(root, tmp_path, rocm_kpack_dir)
    assert seen == [], "the authored side is refused before anything compiles"


@pytest.mark.quick
@pytest.mark.parametrize("roles", [[], ["Merge"], ["merge", "merge"], "merge", [1]])
def test_an_unusable_companion_kernels_value_is_rejected_at_load(
    tmp_path, rocke_importable, rocm_kpack_dir, roles
):
    def _set_roles(docs):
        ks = docs["kdp"]["kernelDescriptors"][0]["kernel_source"]
        ks["companion_kernels"] = roles

    root = _root(tmp_path, "toymod_g/toy.py", mutate=_set_roles)
    with pytest.raises(HkpPackError, match="invalid companion_kernels"):
        _pack(root, tmp_path, rocm_kpack_dir)


# --- B. real comgr: two kernels, one packed object ----------------------------
_REAL_MODULE = """
    import dataclasses

    from rocke.core.ir import F32, I32, IRBuilder, PtrType
    from rocke.helpers import KernelModule

    @dataclasses.dataclass
    class ToySpec:
        tag: str = "a"

    def is_valid_spec(spec: ToySpec, *, arch: str):
        return True, ""

    def _start(name):
        b = IRBuilder(name)
        b.kernel.attrs["max_workgroup_size"] = 64
        return b

    def _reverse_through_lds(b, n, value_at, store_at):
        # Each 64-lane block exchanges values through LDS across a barrier,
        # so both kernels keep a real LDS pool through the link.
        smem = b.smem_alloc(F32, [64], name_hint="stage")
        tid = b.thread_id_x()
        base = b.mul(b.block_id_x(), b.const_i32(64))
        idx = b.add(base, tid)
        safe = b.select(b.cmp_lt(idx, n), idx, b.const_i32(0))
        b.smem_store_vN(smem, [tid], value_at(safe), 1)
        b.sync()
        mirror = b.sub(b.const_i32(63), tid)
        out_idx = b.add(base, mirror)
        with b.scf_if(b.cmp_lt(out_idx, n)):
            staged = b.vec_extract(b.smem_load_vN(smem, mirror, dtype=F32, n=1), 0)
            store_at(out_idx, staged)

    def _scale(name):
        b = _start(name)
        ptr = PtrType(F32, "global")
        x = b.param("X", ptr, noalias=True, readonly=True, align=4)
        y = b.param("Y", ptr, noalias=True, writeonly=True, align=4)
        n = b.param("N", I32)
        alpha = b.param("alpha", F32)
        _reverse_through_lds(
            b,
            n,
            lambda i: b.fmul(b.global_load(x, i, F32, align=4), alpha),
            lambda i, v: b.global_store(y, i, v, align=4),
        )
        return b.kernel

    def _merge(name):
        b = _start(name)
        ptr = PtrType(F32, "global")
        a = b.param("A", ptr, noalias=True, readonly=True, align=4)
        c = b.param("B", ptr, noalias=True, readonly=True, align=4)
        out = b.param("Out", ptr, noalias=True, writeonly=True, align=4)
        n = b.param("N", I32)
        _reverse_through_lds(
            b,
            n,
            lambda i: b.fadd(
                b.global_load(a, i, F32, align=4), b.global_load(c, i, F32, align=4)
            ),
            lambda i, v: b.global_store(out, i, v, align=4),
        )
        return b.kernel

    def build_toy(spec: ToySpec, *, arch="gfx1100"):
        return KernelModule(
            primary=_scale(f"toy_scale_{spec.tag}"),
            companions={"merge": _merge(f"toy_merge_{spec.tag}")},
        )
"""


def test_two_real_kernels_pack_into_one_object(
    tmp_path, rocke_available, rocm_kpack_dir
):
    src = _write_module(tmp_path, "toymod_real", _REAL_MODULE)
    _pack(_root(tmp_path, src), tmp_path, rocm_kpack_dir)

    shipped, ukd = _shipped(tmp_path)
    ks = ukd["kernel_source"]
    blob = _blob(tmp_path, rocm_kpack_dir, ks)
    assert ks["symbol"] == "toy_scale_a"
    assert [a["name"] for a in ks["signature"]] == ["X", "Y", "N", "alpha"]
    assert kernel_signature(blob, "toy_scale_a", "t") == ks["signature"]
    assert ukd["metadata"]["merge_symbol"] == "toy_merge_a"
    assert ukd["metadata"]["merge_signature"] == MERGE_SIGNATURE
    assert kernel_signature(blob, "toy_merge_a", "t") == json.loads(MERGE_SIGNATURE)
    agreement.verify(ukd, _reader_records(tmp_path, shipped, ukd), blob)
