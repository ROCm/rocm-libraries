"""Multi-launch rocke UKDs: `sources` entries compiled by one or several builders.

Section A stubs the compiler (no comgr): entries naming one (source, builder, spec,
target) compile once and share a code object, entries on different builders get
their own, and the claim check holds the entries to what the compile produced. The
checks that need no rocke are in `test_hkp_pack_sources.py`.
"""

import hashlib
import json
import sys
import textwrap

import pytest

from synthesised_objects import _arg, _kernel, _object, requires_msgpack
from hkp_pack import agreement, desk_check
from hkp_pack.errors import HkpPackError
from hkp_pack.kernel_signature import kernel_signature
from pack_helpers import GENERIC_TARGETS, _run_cli, read_shipped, run_pipeline

ARCH = "gfx1100"

#: Each kernel's own argument list, by name prefix, so a signature read for the
#: wrong symbol cannot match.
ARGS = {
    "toy_scale": [
        _arg("global_buffer", 8, 0, "X"),
        _arg("global_buffer", 8, 8, "Y"),
        _arg("by_value", 4, 16, "N"),
    ],
    "toy_merge": [
        _arg("global_buffer", 8, 0, "A"),
        _arg("global_buffer", 8, 8, "B"),
        _arg("global_buffer", 8, 16, "Out"),
        _arg("by_value", 4, 24, "N"),
    ],
    "toy_seg": [_arg("global_buffer", 8, 0, "S"), _arg("by_value", 4, 8, "K")],
    "toy_red": [_arg("global_buffer", 8, 0, "R")],
    "toy_boom": [_arg("global_buffer", 8, 0, "B")],
}


def _args_for(name):
    return next(args for prefix, args in ARGS.items() if name.startswith(prefix))


def _write_module(tmp_path, pkg, body):
    """An importable `pkg/toy.py` holding `body`; returns its UKD `source`."""
    base = tmp_path / pkg
    base.mkdir()
    (base / "__init__.py").write_text("", encoding="utf-8")
    (base / "toy.py").write_text(textwrap.dedent(body), encoding="utf-8")
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    return f"{pkg}/toy.py"


_STUB_MODULE = """
    import dataclasses

    from rocke.core.ir import KernelDef, Region
    from rocke.helpers import KernelModule

    @dataclasses.dataclass
    class ToySpec:
        tag: str = "a"

    # Two stages whose specs show different fields: the first has `block_size`, the
    # second does not, as the builders of different stages may differ.
    @dataclasses.dataclass
    class StageZeroSpec:
        tag: str = "a"
        head_size: int = 64
        block_size: int = 32

    @dataclasses.dataclass
    class StageOneSpec:
        tag: str = "a"
        head_size: int = 64

    # The support predicates the packer resolves and calls before building.
    def is_valid_spec(spec: ToySpec, *, arch: str):
        return True, ""

    def is_valid_stage_zero(spec: StageZeroSpec, *, arch: str):
        return True, ""

    def is_valid_stage_one(spec: StageOneSpec, *, arch: str):
        return True, ""

    def _kernel(name):
        return KernelDef(name, [], Region("entry"))

    def build_toy(spec: ToySpec, *, arch="gfx1100"):
        return KernelModule(
            primary=_kernel(f"toy_scale_{spec.tag}"),
            companions={"merge": _kernel(f"toy_merge_{spec.tag}")},
        )

    def build_seg(spec: ToySpec, *, arch="gfx1100"):
        return _kernel(f"toy_seg_{spec.tag}")

    def build_red(spec: ToySpec, *, arch="gfx1100"):
        return _kernel(f"toy_red_{spec.tag}")

    def build_stage_zero(spec: StageZeroSpec, *, arch="gfx1100"):
        return _kernel(f"toy_seg_{spec.tag}")

    def build_stage_one(spec: StageOneSpec, *, arch="gfx1100"):
        return _kernel(f"toy_red_{spec.tag}")

    def build_boom(spec: ToySpec, *, arch="gfx1100"):
        raise RuntimeError("no such kernel")

    def build_single(spec: ToySpec, *, arch="gfx1100"):
        return _kernel(f"toy_scale_{spec.tag}")
"""


class _FakeArtifact:
    def __init__(self, name, companion_names, data):
        self.kernel_name = name
        self.companion_names = companion_names
        self.hsaco = data


class _FakeComgrError(Exception):
    pass


def _patch_compilers(monkeypatch):
    """Stub both compile entrypoints, recording each call as `(entrypoint, kernels)`."""
    from rocke.helpers import KernelModule

    from hkp_pack import rocke_compile

    seen = []

    def _object_for(names):
        requires_msgpack()
        return _object([_kernel(n, _args_for(n)) for n in names])

    def _fake_kernel(kernel, *, arch, capture_ir_text=False, backend=None):
        seen.append(("kernel", (kernel.name,)))
        return _FakeArtifact(kernel.name, {}, _object_for([kernel.name]))

    def _fake_module(module, *, arch, capture_ir_text=False, backend=None):
        names = {role: k.name for role, k in module.companions.items()}
        seen.append(("module", (module.primary.name, *names.values())))
        return _FakeArtifact(
            module.primary.name,
            names,
            _object_for([module.primary.name, *names.values()]),
        )

    monkeypatch.setattr(
        rocke_compile, "_load_compiler", lambda: (_fake_kernel, _FakeComgrError)
    )
    monkeypatch.setattr(
        rocke_compile, "_load_module_compiler", lambda: (KernelModule, _fake_module)
    )
    return seen


@pytest.fixture(autouse=True)
def _serial_prewarm(monkeypatch):
    """The stub compilers live in this process, not in a prewarm worker."""
    monkeypatch.setenv("HKP_PACK_JOBS", "1")


def _entry(source, builder, role=None, tag="a", **extra):
    ks = {
        "kind": "rocke",
        "source": source,
        "builder": builder,
        "spec": {"tag": tag},
    }
    if role is not None:
        ks["role"] = role
    ks.update(extra)
    return ks


UED_ID = "5b0f2c1e-3a47-5d2b-9f0e-6c1d8a7b4e21"
KMD_ID = "9d3e7a52-1c6b-5f48-8a0d-2e4b6c8f1a37"
UHD_ID = "c4a81f06-7d2e-5b93-a1c5-0f3e9d6b2a58"


def _root(tmp_path, ukd_kernel, *, version="0.1", compiled=None):
    """A source root with one rocke UKD consumed by one engine.

    `ukd_kernel` is `{"kernel_source": ...}` or `{"sources": {...}}`. `compiled` is
    `{field: metadata value}` for integer fields the consumer's declaration binds
    to the spec attribute of the same name, so the compile must show them.
    """
    compiled = compiled or {}
    ukd = {
        "version": version,
        "id": "ukd-toy",
        "name": "Toy kernel",
        **ukd_kernel,
        "metadata": {"family": "toy", **compiled},
        "priority": 0,
        "provenance": {
            "specialization_contract": {
                "schema_version": 1,
                "consumers": [
                    {
                        "engine_id": UED_ID,
                        "kmd_id": KMD_ID,
                        "metadata_fields": list(compiled),
                        "matcher_only_fields": ["family"],
                        "bindings": {name: {"field": name} for name in compiled},
                        "vocabulary": {},
                    }
                ],
            }
        },
    }
    docs = {
        "kmd": {
            "version": "1.0",
            "id": KMD_ID,
            "name": "Toy metadata",
            "fields": [{"name": "family", "type": "string"}]
            + [{"name": name, "type": "int"} for name in compiled],
        },
        "ued": {
            "version": "1.0",
            "id": UED_ID,
            "name": "test_fixture:toy",
            "sort_kernel_catalog": {"default": UHD_ID},
            "metadata": KMD_ID,
        },
        "uhd": {
            "version": "1.0",
            "id": UHD_ID,
            "name": "Toy heuristic",
            "adapter": "native",
            "objective": "max",
            "native": {"symbol": "hipkernel.toy.score"},
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
            "engine": UED_ID,
            "dispatch": "udd-toy",
            "kernelDescriptors": [ukd],
        },
    }
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


def _blobs(tmp_path, rocm_kpack_dir, ukd):
    """Each launch's code object, read back out of the packed archive."""
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(
        tmp_path / "out" / ARCH / "kpack" / f"hip_kernel_provider_{ARCH}.kpack"
    )
    return {
        name: archive.get_kernel(ks["toc_key"], ARCH)
        for name, ks in ukd["sources"].items()
    }


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


def _restamp(doc):
    binding = agreement.descriptor_binding(doc)
    for evidence in doc["provenance"]["effective_spec"].values():
        evidence["descriptor_digest"] = binding


def _module(tmp_path, name):
    # Unique per test: `sys.modules` would otherwise serve one test's module, and
    # its origin file, to the next.
    return _write_module(tmp_path, f"{name}_{tmp_path.name}", _STUB_MODULE)


# --- A. single-launch output is unchanged -------------------------------------
@pytest.mark.quick
@pytest.mark.parametrize("authored", ["0.1", "1.0", "1.2"])
def test_a_single_kernel_ukd_ships_exactly_the_historical_shape(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir, authored
):
    """No `sources` means no new key anywhere and the authored version untouched,
    so the bytes a single-kernel UKD ships are what they were before `sources`."""
    src = _module(tmp_path, "toymod_single")
    seen = _patch_compilers(monkeypatch)
    root = _root(
        tmp_path,
        {"kernel_source": _entry(src, "build_single")},
        version=authored,
    )
    _pack(root, tmp_path, rocm_kpack_dir)

    shipped, ukd = _shipped(tmp_path)
    assert seen == [("kernel", ("toy_scale_a",))]
    assert ukd["version"] == authored
    assert "sources" not in ukd
    assert set(ukd) == {
        "id",
        "name",
        "kernel_source",
        "metadata",
        "priority",
        "version",
        "arch",
        "provenance",
    }
    assert set(ukd["kernel_source"]) == {
        "kind",
        "library",
        "toc_key",
        "symbol",
        "sha256",
        "signature",
    }
    provenance = ukd["provenance"]
    assert {"source", "builder", "spec", "compile_target"} <= set(provenance)
    assert "sources" not in provenance
    evidence = provenance["effective_spec"]
    assert set(evidence) == {
        "schema_version",
        "authored_digest",
        "observations",
        "consumers",
        "descriptor_digest",
    }
    assert set(evidence["observations"]) == {
        "arch",
        "code_object_sha256",
        "compile_target",
        "producer",
        "requests",
        "symbol",
    }
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    archive = kpack.PackedKernelArchive.read(
        tmp_path / "out" / ARCH / "kpack" / f"hip_kernel_provider_{ARCH}.kpack"
    )
    blob = archive.get_kernel(ukd["kernel_source"]["toc_key"], ARCH)
    agreement.verify(ukd, _reader_records(tmp_path, shipped, ukd), bytes(blob))


# --- B. entries on one module compile once -------------------------------------
@pytest.mark.quick
@pytest.mark.parametrize("authored", ["0.1", "1.0", "1.1", "1.2"])
def test_one_module_two_entries_compile_once_and_share_the_object(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir, authored
):
    src = _module(tmp_path, "toymod_one")
    seen = _patch_compilers(monkeypatch)
    root = _root(
        tmp_path,
        {
            "sources": {
                "decode": _entry(src, "build_toy"),
                "merge": _entry(src, "build_toy", role="merge"),
            }
        },
        version=authored,
    )
    _pack(root, tmp_path, rocm_kpack_dir)

    # One compile, through the module entrypoint, for two entries.
    assert seen == [("module", ("toy_scale_a", "toy_merge_a"))]
    shipped, ukd = _shipped(tmp_path)
    assert "kernel_source" not in ukd
    major, minor = (int(p) for p in authored.split("."))
    assert ukd["version"] == (authored if (major, minor) >= (1, 1) else "1.1")
    decode, merge = ukd["sources"]["decode"], ukd["sources"]["merge"]
    assert (decode["symbol"], merge["symbol"]) == ("toy_scale_a", "toy_merge_a")
    assert (decode["library"], decode["toc_key"], decode["sha256"]) == (
        merge["library"],
        merge["toc_key"],
        merge["sha256"],
    )
    assert all("role" not in ks and ks["kind"] == "kpack" for ks in (decode, merge))
    blobs = _blobs(tmp_path, rocm_kpack_dir, ukd)
    assert hashlib.sha256(blobs["decode"]).hexdigest() == decode["sha256"]
    # Each signature is read off the object for its own symbol.
    assert decode["signature"] == kernel_signature(
        bytes(blobs["decode"]), "toy_scale_a", "t"
    )
    assert merge["signature"] == kernel_signature(
        bytes(blobs["merge"]), "toy_merge_a", "t"
    )
    assert decode["signature"] != merge["signature"]
    # Entries sharing a compile carry identical evidence, keyed by launch name.
    evidence = ukd["provenance"]["effective_spec"]
    assert set(evidence) == {"decode", "merge"}
    assert evidence["decode"] == evidence["merge"]
    assert evidence["decode"]["observations"]["companions"] == {"merge": "toy_merge_a"}
    assert "role" not in json.dumps(ukd["sources"])
    payload = {name: bytes(blob) for name, blob in blobs.items()}
    agreement.verify(ukd, _reader_records(tmp_path, shipped, ukd), payload)


@pytest.mark.quick
def test_entries_differing_in_spec_compile_separately(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """The dedupe key is the whole (source, builder, spec, target), not the builder."""
    src = _module(tmp_path, "toymod_spec")
    seen = _patch_compilers(monkeypatch)
    root = _root(
        tmp_path,
        {
            "sources": {
                "k0": _entry(src, "build_single", tag="a"),
                "k1": _entry(src, "build_single", tag="b"),
            }
        },
    )
    _pack(root, tmp_path, rocm_kpack_dir)
    assert sorted(seen) == [("kernel", ("toy_scale_a",)), ("kernel", ("toy_scale_b",))]
    _shipped_doc, ukd = _shipped(tmp_path)
    assert ukd["sources"]["k0"]["toc_key"] != ukd["sources"]["k1"]["toc_key"]


# --- C. two builders, two entries ---------------------------------------------
@pytest.mark.quick
def test_two_builders_ship_two_entries_with_their_own_objects_and_evidence(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    src = _module(tmp_path, "toymod_two")
    seen = _patch_compilers(monkeypatch)
    root = _root(
        tmp_path,
        {
            "sources": {
                "segment": _entry(src, "build_seg"),
                "reduce": _entry(src, "build_red"),
            }
        },
    )
    _pack(root, tmp_path, rocm_kpack_dir)

    assert sorted(seen) == [("kernel", ("toy_red_a",)), ("kernel", ("toy_seg_a",))]
    shipped, ukd = _shipped(tmp_path)
    seg, red = ukd["sources"]["segment"], ukd["sources"]["reduce"]
    assert (seg["symbol"], red["symbol"]) == ("toy_seg_a", "toy_red_a")
    assert seg["toc_key"] != red["toc_key"] and seg["sha256"] != red["sha256"]
    assert seg["library"] == red["library"]
    assert ukd["version"] == "1.1"
    provenance = ukd["provenance"]
    assert set(provenance["sources"]) == {"segment", "reduce"}
    assert provenance["sources"]["segment"]["builder"] == "build_seg"
    assert provenance["sources"]["reduce"]["builder"] == "build_red"
    assert "builder" not in provenance and "spec" not in provenance
    evidence = provenance["effective_spec"]
    assert evidence["segment"] != evidence["reduce"]
    assert evidence["segment"]["observations"]["symbol"] == "toy_seg_a"
    assert evidence["reduce"]["observations"]["symbol"] == "toy_red_a"
    blobs = {n: bytes(b) for n, b in _blobs(tmp_path, rocm_kpack_dir, ukd).items()}
    records = _reader_records(tmp_path, shipped, ukd)
    agreement.verify(ukd, records, blobs)

    # An entry bound to the other builder's object, signature and symbol included.
    swapped = json.loads(json.dumps(ukd))
    swapped["sources"]["segment"], swapped["sources"]["reduce"] = (
        swapped["sources"]["reduce"],
        swapped["sources"]["segment"],
    )
    _restamp(swapped)
    with pytest.raises(HkpPackError, match="launch 'segment'"):
        agreement.verify(swapped, records, blobs)


@pytest.mark.quick
def test_a_packed_tree_passes_the_compiled_agreement_reader(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """`compiled_agreement` reads each launch's object from its own library and
    toc_key and verifies it; it neither skips a `sources` UKD nor refuses it."""
    src = _module(tmp_path, "toymod_reader")
    _patch_compilers(monkeypatch)
    root = _root(
        tmp_path,
        {
            "sources": {
                "decode": _entry(src, "build_toy"),
                "merge": _entry(src, "build_toy", role="merge"),
            }
        },
    )
    _pack(root, tmp_path, rocm_kpack_dir)
    kdp = tmp_path / "out" / ARCH / "toy.kdp.json"
    assert desk_check.compiled_agreement(kdp, GENERIC_TARGETS, rocm_kpack_dir) == (
        [],
        [],
        1,
    )
    # The archive is gone: the reader fails the kernel, naming the launch.
    (tmp_path / "out" / ARCH / "kpack" / f"hip_kernel_provider_{ARCH}.kpack").unlink()
    failures, _unclaimed, verified = desk_check.compiled_agreement(
        kdp, GENERIC_TARGETS, rocm_kpack_dir
    )
    assert verified == 0
    assert len(failures) == 1 and "launch 'decode'" in failures[0]


# --- D. what the compile produced must be what the entries claim --------------
@pytest.mark.quick
@pytest.mark.parametrize(
    "entries, claimed, compiled",
    [
        # The module has `merge`; the entry names `reduce`.
        (
            lambda s: {
                "decode": _entry(s, "build_toy"),
                "merge": _entry(s, "build_toy", role="reduce"),
            },
            r"\['\(primary\)', 'reduce'\]",
            r"\['\(primary\)', 'merge'\]",
        ),
        # The module's `merge` is compiled and no entry claims it.
        (
            lambda s: {
                "decode": _entry(s, "build_toy"),
                "other": _entry(s, "build_seg"),
            },
            r"\['\(primary\)'\]",
            r"\['\(primary\)', 'merge'\]",
        ),
        # The module's primary is compiled and no entry claims it.
        (
            lambda s: {
                "merge": _entry(s, "build_toy", role="merge"),
                "other": _entry(s, "build_seg"),
            },
            r"\['merge'\]",
            r"\['\(primary\)', 'merge'\]",
        ),
        # A single-kernel builder has no companion to claim.
        (
            lambda s: {
                "decode": _entry(s, "build_seg"),
                "merge": _entry(s, "build_seg", role="merge"),
            },
            r"\['\(primary\)', 'merge'\]",
            r"\['\(primary\)'\]",
        ),
        # Two entries on the companion and none on the primary.
        (
            lambda s: {
                "a": _entry(s, "build_toy", role="merge"),
                "b": _entry(s, "build_toy", role="merge"),
            },
            r"\['merge'\]",
            r"\['\(primary\)', 'merge'\]",
        ),
    ],
    ids=[
        "renamed-role",
        "unclaimed-companion",
        "unclaimed-primary",
        "role-on-a-single-kernel-builder",
        "companion-twice",
    ],
)
def test_the_claimed_kernels_must_equal_the_compiled_kernels(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir, entries, claimed, compiled
):
    src = _module(tmp_path, "toymod_claim")
    _patch_compilers(monkeypatch)
    root = _root(tmp_path, {"sources": entries(src)})
    with pytest.raises(HkpPackError) as info:
        _pack(root, tmp_path, rocm_kpack_dir)
    message = str(info.value)
    assert "claim the kernels" in message
    # Both lists, in the order claimed then compiled.
    import re

    assert re.search(
        rf"claim the kernels {claimed} of builder '\w+', "
        rf"but the compile produced {compiled}",
        message,
    ), message


@pytest.mark.quick
@pytest.mark.parametrize(
    "builder, roles",
    [("build_seg", (None, None)), ("build_toy", (None, None, "merge"))],
    ids=["single-kernel-builder", "with-a-companion"],
)
def test_one_kernel_under_two_launch_names_packs_and_passes_the_full_check(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir, builder, roles
):
    """A pipeline stage may run one kernel twice with different arguments: the
    entries share one compile, name one symbol, and the shipped tree is verified
    by the full desk check."""
    src = _module(tmp_path, "toymod_twice")
    seen = _patch_compilers(monkeypatch)
    entries = {
        f"launch{i}": _entry(src, builder, role=role) for i, role in enumerate(roles)
    }
    _pack(_root(tmp_path, {"sources": entries}), tmp_path, rocm_kpack_dir)
    assert len(seen) == 1
    _shipped_doc, ukd = _shipped(tmp_path)
    packed = ukd["sources"]
    assert set(packed) == set(entries)
    assert packed["launch0"] == packed["launch1"]
    kdp = tmp_path / "out" / ARCH / "toy.kdp.json"
    assert desk_check.compiled_agreement(kdp, GENERIC_TARGETS, rocm_kpack_dir) == (
        [],
        [],
        1,
    )
    result = _run_cli(str(kdp), "--kpack-python-dir", rocm_kpack_dir, mode="full")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.quick
def test_a_failed_claim_leaves_no_shipped_descriptor(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    src = _module(tmp_path, "toymod_none")
    _patch_compilers(monkeypatch)
    root = _root(
        tmp_path,
        {
            "sources": {
                "decode": _entry(src, "build_toy"),
                "merge": _entry(src, "build_toy", role="reduce"),
            }
        },
    )
    with pytest.raises(HkpPackError):
        _pack(root, tmp_path, rocm_kpack_dir)
    assert not (tmp_path / "out" / ARCH / "toy.kdp.json").exists()


# --- E. authored validation reaches the packer --------------------------------
@pytest.mark.quick
@pytest.mark.parametrize(
    "kernel, diagnostic",
    [
        (lambda s: {}, "missing required field 'kernel_source'"),
        (
            lambda s: {
                "kernel_source": _entry(s, "build_single"),
                "sources": {"a": _entry(s, "build_toy"), "b": _entry(s, "build_seg")},
            },
            "both 'kernel_source' and 'sources'",
        ),
        (
            lambda s: {"sources": {"only": _entry(s, "build_toy")}},
            "at least two launches",
        ),
        (
            lambda s: {
                "sources": {"K0": _entry(s, "build_toy"), "k1": _entry(s, "build_seg")}
            },
            "invalid launch name",
        ),
        (
            lambda s: {
                "sources": {
                    "k0": _entry(s, "build_toy"),
                    "k1": {"kind": "hsaco", "file": "x.co", "symbol": "s"},
                }
            },
            "'k1' has kind 'hsaco'",
        ),
        (
            lambda s: {
                "sources": {
                    "k0": _entry(s, "build_toy"),
                    "k1": _entry(s, "build_toy", role="Merge"),
                }
            },
            "invalid role",
        ),
    ],
    ids=["neither", "both", "single", "bad-name", "hsaco-entry", "bad-role"],
)
def test_malformed_authored_sources_fail_the_pack(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir, kernel, diagnostic
):
    src = _module(tmp_path, "toymod_auth")
    seen = _patch_compilers(monkeypatch)
    root = _root(tmp_path, kernel(src))
    with pytest.raises(HkpPackError, match=diagnostic):
        _pack(root, tmp_path, rocm_kpack_dir)
    assert seen == []


# --- F. a shipped document tampered after packing -----------------------------
def _packed_one_module(tmp_path, monkeypatch, rocm_kpack_dir):
    src = _module(tmp_path, "toymod_tamper")
    _patch_compilers(monkeypatch)
    root = _root(
        tmp_path,
        {
            "sources": {
                "decode": _entry(src, "build_toy"),
                "merge": _entry(src, "build_toy", role="merge"),
            }
        },
    )
    _pack(root, tmp_path, rocm_kpack_dir)
    shipped, ukd = _shipped(tmp_path)
    blobs = {n: bytes(b) for n, b in _blobs(tmp_path, rocm_kpack_dir, ukd).items()}
    return ukd, _reader_records(tmp_path, shipped, ukd), blobs


@pytest.mark.quick
def test_a_swapped_symbol_across_entries_fails_verification(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    ukd, records, blobs = _packed_one_module(tmp_path, monkeypatch, rocm_kpack_dir)
    a, b = ukd["sources"]["decode"], ukd["sources"]["merge"]
    a["symbol"], b["symbol"] = b["symbol"], a["symbol"]
    a["signature"], b["signature"] = b["signature"], a["signature"]
    _restamp(ukd)
    with pytest.raises(HkpPackError, match="symbol binding mismatch"):
        agreement.verify(ukd, records, blobs)


@pytest.mark.quick
def test_a_signature_disagreeing_with_the_object_fails_verification(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    ukd, records, blobs = _packed_one_module(tmp_path, monkeypatch, rocm_kpack_dir)
    ukd["sources"]["merge"]["signature"] = ukd["sources"]["decode"]["signature"]
    _restamp(ukd)
    with pytest.raises(HkpPackError, match="launch 'merge': signature does not match"):
        agreement.verify(ukd, records, blobs)


@pytest.mark.quick
@pytest.mark.parametrize("where", ["sources", "authored", "evidence"])
def test_a_dropped_entry_fails_verification(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir, where
):
    ukd, records, blobs = _packed_one_module(tmp_path, monkeypatch, rocm_kpack_dir)
    if where == "sources":
        del ukd["sources"]["merge"]
    elif where == "authored":
        del ukd["provenance"]["sources"]["merge"]
    else:
        del ukd["provenance"]["effective_spec"]["merge"]
    _restamp(ukd)
    with pytest.raises(HkpPackError, match="do not match the UKD's launches"):
        agreement.verify(ukd, records, blobs)


@pytest.mark.quick
def test_the_role_cannot_be_dropped_to_rebind_an_entry_to_the_primary(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """Moving `merge` onto the primary's symbol is caught only through the role the
    provenance records for it."""
    ukd, records, blobs = _packed_one_module(tmp_path, monkeypatch, rocm_kpack_dir)
    ukd["sources"]["merge"]["symbol"] = ukd["sources"]["decode"]["symbol"]
    ukd["sources"]["merge"]["signature"] = ukd["sources"]["decode"]["signature"]
    _restamp(ukd)
    with pytest.raises(HkpPackError, match="launch 'merge': .*symbol binding"):
        agreement.verify(ukd, records, blobs)


# --- G. review findings: single kernel_source, launch naming ----------------------
@pytest.mark.quick
def test_a_plain_kernel_source_naming_a_module_builder_fails_the_pack(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """The module's companion would be compiled and recorded in the evidence, but a
    `kernel_source` has no way to claim it: that is what `sources` is for."""
    src = _module(tmp_path, "toymod_plain")
    _patch_compilers(monkeypatch)
    root = _root(tmp_path, {"kernel_source": _entry(src, "build_toy")})
    with pytest.raises(HkpPackError) as info:
        _pack(root, tmp_path, rocm_kpack_dir)
    message = str(info.value)
    assert "launches ['kernel_source'] claim the kernels ['(primary)']" in message
    assert "but the compile produced ['(primary)', 'merge']" in message
    assert not (tmp_path / "out" / ARCH / "toy.kdp.json").exists()


# --- H. fields a launch's builder does not show --------------------------------
def _spec_entry(src, builder, **spec):
    entry = _entry(src, builder)
    entry["spec"] = {"tag": "a", **spec}
    return entry


def _stages(src, zero=None, one=None):
    """Two launches on builders with different specs: stage zero shows
    `head_size` and `block_size`, stage one shows `head_size` only."""
    return {
        "sources": {
            "zero": _spec_entry(src, "build_stage_zero", **(zero or {})),
            "one": _spec_entry(src, "build_stage_one", **(one or {})),
        }
    }


COMPILED = {"head_size": 64, "block_size": 32}


@pytest.mark.quick
def test_a_field_only_one_launch_reports_passes_when_it_matches(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """`block_size` is on stage zero's spec and not stage one's. Stage one simply
    does not report it; stage zero does, and agrees. That is the case the rule is
    for."""
    src = _module(tmp_path, "toymod_partial")
    _patch_compilers(monkeypatch)
    root = _root(tmp_path, _stages(src), compiled=COMPILED)
    _pack(root, tmp_path, rocm_kpack_dir)
    shipped, ukd = _shipped(tmp_path)
    evidence = ukd["provenance"]["effective_spec"]
    (zero,) = evidence["zero"]["observations"]["requests"].values()
    (one,) = evidence["one"]["observations"]["requests"].values()
    assert zero["values"] == {"head_size": 64, "block_size": 32}
    assert "unreported" not in zero
    assert one["values"] == {"head_size": 64}
    assert one["unreported"] == ["block_size"]
    blobs = {n: bytes(b) for n, b in _blobs(tmp_path, rocm_kpack_dir, ukd).items()}
    agreement.verify(ukd, _reader_records(tmp_path, shipped, ukd), blobs)


@pytest.mark.quick
def test_a_launch_reporting_a_value_the_metadata_does_not_hold_fails(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """Stage zero reports block_size 16 while the metadata says 32."""
    src = _module(tmp_path, "toymod_wrong")
    _patch_compilers(monkeypatch)
    root = _root(tmp_path, _stages(src, zero={"block_size": 16}), compiled=COMPILED)
    with pytest.raises(HkpPackError) as info:
        _pack(root, tmp_path, rocm_kpack_dir)
    message = str(info.value)
    assert "launch 'zero': compiled specialization disagrees with metadata" in message
    assert "block_size observed=16, metadata=32" in message


@pytest.mark.quick
def test_a_field_no_launch_reports_fails_naming_it_and_the_launches(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """Both launches are on the builder without `block_size`."""
    src = _module(tmp_path, "toymod_none_report")
    _patch_compilers(monkeypatch)
    kernel = {
        "sources": {
            "zero": _spec_entry(src, "build_stage_one", tag="a"),
            "one": _spec_entry(src, "build_stage_one", tag="b"),
        }
    }
    root = _root(tmp_path, kernel, compiled=COMPILED)
    with pytest.raises(HkpPackError) as info:
        _pack(root, tmp_path, rocm_kpack_dir)
    assert "field(s) ['block_size'] reported by no launch" in str(info.value)
    assert "checked launches ['one', 'zero']" in str(info.value)


@pytest.mark.quick
def test_a_misspelled_binding_is_missing_on_every_launch_and_fails(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """The binding is shared, so a typo cannot be excused by one stage's spec."""
    src = _module(tmp_path, "toymod_typo")
    _patch_compilers(monkeypatch)
    root = _root(tmp_path, _stages(src), compiled={"head_sizee": 64})
    with pytest.raises(HkpPackError) as info:
        _pack(root, tmp_path, rocm_kpack_dir)
    assert "field(s) ['head_sizee'] reported by no launch" in str(info.value)


@pytest.mark.quick
def test_a_field_reported_on_both_launches_must_match_on_each(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """head_size is reported on both; stage one's differs from the metadata."""
    src = _module(tmp_path, "toymod_both")
    _patch_compilers(monkeypatch)
    root = _root(tmp_path, _stages(src, one={"head_size": 128}), compiled=COMPILED)
    with pytest.raises(HkpPackError) as info:
        _pack(root, tmp_path, rocm_kpack_dir)
    assert "launch 'one': compiled specialization disagrees" in str(info.value)
    assert "head_size observed=128, metadata=64" in str(info.value)


@pytest.mark.quick
def test_a_single_launch_must_still_report_every_declared_field(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """Unchanged for a plain kernel_source: the spec lacks `block_size`."""
    src = _module(tmp_path, "toymod_strict")
    _patch_compilers(monkeypatch)
    root = _root(
        tmp_path,
        {"kernel_source": _spec_entry(src, "build_stage_one")},
        compiled=COMPILED,
    )
    with pytest.raises(HkpPackError, match="cannot observe block_size via"):
        _pack(root, tmp_path, rocm_kpack_dir)


def _packed_partial(tmp_path, monkeypatch, rocm_kpack_dir):
    src = _module(tmp_path, "toymod_reports")
    _patch_compilers(monkeypatch)
    _pack(_root(tmp_path, _stages(src), compiled=COMPILED), tmp_path, rocm_kpack_dir)
    shipped, ukd = _shipped(tmp_path)
    blobs = {n: bytes(b) for n, b in _blobs(tmp_path, rocm_kpack_dir, ukd).items()}
    ukd = json.loads(json.dumps(ukd))
    return ukd, _reader_records(tmp_path, shipped, ukd), blobs


def _report(ukd, launch):
    (entry,) = ukd["provenance"]["effective_spec"][launch]["observations"][
        "requests"
    ].values()
    return entry


@pytest.mark.quick
def test_verify_refuses_a_field_whose_only_report_is_removed(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    ukd, records, blobs = _packed_partial(tmp_path, monkeypatch, rocm_kpack_dir)
    agreement.verify(ukd, records, blobs)
    zero = _report(ukd, "zero")
    del zero["values"]["block_size"]
    zero["unreported"] = ["block_size"]
    with pytest.raises(HkpPackError) as info:
        agreement.verify(ukd, records, blobs)
    assert "field(s) ['block_size'] reported by no launch" in str(info.value)


@pytest.mark.quick
def test_verify_refuses_a_tampered_report(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    ukd, records, blobs = _packed_partial(tmp_path, monkeypatch, rocm_kpack_dir)
    _report(ukd, "zero")["values"]["block_size"] = 16
    with pytest.raises(HkpPackError, match="launch 'zero': compiled specialization"):
        agreement.verify(ukd, records, blobs)


@pytest.mark.quick
def test_verify_refuses_a_report_removed_without_being_accounted_for(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """Dropping a report out of the record must not hide a mismatch: the launch has
    to account for every declared field."""
    ukd, records, blobs = _packed_partial(tmp_path, monkeypatch, rocm_kpack_dir)
    del _report(ukd, "one")["values"]["head_size"]
    with pytest.raises(HkpPackError, match="launch 'one': evidence accounts for"):
        agreement.verify(ukd, records, blobs)


@pytest.mark.quick
def test_verify_refuses_a_field_both_reported_and_unreported(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    ukd, records, blobs = _packed_partial(tmp_path, monkeypatch, rocm_kpack_dir)
    _report(ukd, "zero")["unreported"] = ["block_size"]
    with pytest.raises(HkpPackError, match="launch 'zero': evidence accounts for"):
        agreement.verify(ukd, records, blobs)


class _InProcessPool:
    """A pool stand-in running jobs here, where the stub compilers are installed."""

    def map(self, fn, jobs, chunksize=1):
        return (fn(job) for job in jobs)

    def shutdown(self, **_kwargs):
        pass


@pytest.mark.quick
@pytest.mark.parametrize("jobs", ["1", "2"], ids=["serial", "prewarmed"])
def test_a_compile_failure_of_a_launch_names_the_launch(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir, jobs
):
    """A real compile failure, not a field a spec lacks: the builder of the second
    launch raises. The serial walk and the concurrent prewarm both name it."""
    from hkp_pack import pipeline

    src = _module(tmp_path, "toymod_boom")
    _patch_compilers(monkeypatch)
    monkeypatch.setenv("HKP_PACK_JOBS", jobs)
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", lambda **_kw: _InProcessPool())
    root = _root(
        tmp_path,
        {
            "sources": {
                "segment": _entry(src, "build_seg"),
                "boom": _entry(src, "build_boom"),
            }
        },
    )
    with pytest.raises(HkpPackError) as info:
        _pack(root, tmp_path, rocm_kpack_dir)
    message = str(info.value)
    assert "launch 'boom'" in message and "launch 'segment'" not in message
    assert "builder call failed" in message


@pytest.mark.quick
def test_dropping_a_launch_and_its_companion_evidence_is_caught_by_the_object(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    """Launch dropped everywhere, `observations.companions` popped, and the digest
    re-stamped: only what the code object itself exports still says `merge` is
    there and unclaimed."""
    ukd, records, blobs = _packed_one_module(tmp_path, monkeypatch, rocm_kpack_dir)
    del ukd["sources"]["merge"]
    del ukd["provenance"]["sources"]["merge"]
    del ukd["provenance"]["effective_spec"]["merge"]
    del blobs["merge"]
    del ukd["provenance"]["effective_spec"]["decode"]["observations"]["companions"]
    _restamp(ukd)
    with pytest.raises(HkpPackError) as info:
        agreement.verify(ukd, records, blobs)
    assert "claim the symbols ['toy_scale_a']" in str(info.value)
    assert "exports ['toy_merge_a', 'toy_scale_a']" in str(info.value)


@pytest.mark.quick
def test_sibling_launches_must_carry_identical_evidence(
    tmp_path, monkeypatch, rocke_importable, rocm_kpack_dir
):
    ukd, records, blobs = _packed_one_module(tmp_path, monkeypatch, rocm_kpack_dir)
    agreement.verify(ukd, records, blobs)
    # `decode` alone loses the companions the module compiled.
    del ukd["provenance"]["effective_spec"]["decode"]["observations"]["companions"]
    _restamp(ukd)
    with pytest.raises(HkpPackError, match="share one code object but carry different"):
        agreement.verify(ukd, records, blobs)
