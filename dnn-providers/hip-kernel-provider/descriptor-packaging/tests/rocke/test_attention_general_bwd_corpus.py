# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The committed artifacts of hipkernel:AttentionGeneralBwd match rocKE's Python.

The engine's C++ lowering, predicate and plan restate rocKE's backward reference
(``rocke/library/dispatch/attention_bwd/hipdnn.py``). Generated artifacts tie the
halves together: the constants / catalog headers the C++ compiles against, the
replay corpus its tests assert against, and the IngestorGenerator config and
descriptor bundle the kernels ship in. All are regenerated here from the in-tree
rocKE sources and must equal the committed files, so a Python change that moves a
verdict, a kernel symbol or a kernarg fails here rather than as a silent mismatch on a
device. The descriptors must also serve every accepted corpus plan, and every bound
KMD field must agree with the specs hkp_pack hydrates.

The GPU suites' device cases are generated the same way. These are rocKE tests: they
run only in a rocKE-enabled build, where a missing requirement (the rocKE sources,
PyYAML and Jinja2, clang-format) fails rather than skips.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HKP = Path(__file__).resolve().parents[3]
TOOLS = HKP / "descriptor-packaging" / "tools"
KIE = HKP / "src" / "engines" / "kernel_ingestor_engine"
PACKS = KIE / "packs"
BUNDLE = KIE / "descriptors" / "rocKE" / "attention_general_bwd"
FIXTURES = HKP / "src" / "tests" / "fixtures" / "attention_general_bwd"
ARCHES = ("gfx942", "gfx950")
MAX_BYTES = 6 * 1024 * 1024
_NO_CLANG_FORMAT = (
    "clang-format not found: set HKP_CLANG_FORMAT (the ctest entries forward the one "
    "next to the configured Python, else on PATH)"
)


def _env():
    env = dict(os.environ)
    env.setdefault("ROCKE_BACKEND", "python")
    return env


def _run(*args):
    return subprocess.run(
        [sys.executable, *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
        env=_env(),
        cwd=TOOLS,
    )


@pytest.fixture(scope="module")
def rocke_sources():
    probe = _run(
        "-c",
        "import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
        "import dispatch.attention_bwd.hipdnn_catalog",
        HKP / "rocke" / "platform" / "python",
        HKP / "rocke" / "library",
    )
    if probe.returncode != 0:
        pytest.fail(f"rocKE sources not importable: {probe.stderr.strip()[-300:]}")


def _corpus(arch):
    lines = (FIXTURES / f"corpus_{arch}.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0])["meta"], [json.loads(x) for x in lines[1:]]


@pytest.mark.parametrize("arch", ARCHES)
def test_corpus_subset_is_current(rocke_sources, arch):
    out = FIXTURES / f"corpus_{arch}.jsonl"
    r = _run(
        "general_bwd_corpus.py",
        "--arch",
        arch,
        "--mode",
        "subset",
        "--out",
        out,
        "--check",
    )
    assert r.returncode == 0, r.stderr


def test_constants_headers_are_current(rocke_sources):
    sys.path.insert(0, str(TOOLS))
    try:
        from general_bwd_constants import find_clang_format
    finally:
        sys.path.remove(str(TOOLS))
    binary = find_clang_format()
    if binary is None:
        pytest.fail(_NO_CLANG_FORMAT)
    r = _run(
        "general_bwd_constants.py",
        "--out-dir",
        PACKS,
        "--check",
        "--clang-format",
        binary,
    )
    assert r.returncode == 0, r.stderr


def test_gpu_cases_header_is_current(rocke_sources):
    _require_artifacts()
    sys.path.insert(0, str(TOOLS))
    try:
        from general_bwd_constants import find_clang_format
    finally:
        sys.path.remove(str(TOOLS))
    binary = find_clang_format()
    if binary is None:
        pytest.fail(_NO_CLANG_FORMAT)
    r = _run(
        "general_bwd_gpu_cases.py",
        "--out-dir",
        HKP / "src" / "integration_tests" / "kernel_ingestor_engine",
        "--check",
        "--clang-format",
        binary,
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "field, value",
    [("xcdChunk", 4), ("dkvSink", "slices")],
    ids=["xcd_chunk", "dkv_sink"],
)
def test_tuned_record_refuses_a_knob_the_engine_cannot_honour(
    rocke_sources, field, value
):
    sys.path.insert(0, str(TOOLS))
    try:
        from general_bwd_constants import _check_tuned_record
    finally:
        sys.path.remove(str(TOOLS))
    record = {"key": "k", "id": 0, "xcdChunk": 0, "dkvSink": "atomic"}
    _check_tuned_record(record, [])
    with pytest.raises(AssertionError):
        _check_tuned_record({**record, field: value}, [])


@pytest.mark.parametrize("arch", ARCHES)
def test_corpus_shape(arch):
    meta, entries = _corpus(arch)
    assert meta["arch"] == arch and meta["mode"] == "subset"
    assert len(entries) == meta["entries"]
    assert (FIXTURES / f"corpus_{arch}.jsonl").stat().st_size <= MAX_BYTES
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    codes = {}
    for e in entries:
        codes[e["expect"]["code"]] = codes.get(e["expect"]["code"], 0) + 1
        assert e["arch"] == arch and e["num_cus"] > 0
    assert codes.get("OK", 0) > 0
    for code, n in codes.items():
        assert code == "OK" or n >= 10, (code, n)
    assert not set(meta["unreachable"]) & set(codes)
    # every accepted plan launches prep then main, and dQ next
    for e in entries:
        if e["expect"]["code"] != "OK":
            continue
        launches = [(x["slot"], x["launch"]) for x in e["expect"]["launches"]]
        assert launches[:3] == [("prep", ""), ("main", "main"), ("main", "dq")], e["id"]
        assert set(launches[3:]) <= {("main", "convert_dk"), ("main", "convert_dv")}


@pytest.mark.parametrize("arch", ARCHES)
def test_corpus_kernels_are_catalog_kernels(arch):
    """Every kernel a corpus plan launches is a kernel the catalog header ships."""
    catalog = (PACKS / "AttentionGeneralBwdCatalog.generated.hpp").read_text(
        encoding="utf-8"
    )
    _meta, entries = _corpus(arch)
    missing = set()
    for e in entries:
        if e["expect"]["code"] == "OK":
            for launch in e["expect"]["launches"]:
                if f'"{launch["kernel"]}"' not in catalog:
                    missing.add(launch["kernel"])
    assert not missing, sorted(missing)[:10]


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle():
    """``(ued, {slot: kmd}, {pack: kdp})`` of the committed bundle."""
    ued = _json(BUNDLE / "attention_general_bwd.ued.json")
    kmds = {d["id"]: d for d in map(_json, BUNDLE.glob("*.kmd.json"))}
    slots = {s["name"]: kmds[s["metadata"]] for s in ued["slots"]}
    kdps = {}
    for path in [*BUNDLE.glob("*.kdp.json"), *BUNDLE.glob("artifacts/*.kdp.json")]:
        kdps[path.name[len("attention_general_bwd_") : -len(".kdp.json")]] = _json(path)
    return ued, slots, kdps


def _require_artifacts():
    """Skips when the bundle's DVC output (its KDPs) is not pulled
    (descriptors/README.md: a build without a `dvc pull` still runs its tests)."""
    if not any((BUNDLE / "artifacts").glob("*.kdp.json")):
        pytest.skip(
            f"{BUNDLE.name} KDPs not pulled; run: "
            f"dvc pull -r ingestor {BUNDLE / 'artifacts.dvc'}"
        )


def _launch_sources(kernel):
    return kernel.get("sources") or {"": kernel["kernel_source"]}


@pytest.fixture(scope="module")
def generator_deps(rocke_sources):
    probe = _run("-c", "import yaml, jinja2")
    if probe.returncode != 0:
        pytest.fail(f"IngestorGenerator needs PyYAML and Jinja2: {probe.stderr[-200:]}")


def test_catalog_config_and_descriptors_are_current(generator_deps):
    """The config and the bundle regenerate byte for byte (ids reused) and pass
    the structural variant-set gate."""
    r = _run("general_bwd_catalog.py", "--check", "--generate", "--verify")
    assert r.returncode == 0, r.stdout + r.stderr


def test_launch_surface_profile_checks():
    """Every surface of the launch-surface profile names real files, symbols and KMD
    fields."""
    if _run("-c", "import yaml").returncode != 0:
        pytest.fail("tools/launch_surface.py needs PyYAML")
    generator = HKP.parents[1] / "projects" / "hipdnn" / "tools" / "IngestorGenerator"
    r = _run(
        generator / "tools" / "launch_surface.py",
        generator / "configs" / "attention_general_bwd.profile.yaml",
        "--check",
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_bundle_shape():
    _require_artifacts()
    ued, slots, kdps = _bundle()
    assert ued["version"] == "1.1" and ued["name"] == "hipkernel:AttentionGeneralBwd"
    assert ued["sdk_version"] == "1.4.0" and ued["shared_workspace"] == "ws.bytes"
    assert [s["name"] for s in ued["slots"]] == ["prep", "main"]
    expected = {
        "prep_gfx942": ("prep", "gfx942"),
        "prep_gfx950": ("prep", "gfx950"),
        "main_gfx942": ("main", "gfx942"),
        "main_gfx950": ("main", "gfx950"),
        "main_splitdq_gfx950": ("main", "gfx950"),
    }
    assert {p: (k["slot"], k["arch"][0]) for p, k in kdps.items()} == expected
    for pack, kdp in kdps.items():
        assert kdp["version"] == "1.1" and kdp["arch"] == [expected[pack][1]]
        assert kdp["engine"] == ued["id"]
        assert 0 < len(kdp["kernelDescriptors"]) <= 1000, pack
        consumer = kdp["provenance"]["specialization_contract"]["consumers"][0]
        assert consumer["kmd_id"] == slots[kdp["slot"]]["id"]
        for kernel in kdp["kernelDescriptors"]:
            if kdp["slot"] == "prep":
                assert "kernel_source" in kernel and "sources" not in kernel
            else:
                assert list(kernel["sources"]) == [
                    "main",
                    "dq",
                    "convert_dk",
                    "convert_dv",
                ]
                split = kernel["metadata"]["dq_mode"] == "split"
                assert split == (pack == "main_splitdq_gfx950"), kernel["name"]
                dq = kernel["sources"]["dq"]["builder"]
                assert dq == (
                    "build_attn_bwd_dq" if split else "build_attn_bwd_convert"
                )


def _rocke_path():
    for sub in ("rocke/platform/python", "rocke/library"):
        if str(HKP / sub) not in sys.path:
            sys.path.insert(0, str(HKP / sub))


def _hydrate(source):
    import importlib

    from hkp_pack.rocke_compile import (
        _module_from_source,
        _resolve_spec_class,
        build_spec,
    )

    module = importlib.import_module(_module_from_source(source["source"]))
    builder = getattr(module, source["builder"])
    return build_spec(_resolve_spec_class(module, builder), source["spec"]), builder


def _symbol(spec, builder):
    stage = {
        "build_attn_bwd_main": "main",
        "build_attn_bwd_prep": "prep",
        "build_attn_bwd_convert": "convert",
    }.get(builder.__name__)
    return spec.kernel_name() if stage is None else spec.kernel_name(stage)


@pytest.fixture(scope="module")
def hydrated(rocke_sources):
    """``{pack: [(kernel, {launch: (spec, builder)})]}`` of the committed bundle."""
    _require_artifacts()
    _rocke_path()
    os.environ.setdefault("ROCKE_BACKEND", "python")
    _ued, _slots, kdps = _bundle()
    return {
        pack: [
            (k, {n: _hydrate(s) for n, s in _launch_sources(k).items()})
            for k in kdp["kernelDescriptors"]
        ]
        for pack, kdp in kdps.items()
    }


def test_bound_fields_agree_with_the_hydrated_specs(hydrated):
    """What hkp_pack checks at pack time, without compiling: every bound KMD field
    of every kernel is observed off each launch's hydrated spec and agrees."""
    from hkp_pack import agreement

    _ued, slots, kdps = _bundle()
    for pack, kernels in hydrated.items():
        kmd = slots[kdps[pack]["slot"]]
        consumer = kdps[pack]["provenance"]["specialization_contract"]["consumers"][0]
        agreement.validate_consumer(consumer, kmd)
        request = agreement.observation_request(consumer, kmd)
        requests = {agreement.digest(request): request}
        for kernel, launches in kernels:
            observed = {
                name: agreement.observe(
                    spec,
                    builder,
                    requests,
                    agreement.OriginObserver(),
                    tolerate_missing=True,
                )
                for name, (spec, builder) in launches.items()
            }
            if "sources" in kernel:
                agreement.compare_launches(consumer, kmd, kernel["metadata"], observed)
            else:
                agreement.compare(consumer, kmd, kernel["metadata"], observed[""])


def _encoded(metadata, prefix, fields, arch):
    from dispatch.attention_bwd import hipdnn as H

    ids = {s: i for i, _k, s in H.tuned_keys(arch)}
    out = {}
    for f in fields:
        v = metadata[f]
        if f == "dtype":
            v = v.lower()
        if f == "tuning_key":
            out[f"{prefix}.{f}"] = ids.get(v, 0) if v else 0
        elif f in H.ENCODINGS:
            out[f"{prefix}.{f}"] = H.ENCODINGS[f][v]
        else:
            out[f"{prefix}.{f}"] = int(v)
    return out


@pytest.mark.parametrize("arch", ARCHES)
def test_corpus_plans_select_one_kernel_per_slot(hydrated, arch):
    """Every accepted corpus plan's class tokens select exactly one prep and one
    main kernel of the arch's packs, and those kernels' launches are the plan's
    kernel symbols."""
    from dispatch.attention_bwd import hipdnn as H

    prep_prefix, main_prefix = (
        "attention_general_bwd.prep",
        "attention_general_bwd.main",
    )
    index = {"prep": {}, "main": {}}
    for pack, kernels in hydrated.items():
        slot = "prep" if pack.startswith("prep_") else "main"
        if not pack.endswith(arch):
            continue
        prefix, fields = (
            (prep_prefix, H.PREP_TOKENS)
            if slot == "prep"
            else (main_prefix, H.MAIN_TOKENS)
        )
        for kernel, launches in kernels:
            key = tuple(
                sorted(_encoded(kernel["metadata"], prefix, fields, arch).items())
            )
            assert key not in index[slot], kernel["name"]
            index[slot][key] = {n: _symbol(*sb) for n, sb in launches.items()}
    _meta, entries = _corpus(arch)
    served = 0
    for e in entries:
        if e["expect"]["code"] != "OK":
            continue
        tokens = e["expect"]["tokens"]
        picked = {}
        for slot, prefix in (("prep", prep_prefix), ("main", main_prefix)):
            key = tuple(
                sorted((k, v) for k, v in tokens.items() if k.startswith(prefix + "."))
            )
            assert key in index[slot], (e["id"], slot)
            picked[slot] = index[slot][key]
        for launch in e["expect"]["launches"]:
            symbols = picked[launch["slot"]]
            assert symbols[launch["launch"]] == launch["kernel"], (
                e["id"],
                launch["launch"],
            )
        served += 1
    assert served > 0
