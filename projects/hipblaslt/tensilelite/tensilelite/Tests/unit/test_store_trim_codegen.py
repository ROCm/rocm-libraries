# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Store trimming: observable store paths, not helper names or diagnostic comments.

Feature: AF0/AF1 full-tile proofs remove the corresponding checks and bodies;
reinstating those paths breaks the feature nodes. Preservation: unproved,
packed and forced-edge dimensions retain their conditional edge dispatch;
unconditionally eliding checkIsEdge breaks those nodes.
"""

import copy
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from store_trim_gemm_helpers import STORE_BOUNDARY_CASES

_DATA = Path(__file__).resolve().parent / "store_trim_data"
_LOGIC = _DATA / "gfx950_sk3_userargs_mt32x64.yaml"
_LOGIC_80 = _DATA / "gfx950_sk5_fdpo_cms_mt256x256.yaml"


def _amdclang():
    compiler = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    assert Path(compiler).is_file(), "the configured assembler is required"
    return compiler


def _emit_variant(tmp_path, name, *, af0=1, af1=1, logic=_LOGIC, sol=None):
    from codegen_harness import emit_kernels_from_logic

    data = copy.deepcopy(yaml.safe_load(logic.read_text()))
    for solution in data["Solutions"]:
        # Bespoke requests: only BespokeNarrowing solutions remove edge paths.
        solution.update(AssertFree0ElementMultiple=af0, AssertFree1ElementMultiple=af1,
                        BespokeNarrowing=True)
        solution.update(sol or {})
    path = tmp_path / (name + ".yaml")
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    result = emit_kernels_from_logic(str(path), canonical=False)
    assert len(result) == 1
    _, src, err = result[0]
    assert err == 0, "the real solution must emit before its store paths are examined"
    return src


def _assemble(tmp_path, name, src):
    path = tmp_path / (name + ".s")
    path.write_text(src)
    result = subprocess.run([_amdclang(), "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
                             "-mcpu=gfx950", "-c", str(path), "-o", str(path.with_suffix(".o"))],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]


def _mb_labels(src):
    return set(re.findall(r"^label_GW_B0_FD0_VW(\d+)_MB_(\w+):", src, re.M))


def _mb_branches(src):
    return re.findall(r"s_cbranch_scc1 label_GW_B0_FD0_VW\d+_MB_(Then|Else)", src)


def test_edge_checks_kept_by_default(tmp_path):
    src = _emit_variant(tmp_path, "default")
    assert ("1", "Then") in _mb_labels(src)
    assert sorted(_mb_branches(src)) == ["Else", "Then"]
    _assemble(tmp_path, "default", src)


def test_af0_multiple_of_mt0_elides_size0_edge(tmp_path):
    src = _emit_variant(tmp_path, "m0", af0=32)
    assert {vw for vw, _ in _mb_labels(src)} == {"8"}, "M-full stores must use only widest VW=8"
    assert _mb_branches(src) == ["Then"], "only the N-edge dispatch may remain"
    _assemble(tmp_path, "m0", src)


def test_af1_multiple_of_mt1_elides_size1_edge(tmp_path):
    src = _emit_variant(tmp_path, "m1", af1=64)
    assert _mb_branches(src) == ["Else"], "only the M-edge dispatch may remain"
    assert ("8", "Then") not in _mb_labels(src), "N-only edge body is unreachable"
    assert ("1", "Then") in _mb_labels(src), "partial M still needs scalar edge stores"
    _assemble(tmp_path, "m1", src)


def test_af0_af1_multiple_of_mt_elides_all_edges(tmp_path):
    src = _emit_variant(tmp_path, "m01", af0=32, af1=64)
    assert _mb_branches(src) == [], "both full dimensions need no edge dispatch"
    assert not any(body == "Then" for _, body in _mb_labels(src)), "no edge store body may survive"
    assert {vw for vw, _ in _mb_labels(src)} == {"8"}, "only the widest store vector width survives"
    _assemble(tmp_path, "m01", src)


def test_adaptive_gemm_af0_multiple_elides_size0_edge(tmp_path):
    src = _emit_variant(tmp_path, "adaptive", af0=256, logic=_LOGIC_80, sol={"StreamK": 3})
    # Adaptive dispatch goes through NonEdgeEnd rather than Then/Else.
    vws = set(re.findall(r"^label_GW_B0_FD0_VW(\d+)_GSU1_\w+:", src, re.M))
    assert vws == {"8"}, "adaptive M-full stores must keep only widest VW=8"
    assert not re.search(r"s_cbranch_scc1 label_GW_B0_FD0_VW\d+_GSU1_Else", src)
    _assemble(tmp_path, "adaptive", src)


def _edge_check(af0, af1, *, mt0=32, mt1=64, packed0=1, packed1=1,
                force=False, missing=False, dim=0):
    from gpu_test_helpers import init_rocisa
    from rocisa.code import Label
    from rocisa.container import ContinuousRegister
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    init_rocisa(target="gfx950")
    writer = object.__new__(KernelWriterAssembly)
    writer.states = SimpleNamespace(asmCaps={"HasSCMPK": True})
    writer.do = {"EdgeWrite": True}
    writer.db = {"ForceEdgeStores": int(force)}
    kernel = {"AssertFree0ElementMultiple": af0, "AssertFree1ElementMultiple": af1,
              "BespokeNarrowing": True,
              "MacroTile0": mt0, "MacroTile1": mt1,
              "PackedC0IndicesX": list(range(packed0)),
              "PackedC1IndicesX": list(range(packed1)),
              "ProblemType": {"Index0": 0, "Index1": 1}}
    if missing:
        del kernel["AssertFree0ElementMultiple"], kernel["AssertFree1ElementMultiple"]
    module = KernelWriterAssembly.checkIsEdge(writer, kernel, ContinuousRegister(idx=4, size=4),
                                            Label(label="edge_target", comment=""),
                                            mt1 if dim else mt0, isSize1=bool(dim))
    return str(module)


@pytest.mark.parametrize("af0,af1,mt0,mt1,dim", [
    (32, 1, 32, 64, 0), (1, 64, 32, 64, 1),
    (64, 128, 32, 64, 0), (64, 128, 32, 64, 1),
    (192, 128, 96, 64, 0),  # a non-power-of-two tile; a true multiple
], ids=["m", "n", "m_two_tiles", "n_two_tiles", "m_non_power_of_two"])
def test_proven_full_dimension_emits_no_edge_instructions(af0, af1, mt0, mt1, dim):
    src = _edge_check(af0, af1, mt0=mt0, mt1=mt1, dim=dim)
    instructions = [line.split("//", 1)[0].strip() for line in src.splitlines()]
    assert not any(line.startswith("s_") for line in instructions), "proven-full edge check still emits instructions"


@pytest.mark.parametrize("kwargs", [
    {"af0": 1, "af1": 1},
    {"af0": 16, "af1": 64},                # positive size below the M tile
    {"af0": 48, "af1": 128},               # larger but not divisible by M tile
    {"af0": 128, "af1": 128, "mt0": 96},  # non-power-of-two tile, not a multiple
    {"af0": 32, "af1": 64, "packed0": 2},
    {"af0": 32, "af1": 64, "packed1": 2, "dim": 1},
    {"af0": 32, "af1": 64, "force": True},
    {"af0": 1, "af1": 1, "missing": True},
], ids=["default", "below_tile", "non_multiple", "non_power_of_two",
        "packed_m", "packed_n", "forced_edge", "missing_asserts"])
def test_unproved_dimension_keeps_conditional_edge_branch(kwargs):
    src = _edge_check(**kwargs)
    assert re.search(r"\bs_cbranch_scc1\s+label_edge_target\b", src), "unproved edge dispatch was removed"


def _emit_store_boundary(monkeypatch, tmp_path, parent, mode):
    """Observe real batch/remap emission without replacing its output."""
    import inspect
    from store_trim_gemm_helpers import PARENTS, make_request
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    root = Path(__file__).resolve().parents[6]
    request = make_request(tmp_path / "request", root, parent, mode)
    mt0, mt1 = PARENTS[parent][2:4]
    batches, remaps = [], []
    write_batch = KernelWriterAssembly.globalWriteBatch
    signature = inspect.signature(write_batch)
    remap_store = KernelWriterAssembly.storeRemapAddStore

    def capture_batch(writer, *args, **kwargs):
        bound = signature.bind(writer, *args, **kwargs).arguments
        batches.append({"beta": bound["beta"], "edge": bound["edge"],
                        "vw": bound["gwvw"], "factor": bound["factorDim"],
                        "index": bound["batchIdx"], "count": bound["numBatches"],
                        "elements": tuple(tuple(e) for e in bound["batchElements"]),
                        "remap": bound["kernel"]["StoreRemapVectorWidth"]})
        return write_batch(writer, *args, **kwargs)

    def capture_remap(writer, kernel, tmp_vgpr, tmp_sgpr, edge, last):
        module, count = remap_store(writer, kernel, tmp_vgpr, tmp_sgpr, edge, last)
        remaps.append({"edge": edge, "last": last, "count": count,
                       "stores": re.findall(r"\bbuffer_store_\w+", str(module))})
        return module, count

    monkeypatch.setattr(KernelWriterAssembly, "globalWriteBatch", capture_batch)
    monkeypatch.setattr(KernelWriterAssembly, "storeRemapAddStore", capture_remap)
    logic, = (request / "bespoke").glob("*.yaml")
    src = _emit_variant(tmp_path, parent + "_" + mode,
                        af0=mt0 if mode in ("m", "both") else 1,
                        af1=mt1 if mode in ("n", "both") else 1, logic=logic)
    assert batches, "the full GEMM must emit real store batches"
    return src, batches, remaps




@pytest.mark.parametrize("parent,mode", STORE_BOUNDARY_CASES,
                         ids=[p + "_" + m for p, m in STORE_BOUNDARY_CASES])
def test_store_boundary_full_tile_paths_trimmed(monkeypatch, tmp_path, parent, mode):
    """Feature: restoring full-tile checks/bodies or lower VWs breaks it."""
    src, batches, remaps = _emit_store_boundary(monkeypatch, tmp_path, parent, mode)
    labels = re.findall(r"^label_GW_B(\d+)_FD(\d+)_VW(\d+)_(?:MB_|GSU1_)?(Then|Else|NonEdge)(?:_Deferred)?:", src, re.M)
    assert labels and any(beta == "1" for beta, _, _, _ in labels), "beta=1 store trees must be emitted"
    widest = "1" if parent == "classic_remainder" else "4"
    if mode in ("m", "both"):
        assert {vw for _, _, vw, _ in labels} == {widest}, "M-full boundary retains a narrow store body"
    if mode == "both":
        assert not any(body == "Then" for _, _, _, body in labels), "both-full boundary retains an edge body"
    elif mode == "n":
        assert not any(vw == widest and body == "Then" for _, _, vw, body in labels), "N-only edge body survives"
    store = src[src.index("label_GW_B0_FD0"):]
    for dim in ((0, 1) if mode == "both" else (0,) if mode == "m" else (1,)):
        operand = "sgprSizeI" if dim == 0 else "sgprSizeJ"
        assert not re.search(r"s_and_b32[^\n]*s\[" + operand + r"\]", store), "proven-full store remainder check survives"
    assert bool(remaps) == parent.endswith("remap_on"), "requested remap emitter was not exercised"
    _assemble(tmp_path, parent + "_" + mode, src)


@pytest.mark.parametrize("parent,mode", [(p, m) for p, m in STORE_BOUNDARY_CASES if p.endswith("remap_on")],
                         ids=[p + "_" + m for p, m in STORE_BOUNDARY_CASES if p.endswith("remap_on")])
def test_store_remap_writes_survive_full_tile_facts(monkeypatch, tmp_path, parent, mode):
    """Preservation: suppressing storeRemapAddStore writes breaks it."""
    src, _, remaps = _emit_store_boundary(monkeypatch, tmp_path, parent, mode)
    assert remaps and all(r["last"] == 1 and r["count"] > 0 and r["stores"] for r in remaps), "remap flush lost its global stores"
    assert any(not r["edge"] for r in remaps), "interior remap stores are missing"
    if mode != "both":
        assert any(r["edge"] for r in remaps), "partial opposite dimension lost its masked remap stores"
    _assemble(tmp_path, parent + "_" + mode, src)


@pytest.mark.parametrize("parent,mode,widest,sizes", [
    ("classic_remainder", "m", 1, [2, 2, 3, 3, 3, 3]),
    ("classic_remainder", "both", 1, [2, 2, 3, 3, 3, 3]),
    ("subtile_remainder", "m", 4, [16, 16, 16, 8]),
], ids=["classic_m", "classic_both", "subtile_m"])
def test_short_final_store_batch_keeps_every_element(monkeypatch, tmp_path, parent, mode, widest, sizes):
    """Preservation: truncating the store-batch loop to floor division breaks it."""
    src, batches, _ = _emit_store_boundary(monkeypatch, tmp_path, parent, mode)
    groups = []
    for batch in batches:
        if batch["beta"] and not batch["edge"] and batch["vw"] == widest and batch["factor"] == 0:
            if batch["index"] == 0:
                groups.append([])
            groups[-1].append(batch)
    assert groups, "beta=1 widest interior store batches are missing"
    for group in groups:
        observed_sizes = [len(b["elements"]) for b in group]
        budget = 3 if parent == "classic_remainder" else 16
        assert max(observed_sizes) <= budget, "store batch exceeds the recipe budget"
        if parent == "classic_remainder":
            # Classic f32 has rounding step 1. Subtile keeps aligned columns
            # and is intentionally outside the C3 balancing gate.
            assert max(observed_sizes) - min(observed_sizes) <= 1, "store batches are unbalanced"
        assert [len(b["elements"]) for b in group] == sizes, "short final store batch was lost"
        assert [b["index"] for b in group] == list(range(len(sizes)))
        assert all(b["count"] == len(sizes) for b in group)
        elements = [e for b in group for e in b["elements"]]
        assert len(set(elements)) == sum(sizes), "store batches duplicate or omit an element"
    _assemble(tmp_path, parent + "_" + mode, src)
