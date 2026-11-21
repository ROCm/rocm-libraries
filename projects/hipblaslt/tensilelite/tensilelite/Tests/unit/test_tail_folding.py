# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Runtime tail folding of the K tail into the TDM unroll loop (gfx1250).

Every kernel whose structure allows it (TDM A/B, PGR 2, cluster, ...) decides at run time:
fold = (r >= 2) && (DU <= 64 || K >= 3*DU || (K >= DU && r > DU/4)), r = K & (DU-1).
A folded tail runs as one more unroll iteration whose TDM K dim (shrunk at every address step)
zero-fills past K; otherwise the tail loop runs. With GSU each split's K dim starts at its own
first k-tile and the split owning k-tile floor(K/DU) takes the extra iteration.
"""

import re

import pytest
import yaml

from config_harness import emit_kernels_from_config


pytestmark = pytest.mark.unit


def _config(extra=(), mi=(16, 16, 32, 1, 1, 1, 1, 2, 1), wg=(32, 1, 1)):
    # bf16 TN, TDM + SIA4 + PGR2 + PLR1, DepthU 128, cluster 2x2 (no StaggerU code).
    fork_parameters = [
        {"MatrixInstruction": [list(mi)]},
        {"WorkGroup": [list(wg)]},
        {"DepthU": [128]},
        {"LdsBlockSizePerPadA": [256]},
        {"LdsBlockSizePerPadB": [256]},
        {"PrefetchLocalRead": [1]},
        {"PrefetchGlobalRead": [2]},
        {"ScheduleIterAlg": [4]},
        {"TDMInst": [3]},
        {"TransposeLDS": [1]},
        {"LdsPadA": [8]},
        {"LdsPadB": [8]},
        {"GlobalReadVectorWidthA": [1]},
        {"GlobalReadVectorWidthB": [1]},
        {"VectorWidthA": [1]},
        {"VectorWidthB": [1]},
        {"LocalReadVectorWidth": [8]},
        {"StoreVectorWidth": [1]},
        {"SourceSwap": [True]},
        {"MIArchVgpr": [True]},
        {"InitCIterWmma": [1]},
        {"WavefrontSize": [32]},
        {"ExpandPointerSwap": [False]},
        {"ClusterDim": [[2, 2]]},
    ]
    fork_parameters = [p for p in fork_parameters if next(iter(p)) not in dict(extra)] + [{k: [v]} for k, v in extra]
    return {
        "GlobalParameters": {"Architecture": "gfx1250", "PrintLevel": 0},
        "BenchmarkProblems": [[
            {
                "OperationType": "GEMM",
                "DataType": "b",
                "DestDataType": "b",
                "ComputeDataType": "s",
                "HighPrecisionAccumulate": True,
                "TransposeA": True,
                "TransposeB": False,
                "UseBeta": True,
                "Batched": True,
                "SupportUserArgs": True,
            },
            {
                "InitialSolutionParameters": None,
                "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
                "ForkParameters": fork_parameters,
                "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [720, 432, 1, 1248]}]}],
            },
        ]],
    }


def _emit(tmp_path, config, name):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return emit_kernels_from_config(path, limit=1, arch="gfx1250")


def _emit_one(tmp_path, config, name):
    results = _emit(tmp_path, config, name)
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    return assembly


def _kernel_name(assembly):
    return re.search(r"^\.amdhsa_kernel (\S+)", assembly, re.M).group(1)


RULE128 = "// TailFolding: r = K & (DU-1); rule: fold = (r >= 2) && (K >= 3*DU || (K >= DU && r > DU/4)) && !(GSUC && GSU > 1), DU = 128"


@pytest.mark.parametrize("hplr", [0, 2], ids=["noloadloop", "halfplr"])
def test_runtime_fold_codegen(tmp_path, hplr):
    # DU 128: decided right after the first TDM issue, before its address step; the tail loop stays.
    extra = [("HalfPLR", hplr)] if hplr else []
    kw = dict(mi=(16, 16, 32, 1, 1, 2, 2, 2, 2), wg=(32, 4, 1)) if hplr else {}
    asm = _emit_one(tmp_path, _config(extra, **kw), "tf_%u.yaml" % hplr)
    assert "_TF" not in _kernel_name(asm) and "label_TailLoopBeginL" in asm
    for tag in (RULE128, "// TailFolding: t = (r > DU/4) ? DU : 3*DU", "// TailFolding: fold = (K >= t) && (r >= 2)",
                "// TailFolding: fold = fold && !(GSUC && GSU > 1)", "// TailFolding: r == 0, no fold",
                "// TailFolding: no fold, K dim step stays 0", "// TailFolding: r == 0, no fold, K dim unchanged",
                "// TailFolding: loop count += extra", "// TailFolding: folded tail ran in the unroll loop, skip the tail loop",
                "// TailFolding: K dim = K - split start (low half)"):
        assert tag in asm, tag
    first_tdm = asm.index("tensor_load_to_lds")
    assert first_tdm < asm.index(RULE128) < asm.index("// TailFolding: K dim -= fold ? step : 0")
    assert asm.index("TailFolding: fold = 0 until decided after the first TDM issue") < first_tdm
    steps = asm.count("// TDM increment hi (carry)")
    assert steps and asm.count("TailFolding: K dim -= fold ? step : 0 (low half)") == steps


def test_runtime_fold_du64_decides_early(tmp_path):
    # DU <= 64 folds K < DU too, so the decision precedes the loop count's zero test (and the first TDM issue).
    asm = _emit_one(tmp_path, _config([("DepthU", 64), ("LdsBlockSizePerPadA", 128), ("LdsBlockSizePerPadB", 128)]), "tf64.yaml")
    rule = "// TailFolding: r = K & (DU-1); rule: fold = (r >= 2) (DU <= 64) && !(GSUC && GSU > 1), DU = 64"
    assert rule in asm and asm.index(rule) < asm.index("tensor_load_to_lds")


def test_runtime_fold_off_when_illegal(tmp_path):
    # PrefetchGlobalRead 1 is not legal for folding -> the kernel keeps the plain tail loop; never a reject.
    asm = _emit_one(tmp_path, _config([("PrefetchGlobalRead", 1)]), "tf_pgr1.yaml")
    assert "TailFold" not in asm and "label_TailLoopBeginL" in asm


@pytest.mark.parametrize("gsu", [1, 3])
def test_runtime_fold_stagger(tmp_path, gsu):
    # No cluster -> StaggerU: decided before the loop count is used; the split starts StaggerUIter steps in and
    # the wrap back to its first k-tile adds (splitIters-1)*step to the K dim.
    asm = _emit_one(tmp_path, _config([("ClusterDim", [1, 1]), ("GlobalSplitU", gsu), ("GlobalSplitUAlgorithm", "MultipleBuffer")]),
                    "tf_stagger.yaml")
    for tag in ("// TailFolding: stagger start = StaggerUIter * step", "// TailFolding: K dim -= stagger start (low half)",
                "// TailFolding: wrap K step = -(splitIters - 1) * step",
                "// TailFolding: K dim step = wrapIter ? -(splitIters-1)*step : step (low)",
                "// TailFolding: no fold, wrap K step stays 0", "// TailFolding: no fold, K dim unchanged"):
        assert tag in asm, tag
    assert asm.index(RULE128) < asm.index("tensor_load_to_lds")
    assert "fold = 0 until decided after the first TDM issue" not in asm
    steps = asm.count("// TDM increment hi (carry)") + asm.count("TDM addr += inc (with wrap, 64-bit)")
    assert asm.count("TailFolding: K dim -= fold ? (wrapIter ? -(splitIters-1)*step : step) : 0 (low half)") >= 1


@pytest.mark.parametrize("gsu", [2, 3, 4, -1])
def test_runtime_fold_gsu(tmp_path, gsu):
    asm = _emit_one(tmp_path, _config([("GlobalSplitU", gsu), ("GlobalSplitUAlgorithm", "MultipleBuffer"), ("PrefetchGL2", 1)]), "tf_gsu.yaml")
    for tag in ("// TailFolding: owner split ? (GSUSumIdx == floor(K/DU) % GSU)", "// TailFolding: extra = owner ? fold : 0",
                "// TailFolding: split start = GSUSumIdx * DU, in the K dim field", "// TailFolding: K dim step = GSU k-tiles"):
        assert tag in asm, tag
    # DU 128 folds only when K >= DU: no GL2 K clamp
    assert "// TailFolding: past K?" not in asm


@pytest.mark.parametrize("pgl", [1, 2])
def test_runtime_fold_gl2_threshold(tmp_path, pgl):
    # A folded partial last k-tile moves every GL2 prefetch threshold up by one (selected once before the loop).
    asm = _emit_one(tmp_path, _config([("PrefetchGL2", pgl), ("DepthU", 64), ("LdsBlockSizePerPadA", 128), ("LdsBlockSizePerPadB", 128),
                                       ("GlobalSplitU", 2), ("GlobalSplitUAlgorithm", "MultipleBuffer")]), "tf_pgl%u.yaml" % pgl)
    assert re.search(r"^s_cselect_b32 s\[sgprGL2FoldThreshold\], %u, %u +// TailFolding: GL2 threshold = PGR\+GL2 \+ fold" % (3 + pgl, 2 + pgl), asm, re.M)
    assert len(re.findall(r"^s_cmp_le_u32 s\[sgprLoopCounterL\], s\d+ +// TailFolding: counterL <= PGR(\+1)? \+ fold", asm, re.M)) == pgl
    assert re.search(r"^s_cmp_le_u32 s\[sgprLoopCounterL\], s\[sgprGL2FoldThreshold\] +// TailFolding: counterL <= PGR\+GL2 \+ fold", asm, re.M)
    # K < DepthU still prefetches k-tile 0 in the folded iteration: its K index is clamped to K - 1
    assert "// TailFolding: past K?" in asm and "// TailFolding: last K index from the split start" in asm


def test_runtime_fold_asm_tag(tmp_path):
    # every instruction of the fold decision and its uses carries a "TailFolding:" comment
    asm = _emit_one(tmp_path, _config([("PrefetchGL2", 1)]), "tf_tag.yaml")
    for line in asm.split("\n"):
        if re.search(r"sgprTailFold(On|Step)|sgprGL2FoldThreshold", line) and not line.startswith(".set"):
            assert "// TailFolding:" in line, line


def _simulate_stagger(K, DU, G, stagger):
    """Stagger model (decided before the count is used): split idx runs its count k-tiles from position
    s = stagger % count, wraps to its first k-tile; K dim measured from the k-tile base at every issue."""
    n, r = divmod(K, DU)
    fold = r >= 2 and (DU <= 64 or K >= 3 * DU or (K >= DU and 4 * r > DU))
    q, rem = divmod(n + fold, G)
    covered = []
    for idx in range(G):
        count = q + (idx < rem)
        s = stagger % count if count else 0
        step = G * DU if fold else 0
        kdim = K - idx * DU - s * step                               # init: K - split start - stagger start
        wrapk = -(count - 1) * step
        for j in range(count):
            pos = (s + j) % count
            t = idx + pos * G
            if fold:
                assert kdim == K - t * DU, (K, DU, G, stagger, idx, j)
                assert 0 < min(DU, kdim)
                covered.append((t * DU, t * DU + min(DU, kdim)))
            else:
                covered.append((t * DU, t * DU + DU))
            kdim -= wrapk if pos == count - 1 else step              # the wrap step happens after the last k-tile
        if not fold and r and idx == n % G:
            covered.append((n * DU, K))
    covered.sort(); end = 0
    for a, b in covered:
        assert a == end, (K, DU, G, stagger, covered); end = b
    assert end == K


@pytest.mark.parametrize("DU", [32, 128, 256])
@pytest.mark.parametrize("G", [1, 2, 3, 4])
def test_runtime_fold_stagger_model(DU, G):
    for K in list(range(DU, 6 * DU + 3, 7)) + [65601]:
        for stagger in (0, 1, 2, 3, 5, 31):
            _simulate_stagger(K, DU, G, stagger)


def _simulate(K, DU, G, late):
    """Per-split model of the generated arithmetic (GSUC == 0): k-tiles, K dim at every TDM issue, tail loop owner."""
    n, r = divmod(K, DU)
    fold = r >= 2 and (DU <= 64 or K >= 3 * DU or (K >= DU and 4 * r > DU))
    covered = []
    for idx in range(G):
        if late:                                     # distribute floor(K/DU), then the owner takes the folded tile
            q, rem = divmod(n, G)
            count = q + (idx < rem)
            if count > 0 and idx == rem:             # the late block only runs when the split's loop runs
                count += fold
            folded_here = fold and count > 0         # TailFoldOn of this split
        else:                                        # DU <= 64: fold is added before the distribution
            q, rem = divmod(n + fold, G)
            count = q + (idx < rem)
            folded_here = fold
        kdim = K - idx * DU                          # K dim at the split's first k-tile
        for j in range(count):
            t = idx + j * G                          # this split's j-th k-tile
            assert kdim == K - t * DU                # the descriptor K dim always measures from the k-tile base
            assert 0 < min(DU, kdim)                 # TDM clamps the partial tile at K, never past it
            covered.append((t * DU, t * DU + min(DU, kdim)))
            kdim -= G * DU if folded_here else 0
            if not folded_here:
                kdim = K - (t + G) * DU              # without folding the K dim is not tracked (no clamp needed)
        tail_owner = idx == (n % G)                  # the develop tail loop runs on split floor(K/DU) % GSU
        if r and not folded_here and tail_owner:
            covered.append((n * DU, K))
    covered.sort()
    pos = 0
    for a, b in covered:
        assert a == pos, (K, DU, G, late, covered)
        pos = b
    assert pos == K, (K, DU, G, late)


@pytest.mark.parametrize("DU", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("G", [1, 2, 3, 4, 7])
def test_runtime_fold_split_k_model(DU, G):
    for K in list(range(1, 6 * DU + 3)) + [65601, 70000, 100003]:
        _simulate(K, DU, G, late=DU > 64)


def test_rap_keeps_its_k_dim_shrink(tmp_path):
    # ReuseAcrossPersistent folds on its own (StaggerU code present): its per-step K dim shrink stays.
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "common" / "gemm" / "gfx12" / "rap_gfx1250.yaml"
    config = yaml.safe_load(src.read_text())
    problem = config["BenchmarkProblems"][0]
    for p in problem[1]["ForkParameters"]:
        if "ReuseAcrossPersistent" in p:
            p["ReuseAcrossPersistent"] = [1]
    config["BenchmarkProblems"] = [problem]
    results = _emit(tmp_path, config, "rap.yaml")
    assert results
    for _, assembly, error in results:
        assert error == 0
        assert "RAP: K dim -=" in assembly and "TailFoldOn" not in assembly
