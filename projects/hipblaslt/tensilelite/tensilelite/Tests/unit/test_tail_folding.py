# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tail folding of the K tail into the TDM unroll loop (gfx1250), fixed at codegen.

Every kernel whose structure allows it (any input type but packed sub-byte F4/F6, TDM A/B, PGR 1 or 2, ...) folds: the unroll loop runs
ceil(K/DU) iterations, the TDM K dim (shrunk at every address step) zero-fills the partial last k-tile, and
no tail loop is emitted. Any other kernel keeps the develop tail loop. With GSU the folded k-tile count is
distributed; each split's K dim starts at its own first k-tile (GSUC 0: interleaved, GSUC 1: contiguous).
The K dim step is the wave's TDM address increment shifted into the field, so a kernel whose step cannot be
derived that way (swizzled / mirrored K, non-unit K stride, A/B element sizes that differ, ...) keeps the tail loop.
A TDM iterate-mode tensor folds too: its load walks whole rows (dim1) and keeps K in dim0. RAP still rejects it.
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


def _emit(tmp_path, config, name, limit=1):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return emit_kernels_from_config(path, limit=limit, arch="gfx1250")


def _emit_one(tmp_path, config, name):
    results = _emit(tmp_path, config, name)
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    return assembly


def _kernel_name(assembly):
    return re.search(r"^\.amdhsa_kernel (\S+)", assembly, re.M).group(1)


CEIL = "// TailFolding: loop count = ceil(K / DU)"


def _steps(asm):
    return asm.count("// TDM increment hi (carry)")


def _instructions(asm):
    return [l.strip() for l in asm.split("\n") if l.strip() and not l.strip().startswith(("//", ".", "label_")) and not l.strip().endswith(":")]


_SCC_SAFE = re.compile(r"s_(wait\w*|nop|delay_alu|mov_b32|mov_b64|setprio|barrier\w*|sleep|set_vgpr_msb|add_u64|sub_u64)\b")  # s_add_u64 = s_add_nc_u64: SCC unchanged


def _check_borrow_pairs(asm):
    # the scheduler may interleave other work between a K dim low-half subtract and its borrow, but no SALU
    # that writes SCC (VALU / DS / VMEM / TDM and waits leave it alone)
    ins = _instructions(asm)
    lows = [i for i, l in enumerate(ins) if re.match(r"s_sub_u32 .*// TailFolding: K dim -= .*\(low half\)", l)]
    assert lows
    for i in lows:
        j = i + 1
        while not ins[j].startswith("s_subb_u32"):
            assert not ins[j].startswith("s_") or _SCC_SAFE.match(ins[j]), (ins[i], ins[j])
            j += 1
        assert re.match(r"s_subb_u32 .*// TailFolding: K dim -= .*\(high half\)", ins[j]), (ins[i], ins[j])


@pytest.mark.parametrize("hplr", [0, 2], ids=["noloadloop", "halfplr"])
def test_fold_codegen(tmp_path, hplr):
    # fixed fold: ceil(K/DU) before the first TDM issue, a K dim shrink at every address step, no tail loop.
    extra = [("HalfPLR", hplr)] if hplr else []
    kw = dict(mi=(16, 16, 32, 1, 1, 2, 2, 2, 2), wg=(32, 4, 1)) if hplr else {}
    asm = _emit_one(tmp_path, _config(extra, **kw), "tf_%u.yaml" % hplr)
    assert "_TF" not in _kernel_name(asm) and "label_TailLoopBeginL" not in asm
    for tag in (CEIL, "// TailFolding: r == 0, K dim unchanged", "// TailFolding: K dim = K - split start (low half)"):
        assert tag in asm, tag
    for gone in ("TailFoldOn", "fold ?", "skip the tail loop"):
        assert gone not in asm, gone
    first_tdm = asm.index("tensor_load_to_lds")
    assert asm.index(CEIL) < first_tdm
    shrink = re.findall(r"// TailFolding: K dim -= \S.* \(low half\)", asm)
    assert _steps(asm) and len([x for x in shrink if "stagger" not in x]) == _steps(asm)


def test_fold_8bit(tmp_path):
    # 8-bit inputs fold like 16-bit ones.
    config = _config([(k, -1) for k in ("LdsBlockSizePerPadA", "LdsBlockSizePerPadB", "LdsPadA", "LdsPadB", "LocalReadVectorWidth")])
    config["BenchmarkProblems"][0][0].update(DataType="f8", DestDataType="b")
    for p in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "MatrixInstruction" in p:
            p["MatrixInstruction"] = [[16, 16, 64, 1, 1, 1, 1, 2, 1]]
    asm = _emit_one(tmp_path, config, "tf_f8.yaml")
    assert CEIL in asm and "label_TailLoopBeginL" not in asm


@pytest.mark.parametrize("dtype,mik,shift", [("f8", 64, 16), ("b", 32, 15), ("s", 4, 14)], ids=["f8", "bf16", "fp32"])
def test_fold_step_shift_per_type(tmp_path, dtype, mik, shift):
    # the K dim counts elements (bytes for 1-byte types): step = increment << (16 - log2(bpe)), high half >> (16 + log2(bpe))
    extra = [("GlobalSplitU", 2), ("GlobalSplitUAlgorithm", "MultipleBuffer")]
    if dtype != "b":
        extra += [(k, -1) for k in ("LdsBlockSizePerPadA", "LdsBlockSizePerPadB", "LdsPadA", "LdsPadB", "LocalReadVectorWidth")]
    config = _config(extra, mi=(16, 16, mik, 1, 1, 1, 1, 2, 1))
    config["BenchmarkProblems"][0][0].update(DataType=dtype, DestDataType="b" if dtype != "s" else "s")
    asm = _emit_one(tmp_path, config, "tf_shift_%s.yaml" % dtype)
    assert CEIL in asm and "label_TailLoopBeginL" not in asm and "sgprTailFold" not in asm
    lo = re.findall(r"^s_lshl_b32 s\d+, s\[sgprtdmABIncs\], %s +// TailFolding: K dim step" % hex(shift), asm, re.M)
    hi = re.findall(r"^s_lshr_b32 s\d+, s\[sgprtdmABIncs\], %s +// TailFolding: K dim step" % hex(32 - shift), asm, re.M)
    assert len(lo) == len(hi) == _steps(asm) > 0
    _check_borrow_pairs(asm)


def _f6_config():
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "common" / "gemm" / "gfx12" / "f6_tdm_gfx1250.yaml"
    config = yaml.safe_load(src.read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    return config


def _record_fold(monkeypatch):
    # The _TailFolding each kernel reaches codegen with, i.e. what Solution decided.
    from tensilelite.KernelWriter import KernelWriter
    seen = []
    source = KernelWriter._getKernelSource

    def recording(self, kernel):
        seen.append(kernel.get("_TailFolding", False))
        return source(self, kernel)

    monkeypatch.setattr(KernelWriter, "_getKernelSource", recording)
    return seen


def test_fold_subbyte_keeps_tail_loop(tmp_path, monkeypatch):
    # Packed sub-byte inputs (F6 here): the TDM K dim does not step in elements, so Solution does not
    # fold them and codegen emits the tail loop; never a reject.
    seen = _record_fold(monkeypatch)
    results = _emit(tmp_path, _f6_config(), "tf_f6.yaml", limit=64)
    assert results and seen and not any(seen)
    for _, assembly, error in results:
        assert error == 0
        assert CEIL not in assembly and "label_TailLoopBeginL" in assembly


def test_fold_gate_gap_is_an_error(tmp_path, monkeypatch, capsys):
    # A kernel Solution folds although the writer cannot derive its K dim step (simulated here) is an
    # internal error.
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    monkeypatch.setattr(KernelWriterAssembly, "tailFoldStepShift", lambda self, kernel: None)
    seen = _record_fold(monkeypatch)
    results = _emit(tmp_path, _config(), "tf_gap.yaml")
    out = capsys.readouterr().out
    assert any(seen)
    assert len(results) == 1 and results[0][2] != 0
    assert "the K dim step cannot be derived from the TDM increment" in out


@pytest.mark.parametrize("extra", [[("PrefetchGlobalRead", 0)],
                                   [("PrefetchGlobalRead", 1), ("SuppressNoLoadLoop", True), ("GlobalSplitU", 1)]],
                         ids=["pgr0", "pgr1_suppress_nll"])
def test_fold_off_when_illegal(tmp_path, extra):
    # PGR 0, and PGR 1 without a NoLoadLoop (its last iteration loads a k-tile past K) unless HalfPLR
    # disables that load, keep the plain tail loop; never a reject.
    asm = _emit_one(tmp_path, _config(extra), "tf_illegal.yaml")
    assert "TailFold" not in asm and "label_TailLoopBeginL" in asm


_PGR1_CASES = {
    "noloadloop": ([], {}),
    "halfplr": ([("HalfPLR", 2)], dict(mi=(16, 16, 32, 1, 1, 2, 2, 2, 2), wg=(32, 4, 1))),
    "gsu_gl2": ([("GlobalSplitU", -1), ("GlobalSplitUAlgorithm", "MultipleBuffer"), ("PrefetchGL2", 2)], {}),
    "stagger_gsu": ([("ClusterDim", [1, 1]), ("GlobalSplitU", 3), ("GlobalSplitUAlgorithm", "MultipleBuffer")], {}),
}


@pytest.mark.parametrize("case", list(_PGR1_CASES))
def test_fold_pgr1_codegen(tmp_path, case):
    # PGR 1 folds like PGR 2: the prologue issues k-tile 0 and one address step, each loop iteration issues the
    # next k-tile and one address step, the NoLoadLoop (or HalfPLR's disabled last iteration) issues nothing.
    # Every address step shrinks the K dim, so k-tile t is issued after t shrinks.
    extra, kw = _PGR1_CASES[case]
    asm = _emit_one(tmp_path, _config([("PrefetchGlobalRead", 1)] + extra, **kw), "tf_pgr1_%s.yaml" % case)
    assert "label_TailLoopBeginL" not in asm and "toPGR1:" not in asm
    assert asm.index(CEIL) < asm.index("tensor_load_to_lds")
    shrink = re.findall(r"// TailFolding: K dim -= \S.* \(low half\)", asm)
    if case == "stagger_gsu":
        assert asm.count("TailFolding: K dim -= stagger start (low half)") == 2
        assert asm.count("TailFolding: K dim -= wrapIter ? -(splitIters-1)*step : step (low half)") \
            == asm.count("TDM addr += inc (with wrap, 64-bit)") > 0
    else:
        assert _steps(asm) and len(shrink) == _steps(asm)
    if case == "halfplr":
        assert "is this the last iters" in asm
    else:
        # the NoLoadLoop issues no TDM load
        assert "tensor_load_to_lds" not in asm[asm.index("label_LoopEndL:"):]
    if case == "gsu_gl2":
        assert "// TailFolding: split start = start k-tile * DU, in the K dim field" in asm
        assert re.search(r"^s_addc_u32 (s\d+), 1, 0 +// TailFolding: pre-loop GL2 threshold = PGR \(\+1 if K tail\)", asm, re.M)
        assert re.search(r"^s_addc_u32 (s\d+), 3, 0 +// TailFolding: GL2 threshold = PGR\+GL2 \(\+1 if K tail\)", asm, re.M)
    _check_borrow_pairs(asm)


def _simulate_issue(K, DU, pgr):
    """Issue order of the generated unroll loop (no GSU): the prologue issues k-tiles 0..pgr-1 with an address
    step after k-tile 0 only (PGR 2 issues k-tile 1 from the stepped address), every loop iteration steps the
    address (PGR 2: before its issue; PGR 1: after it) and issues the next k-tile, the loop leaves at
    counter == pgr (PGR 2 with count 1 jumps to toPGR1), and the drain issues nothing. Returns the issued
    (k-tile, K dim) pairs and the number of k-tiles computed."""
    n = -(-K // DU)
    kdim, addr, issued = K, 0, []

    def step():
        nonlocal kdim, addr
        addr += 1; kdim -= DU

    def issue():
        if addr < n:                                            # the prologue's 2nd PGR stage skips past the count
            issued.append((addr, kdim))

    issue(); step()
    if pgr == 2:
        issue()
    counter, computed = n, 0
    if counter > pgr:
        while True:
            if pgr == 2:
                step()
            issue()
            if pgr == 1:
                step()
            computed += 1; counter -= 1
            if counter <= pgr:
                break
    computed += min(counter, pgr)                               # NGLL + NLL (or toPGR1: NLL only)
    return n, issued, computed


@pytest.mark.parametrize("pgr", [1, 2])
@pytest.mark.parametrize("DU", [32, 128, 256])
def test_fold_issue_model(DU, pgr):
    # every k-tile 0..ceil(K/DU)-1 is issued once, with the K dim measured from its base (so the partial last
    # one is clamped at K), none past it, and ceil(K/DU) k-tiles are computed: K < DU, r = 0, floor 0 / 1 / 2
    for K in list(range(1, 4 * DU + 3)) + [65535, 65601, 70001]:
        n, issued, computed = _simulate_issue(K, DU, pgr)
        assert [t for t, _ in issued] == list(range(n)), (K, DU, pgr)
        assert all(kd == K - t * DU for t, kd in issued), (K, DU, pgr)
        assert computed == n, (K, DU, pgr)


@pytest.mark.parametrize("gsu", [1, 3])
def test_fold_stagger(tmp_path, gsu):
    # No cluster -> StaggerU: the split starts StaggerUIter steps in and the wrap back to its first k-tile adds
    # (splitIters-1)*step to the K dim. Both are the stagger offset / the selected increment (WrapU on the
    # wrapIter) in K dim units: no SGPR is kept for them.
    asm = _emit_one(tmp_path, _config([("ClusterDim", [1, 1]), ("GlobalSplitU", gsu), ("GlobalSplitUAlgorithm", "MultipleBuffer")]),
                    "tf_stagger.yaml")
    tags = ("// TailFolding: stagger start = stagger offset << 15", "// TailFolding: K dim step = (wrapIter ? WrapU : inc) << 15")
    assert "sgprTailFold" not in asm
    assert len(re.findall(r"^s_lshl_b64 .*// TailFolding: K dim step = \(wrapIter \? WrapU : inc\) << 15", asm, re.M)) \
        == asm.count("TailFolding: K dim -= wrapIter ? -(splitIters-1)*step : step (low half)")
    for tag in tags + ("// TailFolding: K dim -= stagger start (low half)", CEIL):
        assert tag in asm, tag
    _check_borrow_pairs(asm)
    assert "label_TailLoopBeginL" not in asm and "TailFoldOn" not in asm
    assert asm.index(CEIL) < asm.index("tensor_load_to_lds")
    assert asm.count("TailFolding: K dim -= wrapIter ? -(splitIters-1)*step : step (low half)") >= 1


@pytest.mark.parametrize("gsu", [2, 3, 4, -1])
def test_fold_gsu(tmp_path, gsu):
    # GSUC is a run-time bit: the k-tile step is GSUC ? 1 : GSU k-tiles and every GSUC == 1 start offset is
    # taken from the folded count ceil(K/DU). The step is the wave's TDM address increment
    # ((GSUC ? 1 : GSU) * DU * 2 bytes) << 15, low and high half, at every address step; no SGPR is kept for it.
    asm = _emit_one(tmp_path, _config([("GlobalSplitU", gsu), ("GlobalSplitUAlgorithm", "MultipleBuffer"), ("PrefetchGL2", 1)]), "tf_gsu.yaml")
    assert "sgprTailFold" not in asm
    lo = re.findall(r"^s_lshl_b32 (s\d+), s\[sgprtdmABIncs\], 0xf +// TailFolding: K dim step = address increment << 15 \(low half\)", asm, re.M)
    hi = re.findall(r"^s_lshr_b32 (s\d+), s\[sgprtdmABIncs\], 0x11 +// TailFolding: K dim step = address increment << 15 \(high half\)", asm, re.M)
    assert len(lo) == len(hi) == _steps(asm) > 0
    _check_borrow_pairs(asm)
    for tag in (CEIL,
                "// TailFolding: split start = start k-tile * DU, in the K dim field",
                "// TailFolding: numIterL = ceil(SizesSum / DepthU(128))",
                "// TailFolding: numIter = ceil(SizesSum / DepthU(128))",
                "// TailFolding: last K index from the split start"):
        assert tag in asm, tag
    assert "label_TailLoopBeginL" not in asm
    # the fold is added to the k-tile count before GSU distributes it
    assert asm.index(CEIL) < asm.index("tensor_load_to_lds")
    assert "numIterL = SizesSum / DepthU" not in asm and "numIter = SizesSum / DepthU" not in asm


@pytest.mark.parametrize("pgl", [1, 2])
def test_fold_gl2_threshold(tmp_path, pgl):
    # A partial last k-tile moves every GL2 prefetch threshold up by one: SCC = (K & (DU-1) != 0), threshold =
    # s_addc(PGR(+GL2), 0), recomputed in the loop (no SGPR kept across it).
    asm = _emit_one(tmp_path, _config([("PrefetchGL2", pgl), ("DepthU", 64), ("LdsBlockSizePerPadA", 128), ("LdsBlockSizePerPadB", 128),
                                       ("GlobalSplitU", 2), ("GlobalSplitUAlgorithm", "MultipleBuffer")]), "tf_pgl%u.yaml" % pgl)
    assert "GL2FoldThreshold" not in asm
    assert re.search(r"^s_addc_u32 (s\d+), %u, 0 +// TailFolding: GL2 threshold = PGR\+GL2 \(\+1 if K tail\)" % (2 + pgl), asm, re.M)
    assert re.search(r"^s_addc_u32 (s\d+), 2, 0 +// TailFolding: pre-loop GL2 threshold = PGR \(\+1 if K tail\)", asm, re.M)
    assert len(re.findall(r"^s_cmp_le_u32 s\[sgprLoopCounterL\], s\d+ +// TailFolding: counterL <= PGR(\+1)? \(\+1 if K tail\)", asm, re.M)) == pgl
    assert re.search(r"^s_cmp_le_u32 s\[sgprLoopCounterL\], s\d+ +// TailFolding: counterL <= PGR\+GL2 \(\+1 if K tail\)", asm, re.M)
    _check_borrow_pairs(asm)
    # K < DepthU still prefetches k-tile 0 in the folded iteration: its K index is clamped to K - 1
    assert "// TailFolding: past K?" in asm and "// TailFolding: last K index from the split start" in asm


def _split_model(K, DU, G, gsuc):
    """Per-split start k-tile and count of the generated arithmetic: GSU distributes n = ceil(K/DU) k-tiles."""
    n = -(-K // DU)
    q, rem = divmod(n, G)
    out = []
    for idx in range(G):
        count = q + (idx < rem)
        start = idx if not gsuc else q * idx + min(idx, rem)     # calculateLoopNumIterOffsetGsu on the folded n
        out.append((start, count))
    return out


def _check_cover(covered, K, ctx):
    covered.sort(); end = 0
    for a, b in covered:
        assert a == end, (ctx, covered); end = b
    assert end == K, ctx


def _simulate(K, DU, G, gsuc):
    """K dim at every TDM issue (GSUC 0: split idx reads k-tiles idx + j*G, GSUC 1: a contiguous block)."""
    r = K % DU
    covered = []
    tile_step = 1 if (gsuc or G == 1) else G
    for idx, (start, count) in enumerate(_split_model(K, DU, G, gsuc)):
        kdim = K - start * DU if r else K                      # split K dim init (skipped when r == 0)
        for j in range(count):
            t = start + j * tile_step
            remaining = K - t * DU
            assert remaining > 0, (K, DU, G, gsuc, idx, j)
            if r:
                assert kdim == remaining                       # the K dim measures from the k-tile base
            else:
                assert kdim >= remaining >= DU                 # unchanged init: never clamps a full k-tile
            covered.append((t * DU, t * DU + min(DU, kdim)))
            kdim -= tile_step * DU                             # the K dim step
        if count and r and start + (count - 1) * tile_step == -(-K // DU) - 1:
            assert idx == max(i for i, (_, c) in enumerate(_split_model(K, DU, G, gsuc)) if c) or not gsuc
    _check_cover(covered, K, (K, DU, G, gsuc))


@pytest.mark.parametrize("gsuc", [0, 1], ids=["gsuc0", "gsuc1"])
@pytest.mark.parametrize("DU", [32, 64, 128, 256, 512])
@pytest.mark.parametrize("G", [1, 2, 3, 4, 7])
def test_fold_split_k_model(DU, G, gsuc):
    for K in list(range(1, 6 * DU + 3)) + [65601, 70000, 100003]:
        _simulate(K, DU, G, gsuc)


def test_fold_gsuc1_partial_tile_on_last_split():
    # GSUC 1: the partial k-tile is the last k-tile of the last split that has any.
    for DU in (64, 128):
        for G in (2, 3, 4):
            for K in range(1, 9 * DU, 5):
                m = _split_model(K, DU, G, 1)
                last = max(i for i, (_, c) in enumerate(m) if c)
                s, c = m[last]
                assert s + c == -(-K // DU)


def _simulate_stagger(K, DU, G, stagger):
    """Stagger model (GSUC 0): split idx runs its count k-tiles from position s = stagger % count, wraps to its
    first k-tile; K dim measured from the k-tile base at every issue."""
    r = K % DU
    covered = []
    for idx, (start, count) in enumerate(_split_model(K, DU, G, 0)):
        s = stagger % count if count else 0
        step = G * DU
        kdim = (K - idx * DU if r else K) - s * step              # init: K - split start - stagger start
        wrapk = -(count - 1) * step
        for j in range(count):
            pos = (s + j) % count
            t = idx + pos * G
            if r:
                assert kdim == K - t * DU, (K, DU, G, stagger, idx, j)
            else:
                assert kdim >= K - t * DU >= DU
            covered.append((t * DU, t * DU + min(DU, kdim)))
            kdim -= wrapk if pos == count - 1 else step              # the wrap step happens after the last k-tile
    _check_cover(covered, K, (K, DU, G, stagger))


@pytest.mark.parametrize("DU", [32, 128, 256])
@pytest.mark.parametrize("G", [1, 2, 3, 4])
def test_fold_stagger_model(DU, G):
    for K in list(range(1, 6 * DU + 3, 7)) + [65601]:
        for stagger in (0, 1, 2, 3, 5, 31):
            _simulate_stagger(K, DU, G, stagger)


@pytest.mark.parametrize("bpe", [1, 2, 4, 8])
def test_fold_step_from_inc_model(bpe):
    # the K dim step derived from the address increment inc = k * DU * bpe bytes (k = GSUC ? 1 : GSU, up to the
    # 12/14-bit GSU field) equals k * DU in the 32-bit field at bit 16 of a dword pair, high half included
    # (k * DU >= 2^16); the StaggerU wrap WrapU = -(n - 1) * inc (64-bit) shifted the same way is -(n - 1) * k * DU.
    shift = 16 - (bpe.bit_length() - 1)
    for DU in (16, 32, 64, 128, 256, 512):
        for k in (1, 2, 3, 127, 128, 255, 256, 4095, 16383):
            inc = k * DU * bpe
            assert inc < 2 ** 32
            lo, hi = (inc << shift) & 0xFFFFFFFF, inc >> (32 - shift)
            assert (hi << 32) | lo == (k * DU) << 16
            for n in (0, 1, 2, 7, 1000):
                wrap = (inc - n * inc) & (2 ** 64 - 1)
                assert (wrap << shift) & (2 ** 64 - 1) == (-(n - 1) * k * DU << 16) & (2 ** 64 - 1)


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
        assert "RAP: K dim -=" in assembly and "TailFolding:" not in assembly


# TDM iterate mode on A: each load walks MT rows of A in LdsBlockSizePerPadA / (DU * bpe) = 8 row steps along dim1.
_ITER = [("TDMIterateMode", 1), ("VectorWidthA", 8), ("LdsBlockSizePerPadA", 2048)]
_ITER_MI = dict(mi=(16, 16, 32, 1, 1, 8, 2, 2, 2), wg=(32, 4, 1))


def _k_dim_fields(asm, what):
    # the descriptor SGPRs written by the TailFolding K dim updates whose comment starts with `what`: ({low}, {high})
    tag = r"\], .*// TailFolding: K dim " + re.escape(what)
    lo = set(re.findall(r"^s_sub_u32 s\[(sgprtdm[AB]Group1\+\d)" + tag + r".*\(low half\)", asm, re.M))
    hi = set(re.findall(r"^s_subb_u32 s\[(sgprtdm[AB]Group1\+\d)" + tag + r".*\(high half\)", asm, re.M))
    return lo, hi


_DIM0 = ({"sgprtdmAGroup1+1"}, {"sgprtdmAGroup1+2"})
_DIM0_AB = ({"sgprtdmAGroup1+1", "sgprtdmBGroup1+1"}, {"sgprtdmAGroup1+2", "sgprtdmBGroup1+2"})


@pytest.mark.parametrize("case", ["gsu", "stagger_gsu"])
def test_fold_iterate_mode(tmp_path, case):
    # An iterate-mode tensor keeps K in dim0 (group 1 dwords 1-2) like any unroll-major tensor: the per-step shrink,
    # the GSU split start and the stagger start all update that field, and no tail loop is emitted.
    extra = [("GlobalSplitU", -1), ("GlobalSplitUAlgorithm", "MultipleBuffer")] if case == "gsu" else \
            [("ClusterDim", [1, 1]), ("GlobalSplitU", 3), ("GlobalSplitUAlgorithm", "MultipleBuffer")]
    asm = _emit_one(tmp_path, _config(_ITER + extra, **_ITER_MI), "tf_iter_%s.yaml" % case)
    assert re.search(r"^s_or_b32 s\[sgprtdmAGroup1\], s\[sgprtdmAGroup1\], 0x80000 +// set iterate_enable", asm, re.M)
    assert asm.index(CEIL) < asm.index("tensor_load_to_lds") and "label_TailLoopBeginL" not in asm
    shrink = "-= (GSUC" if case == "gsu" else "-= wrapIter"
    assert _k_dim_fields(asm, shrink) == _DIM0
    assert len(re.findall(r"// TailFolding: K dim %s.* \(low half\)" % re.escape(shrink), asm)) > 0
    if case == "gsu":
        assert len(re.findall(r"// TailFolding: K dim -= \S.* \(low half\)", asm)) == _steps(asm)
    starts = ["= K - split start"] + (["-= stagger start"] if case == "stagger_gsu" else [])
    for what in starts:
        lo, hi = _k_dim_fields(asm, what)
        assert lo and lo <= _DIM0_AB[0] and hi and hi <= _DIM0_AB[1], what
    _check_borrow_pairs(asm)


def test_rap_iterate_mode_k_tail_still_rejected(tmp_path, capsys):
    # RAP keeps rejecting an iterate-mode tensor when it would fold a K remainder (F8xF4 forces ASEM 32 < DepthU).
    import pathlib
    from tensilelite.Components.TDMFuse import tdmKDimTrackable
    src = pathlib.Path(__file__).resolve().parents[1] / "common" / "gemm" / "gfx12" / "rap_gfx1250.yaml"
    config = yaml.safe_load(src.read_text())
    problem = config["BenchmarkProblems"][0]
    keep = ("MatrixInstruction", "ReuseAcrossPersistent", "PrefetchAcrossPersistent")
    problem[1]["ForkParameters"] = [p for p in problem[1]["ForkParameters"] if next(iter(p)) not in keep] + [
        {"MatrixInstruction": [[16, 16, 128, 1, 1, 8, 2, 2, 2]]}, {"ReuseAcrossPersistent": [1]}, {"PrefetchAcrossPersistent": [0]},
        {"TDMIterateMode": [1]}, {"VectorWidthA": [8]}, {"LdsBlockSizePerPadA": [2048]}, {"LdsPadA": [16]}]
    config["BenchmarkProblems"] = [problem]
    config["GlobalParameters"]["PrintSolutionRejectionReason"] = True
    assert _emit(tmp_path, config, "rap_iter.yaml") == []
    assert "ReuseAcrossPersistent with AssertSummationElementMultiple % DepthU != 0 requires TDM descriptor sets that keep K " \
           "in one field (TDMFuse = 0, K-contiguous A and B that step K alike or by bytes, no sparse, no TDM iterate mode)" \
           in capsys.readouterr().out
    # the descriptor-level answer RAP asks for is unchanged; only tail folding allows iterate mode
    from tensilelite.Common.DataType import DataType
    kernel = {"TDMFuse": 0, "NumWaves": 4, "DepthU": 256, "MatrixInstK": 128, "_TDMIterateModeA": True,
              "ProblemType": {"DataTypeA": DataType("F8"), "DataTypeB": DataType("F8"), "TLUA": False, "TLUB": False, "Sparse": 0,
                              "MXBlockA": 0, "MXBlockB": 0}}
    assert tdmKDimTrackable(kernel) is False and tdmKDimTrackable(kernel, allowIterate=True) is True


def _next_free_sgpr(asm):
    return int(re.search(r"\.amdhsa_next_free_sgpr (\d+)", asm).group(1))


def _store_kernarg_config(pgl, amaxd):
    # bias + ScaleAlphaVec + activation (+ amaxD): the store loads 13 kernarg SGPRs after the loop into the SGPRs
    # endSummation frees
    config = _config([("PrefetchGL2", pgl), ("PrefetchGL2SAddr", bool(pgl)), ("MatrixInstruction", [16, 16, 32, 1, 1, 4, 4, 2, 2])])
    problem = config["BenchmarkProblems"][0][0]
    problem.update(UseBias=1, BiasDataTypeList=["b"], UseScaleAB="Scalar", UseScaleAlphaVec=1, Activation=True,
                   ActivationType="all")
    if amaxd:
        problem.update(OutputAmaxD=True, DataTypeAmaxD="s", UseScaleCD=True)
    config["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"] += [{"BiasTypeArgs": ["b"]}, {"ActivationArgs": [[{"Enum": "none"}]]}]
    return config


_SGPR_CASES = {
    "cluster": lambda: _config(),
    "cluster_gsu_gl2": lambda: _config([("GlobalSplitU", 3), ("GlobalSplitUAlgorithm", "MultipleBuffer"), ("PrefetchGL2", 1)]),
    "stagger_gsu": lambda: _config([("ClusterDim", [1, 1]), ("GlobalSplitU", 3), ("GlobalSplitUAlgorithm", "MultipleBuffer")]),
    "bias_sav_act": lambda: _store_kernarg_config(0, 0),
    "bias_sav_act_gl2_amaxd": lambda: _store_kernarg_config(1, 1),
    "iterate": lambda: _config(_ITER, **_ITER_MI),
    "iterate_stagger_gsu": lambda: _config(_ITER + [("ClusterDim", [1, 1]), ("GlobalSplitU", 3), ("GlobalSplitUAlgorithm", "MultipleBuffer")], **_ITER_MI),
}


@pytest.mark.parametrize("case", list(_SGPR_CASES))
def test_fold_sgprs_match_tail_loop_kernel(tmp_path, monkeypatch, case):
    # Folding keeps no SGPR across the loop (only short-lived temporaries), so a folding kernel uses as many
    # SGPRs as the same kernel with the tail loop.
    config = _SGPR_CASES[case]()
    fold = _emit_one(tmp_path, config, "tf_sgpr_fold.yaml")
    assert CEIL in fold and "label_TailLoopBeginL" not in fold
    import sys
    import tensilelite.SolutionStructs.Solution  # noqa: F401  (the package re-exports the class under the same name)
    monkeypatch.setattr(sys.modules["tensilelite.SolutionStructs.Solution"], "tdmSeparateABDescriptors", lambda state: True)   # fold unsupported -> develop code
    dev = _emit_one(tmp_path, config, "tf_sgpr_dev.yaml")
    assert CEIL not in dev and "label_TailLoopBeginL" in dev
    assert _next_free_sgpr(fold) == _next_free_sgpr(dev)


def _next_free_vgpr(asm):
    return int(re.search(r"\.amdhsa_next_free_vgpr (\d+)", asm).group(1))


def _mt32x128x32_bias_sav_config():
    # MT32x128x32 bf16 TN + bias + ScaleAlphaVec + activation: holding the tail-loop VGPRs took it from 144 to
    # 148 VGPRs, one allocation granule (16) more, and cost 32x207648x64 21% on gfx1250
    config = _store_kernarg_config(0, 0)
    problem = config["BenchmarkProblems"][0][0]
    problem.pop("UseScaleAB")
    problem["ActivationType"] = "hipblaslt_all"
    over = {"MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 1, 4], "WorkGroup": [16, 8, 1], "DepthU": 32, "VectorWidthA": 2,
            "VectorWidthB": 2, "StoreVectorWidth": 2, "ClusterDim": [1, 1]}
    fork = config["BenchmarkProblems"][0][1]["ForkParameters"]
    fork[:] = [p for p in fork if next(iter(p)) not in over] + [{k: [v]} for k, v in over.items()]
    return config


_VGPR_CASES = dict(_SGPR_CASES, mt32x128x32_bias_sav=_mt32x128x32_bias_sav_config)


@pytest.mark.parametrize("case", list(_VGPR_CASES))
def test_fold_vgprs_match_tail_loop_kernel(tmp_path, monkeypatch, case):
    # The tail loop releases the misc and LocalReadAddr VGPRs before the store; a folding kernel has no tail loop
    # and must release them too, or the store's VGPR peak rises and can cost occupancy.
    config = _VGPR_CASES[case]()
    fold = _emit_one(tmp_path, config, "tf_vgpr_fold.yaml")
    assert CEIL in fold and "label_TailLoopBeginL" not in fold
    import sys
    import tensilelite.SolutionStructs.Solution  # noqa: F401
    monkeypatch.setattr(sys.modules["tensilelite.SolutionStructs.Solution"], "tdmSeparateABDescriptors", lambda state: True)   # fold unsupported -> develop code
    dev = _emit_one(tmp_path, config, "tf_vgpr_dev.yaml")
    assert CEIL not in dev and "label_TailLoopBeginL" in dev
    assert _next_free_vgpr(fold) == _next_free_vgpr(dev)


def _excluded(kind):
    config = _config()
    problem = config["BenchmarkProblems"][0][0]
    if kind == "mirror_k":
        problem["MirrorDimsA"] = [3]                       # 3 = the unroll (K) index of a batched GEMM
    elif kind == "mixed_bytes":
        problem.update(DataTypeA="f8", DataTypeB="b")
    return config


@pytest.mark.parametrize("kind", ["mirror_k", "mixed_bytes"])
def test_fold_step_not_derivable_keeps_tail_loop(tmp_path, monkeypatch, kind):
    # Solution does not fold a kernel whose K dim step cannot be derived from its TDM address increment: it
    # keeps the tail loop (never a reject). (Swizzled tensors also need
    # DirectToVgpr, UseInitialStridesAB is rejected on gfx1250 and BufferLoad 0 rejects PGR 2, so those
    # never reach the gate in a valid config.)
    seen = _record_fold(monkeypatch)
    results = _emit(tmp_path, _excluded(kind), "tf_excl_%s.yaml" % kind)
    assert results and seen and not any(seen)
    for _, assembly, error in results:
        assert error == 0
        assert CEIL not in assembly and "TailFolding:" not in assembly and "label_TailLoopBeginL" in assembly


def _single_loop_optnll_config():
    # The OOB kernel MT256x128x32 (bf16 TN, PGR2, PLR1, SIA4) whose fold failed only in the hipblaslt build:
    # K <= DU folds to one unroll iteration, which takes the PGR2 one-tile escape to toPGR1, and an edge
    # tile then enters the ordinary NLL behind the OptNLL dispatch, which has no barrier of its own.
    return _config([("DepthU", 32), ("LdsBlockSizePerPadA", 512), ("VectorWidthA", 8), ("VectorWidthB", 4),
                    ("StoreVectorWidth", 8), ("ClusterDim", [1, 1]), ("ClusterLocalRead", 0), ("LDSTrInst", True),
                    ("LocalReadVectorWidth", -1), ("StaggerU", 32), ("StaggerUStride", 256)],
                   mi=(16, 16, 32, 1, 1, 8, 4, 2, 2), wg=(32, 4, 1))


@pytest.mark.parametrize("no_comment", [False, True], ids=["comments", "disable_asm_comments"])
def test_fold_single_loop_escape_keeps_its_sync(tmp_path, no_comment):
    # The barrier planner keeps the one-tile escape barrier by its comment. The hipblaslt build passes
    # --disable-asm-comments, under which str() of the barrier has no comment: the planner dropped the
    # barrier and its tensorcnt drain, so the ordinary NLL of an edge tile read LDS before the TDM tile
    # landed (K <= DU with the fold; K in [DU, 2*DU) without it).
    from rocisa import rocIsa
    ri = rocIsa.getInstance()
    opts = ri.getOutputOptions()
    previous = opts.outputNoComment
    opts.outputNoComment = no_comment
    ri.setOutputOptions(opts)
    try:
        asm = _emit_one(tmp_path, _single_loop_optnll_config(), "tf_single_loop_%u.yaml" % no_comment)
    finally:
        opts = ri.getOutputOptions()
        opts.outputNoComment = previous
        ri.setOutputOptions(opts)
    assert CEIL in asm and "label_TailLoopBeginL" not in asm
    assert "label_OptNLL_End:" in asm, "no ordinary NLL behind an OptNLL dispatch"
    ins = _instructions(asm)
    i = next(n for n, l in enumerate(ins) if l.startswith("s_cbranch_scc0 label_AfterSingleLoopPGRSync"))
    j = next(n for n in range(i, len(ins)) if ins[n].startswith("s_branch label_toPGR1"))
    escape = [l.split("//")[0].strip() for l in ins[i + 1:j]]
    assert "s_wait_tensorcnt 0" in escape and escape[-2:] == ["s_barrier_signal -1", "s_barrier_wait -1"], escape
