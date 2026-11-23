# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tail folding of the K tail into the TDM unroll loop (gfx1250), fixed at codegen.

Every kernel whose structure allows it (any input type but packed sub-byte F4/F6, TDM A/B, PGR 2, ...) folds: the unroll loop runs
ceil(K/DU) iterations, the TDM K dim (shrunk at every address step) zero-fills the partial last k-tile, and
no tail loop is emitted. Any other kernel keeps the develop tail loop. With GSU the folded k-tile count is
distributed; each split's K dim starts at its own first k-tile (GSUC 0: interleaved, GSUC 1: contiguous).
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


def _no_step_sgpr(monkeypatch, derive):
    # derive=False: force the TailFoldStep SGPR fallback (a kernel whose K dim step is not its address increment)
    if not derive:
        import tensilelite.KernelWriterAssembly as kwa
        monkeypatch.setattr(kwa.KernelWriterAssembly, "tailFoldStepShift", lambda self, kernel: None)


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


def test_fold_no_gsu_immediate_step(tmp_path):
    # GlobalSplitU 0 (no runtime GSU) and a cluster (no StaggerU): the K dim step is a codegen constant, no SGPRs.
    results = _emit(tmp_path, _config([("GlobalSplitU", 0)]), "tf_gsu0.yaml")
    if len(results) != 1:
        pytest.skip("GlobalSplitU 0 is rejected for this config")
    _, asm, error = results[0]
    assert error == 0
    assert CEIL in asm and "label_TailLoopBeginL" not in asm and "sgprTailFoldStep" not in asm
    assert asm.count("// TailFolding: K dim -= 128 (low half)") == _steps(asm)


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


FALLBACK = "tail folding does not fit"


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


def test_fold_subbyte_keeps_tail_loop(tmp_path, monkeypatch, capsys):
    # Packed sub-byte inputs (F6 here): the TDM K dim does not step in elements, so Solution does not
    # fold them and codegen emits the tail loop directly (no fallback); never a reject.
    seen = _record_fold(monkeypatch)
    results = _emit(tmp_path, _f6_config(), "tf_f6.yaml", limit=64)
    assert results and seen and not any(seen)
    for _, assembly, error in results:
        assert error == 0
        assert CEIL not in assembly and "label_TailLoopBeginL" in assembly
    assert FALLBACK not in capsys.readouterr().out


def test_fold_subbyte_forced_is_an_error(tmp_path, monkeypatch, capsys):
    # The codegen check stays an internal error: a sub-byte kernel forced to fold is rejected, not
    # silently regenerated with the tail loop.
    import sys
    import tensilelite.SolutionStructs.Solution  # noqa: F401
    monkeypatch.setattr(sys.modules["tensilelite.SolutionStructs.Solution"], "tdmKDimStepsInElements", lambda state: True)
    seen = _record_fold(monkeypatch)
    results = _emit(tmp_path, _f6_config(), "tf_f6_forced.yaml", limit=64)
    out = capsys.readouterr().out
    assert any(seen)
    assert any(error != 0 for _, _, error in results)
    assert "the A/B K dim step must be DepthU" in out and FALLBACK not in out


def test_fold_off_when_illegal(tmp_path):
    # PrefetchGlobalRead 1 is not legal for folding -> the kernel keeps the plain tail loop; never a reject.
    asm = _emit_one(tmp_path, _config([("PrefetchGlobalRead", 1)]), "tf_pgr1.yaml")
    assert "TailFold" not in asm and "label_TailLoopBeginL" in asm


@pytest.mark.parametrize("derive", [True, False], ids=["from_inc", "step_sgpr"])
@pytest.mark.parametrize("gsu", [1, 3])
def test_fold_stagger(tmp_path, monkeypatch, gsu, derive):
    # No cluster -> StaggerU: the split starts StaggerUIter steps in and the wrap back to its first k-tile adds
    # (splitIters-1)*step to the K dim. from_inc: both are the stagger offset / the selected increment (WrapU on
    # the wrapIter) in K dim units, so neither TailFoldStep nor TailFoldWrapK exists.
    _no_step_sgpr(monkeypatch, derive)
    asm = _emit_one(tmp_path, _config([("ClusterDim", [1, 1]), ("GlobalSplitU", gsu), ("GlobalSplitUAlgorithm", "MultipleBuffer")]),
                    "tf_stagger.yaml")
    if derive:
        tags = ("// TailFolding: stagger start = stagger offset << 15", "// TailFolding: K dim step = (wrapIter ? WrapU : inc) << 15")
        assert "sgprTailFold" not in asm
        assert len(re.findall(r"^s_lshl_b64 .*// TailFolding: K dim step = \(wrapIter \? WrapU : inc\) << 15", asm, re.M)) \
            == asm.count("TailFolding: K dim -= wrapIter ? -(splitIters-1)*step : step (low half)")
    else:
        tags = ("// TailFolding: stagger start = StaggerUIter * step", "// TailFolding: wrap K step = -(splitIters - 1) * step",
                "// TailFolding: K dim step = wrapIter ? -(splitIters-1)*step : step (low)")
    for tag in tags + ("// TailFolding: K dim -= stagger start (low half)", CEIL):
        assert tag in asm, tag
    _check_borrow_pairs(asm)
    assert "label_TailLoopBeginL" not in asm and "TailFoldOn" not in asm
    assert asm.index(CEIL) < asm.index("tensor_load_to_lds")
    assert asm.count("TailFolding: K dim -= wrapIter ? -(splitIters-1)*step : step (low half)") >= 1


@pytest.mark.parametrize("derive", [True, False], ids=["from_inc", "step_sgpr"])
@pytest.mark.parametrize("gsu", [2, 3, 4, -1])
def test_fold_gsu(tmp_path, monkeypatch, gsu, derive):
    # GSUC is a run-time bit: the k-tile step is GSUC ? 1 : GSU k-tiles and every GSUC == 1 start offset is
    # taken from the folded count ceil(K/DU). from_inc: the step is the wave's TDM address increment
    # ((GSUC ? 1 : GSU) * DU * 2 bytes) << 15, low and high half, at every address step; no TailFoldStep.
    _no_step_sgpr(monkeypatch, derive)
    asm = _emit_one(tmp_path, _config([("GlobalSplitU", gsu), ("GlobalSplitUAlgorithm", "MultipleBuffer"), ("PrefetchGL2", 1)]), "tf_gsu.yaml")
    if derive:
        assert "sgprTailFold" not in asm
        lo = re.findall(r"^s_lshl_b32 (s\d+), s\[sgprtdmABIncs\], 0xf +// TailFolding: K dim step = address increment << 15 \(low half\)", asm, re.M)
        hi = re.findall(r"^s_lshr_b32 (s\d+), s\[sgprtdmABIncs\], 0x11 +// TailFolding: K dim step = address increment << 15 \(high half\)", asm, re.M)
        assert len(lo) == len(hi) == _steps(asm) > 0
    else:
        assert "// TailFolding: k-tiles per step = GSUC ? 1 : GSU" in asm
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


def test_fold_asm_tag(tmp_path):
    # every instruction using a fold SGPR carries a "TailFolding:" comment
    asm = _emit_one(tmp_path, _config([("PrefetchGL2", 1)]), "tf_tag.yaml")
    for line in asm.split("\n"):
        if re.search(r"sgprTailFold(Step|WrapK)", line) and not line.startswith(".set"):
            assert "// TailFolding:" in line, line


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
            kdim -= tile_step * DU                             # TailFoldStep
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


@pytest.mark.parametrize("stagger", [0, 1], ids=["cluster", "stagger"])
def test_fold_sgprs_released_before_store(tmp_path, monkeypatch, stagger):
    # The fold SGPRs are dead after the loop: endSummation releases all of them, so the store's
    # SGPR block can reuse them (an alignment hole tagged with a fold name used to keep one live).
    # (TailFoldStep / TailFoldWrapK only exist where the step is not derived from the address increment.)
    _no_step_sgpr(monkeypatch, False)
    extra = [("ClusterDim", [1, 1])] if stagger else []
    asm = _emit_one(tmp_path, _config(extra), "tf_rel%u.yaml" % stagger)
    names = ["TailFoldStep"] + (["TailFoldWrapK"] if stagger else [])
    if re.search(r"^\.set sgprTailFoldPad,", asm, re.M):
        names.append("TailFoldPad")
    for n in names:
        assert re.search(r"^\.set sgpr%s, UNDEF" % n, asm, re.M), n


def _next_free_sgpr(asm):
    return int(re.search(r"\.amdhsa_next_free_sgpr (\d+)", asm).group(1))


@pytest.mark.parametrize("amaxd", [0, 1], ids=["bias_sav_act", "amaxd"])
@pytest.mark.parametrize("pgl", [0, 1])
def test_fold_store_kernargs_reuse_loop_sgprs(tmp_path, monkeypatch, amaxd, pgl):
    # bias + ScaleAlphaVec + activation (+ amaxD) kernels load 13 store-kernarg SGPRs after the loop
    # into the SGPRs endSummation frees. An alignment hole tagged with the next SGPR's name must not
    # keep that SGPR live, or the store block moves past every loop SGPR (+9..+12 SGPRs). The fold
    # kernel stays within the develop SGPR count (+ alignment).
    config = _config([("PrefetchGL2", pgl), ("PrefetchGL2SAddr", bool(pgl)), ("MatrixInstruction", [16, 16, 32, 1, 1, 4, 4, 2, 2])])
    problem = config["BenchmarkProblems"][0][0]
    problem.update(UseBias=1, BiasDataTypeList=["b"], UseScaleAB="Scalar", UseScaleAlphaVec=1, Activation=True,
                   ActivationType="all")
    final = [{"BiasTypeArgs": ["b"]}, {"ActivationArgs": [[{"Enum": "none"}]]}]
    if amaxd:
        problem.update(OutputAmaxD=True, DataTypeAmaxD="s", UseScaleCD=True)
    config["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"] += final
    fold = _emit_one(tmp_path, config, "tf_store_fold.yaml")
    assert CEIL in fold and "label_TailLoopBeginL" not in fold
    import sys
    import tensilelite.SolutionStructs.Solution  # noqa: F401  (the package re-exports the class under the same name)
    monkeypatch.setattr(sys.modules["tensilelite.SolutionStructs.Solution"], "tdmSeparateABDescriptors", lambda state: True)   # fold unsupported -> develop code
    dev = _emit_one(tmp_path, config, "tf_store_dev.yaml")
    assert CEIL not in dev and "label_TailLoopBeginL" in dev
    assert _next_free_sgpr(fold) - _next_free_sgpr(dev) <= 4


def test_fold_overflow_falls_back(tmp_path, monkeypatch, capsys):
    # A kernel whose fold SGPRs do not fit is generated again with the tail loop (develop code):
    # never a reject. The overflow verdict makes the first generation raise; simulated by 40 extra
    # SGPRs defined only when the kernel folds.
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    define = KernelWriterAssembly.defineVariableSgprs

    def bloated(self, kernel):
        module = define(self, kernel)
        if kernel.get("_TailFolding", False):
            module.add(self.defineSgpr("TailFoldTestBloat", 40))
        return module

    monkeypatch.setattr(KernelWriterAssembly, "defineVariableSgprs", bloated)
    asm = _emit_one(tmp_path, _config(), "tf_overflow.yaml")
    assert CEIL not in asm and "label_TailLoopBeginL" in asm and "TailFoldTestBloat" not in asm
    assert FALLBACK in capsys.readouterr().out


def test_fold_vgpr_overflow_not_retried(tmp_path, monkeypatch, capsys):
    # Only an SGPR overflow falls back to the tail loop. Any other verdict (a VGPR overflow here,
    # simulated on the fold attempt only) rejects the kernel as it would without folding.
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    judge = KernelWriterAssembly.judgeRegisterBudget

    def overflowing(self, kernel, *args, **kwargs):
        judge(self, kernel, *args, **kwargs)
        if kernel.get("_TailFolding", False):
            self.states.overflowedResources = 1

    monkeypatch.setattr(KernelWriterAssembly, "judgeRegisterBudget", overflowing)
    results = _emit(tmp_path, _config(), "tf_vgpr_overflow.yaml")
    out = capsys.readouterr().out
    assert len(results) == 1 and results[0][2] != 0
    assert "resulted in error 1" in out and FALLBACK not in out
