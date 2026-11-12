# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Solution checks for TDM with 1LDSBuffer on gfx1250."""
import pytest
import yaml

from config_harness import solutions_from_config

pytestmark = pytest.mark.unit

_REJECT_REASON = "reject: 1LDSBuffer must be 0 for TDM"


def _tdm_config(**forks):
    """F8 TDM LDSTrInst config, as in lds_segment_interleave_2.yaml."""
    params = {
        "MatrixInstruction": [[16, 16, 64, 1, 1, 2, 2, 2, 2]],
        "DepthU": [64],
        "WavefrontSize": [32],
        "TDMInst": [3],
        "LDSTrInst": [True],
        "ExpandPointerSwap": [False],
        "PrefetchGlobalRead": [1],
        "ScheduleIterAlg": [3],
        "1LDSBuffer": [0],
    }
    params.update({k: [v] for k, v in forks.items()})
    return {
        "GlobalParameters": {"Architecture": "gfx1250", "PrintLevel": 0,
                             "PrintSolutionRejectionReason": True},
        "BenchmarkProblems": [[
            {
                "OperationType": "GEMM",
                "DataType": "F8",
                "DestDataType": "s",
                "ComputeDataType": "s",
                "HighPrecisionAccumulate": True,
                "TransposeA": False,
                "TransposeB": False,
                "UseBeta": True,
                "Batched": True,
            },
            {
                "InitialSolutionParameters": None,
                "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
                "ForkParameters": [{k: v} for k, v in params.items()],
                "BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [128, 128, 1, 128]}]}],
            },
        ]],
    }


def _derive(tmp_path, monkeypatch, capsys, **forks):
    """(solutions, reject lines) for the config, derived serially so the reasons are captured."""
    import tensilelite.BenchmarkProblems as benchmark_problems

    def serial_map(function, objects, *_args, **_kwargs):
        return [function(*args) for args in objects]

    monkeypatch.setattr(benchmark_problems, "ParallelMap2", serial_map)
    path = tmp_path / "tdm_1ldsbuffer.yaml"
    path.write_text(yaml.safe_dump(_tdm_config(**forks)))
    sols = solutions_from_config(path, arch="gfx1250")
    rejects = {line.strip() for line in capsys.readouterr().out.splitlines()
               if line.startswith("reject:")}
    return sols, rejects


def test_tdm_1ldsbuffer_0_is_valid(tmp_path, monkeypatch, capsys):
    sols, rejects = _derive(tmp_path, monkeypatch, capsys)
    assert len(sols) == 1, rejects
    assert sols[0]["1LDSBuffer"] == 0


def test_tdm_resolves_auto_1ldsbuffer_to_0(tmp_path, monkeypatch, capsys):
    sols, rejects = _derive(tmp_path, monkeypatch, capsys, **{"1LDSBuffer": -1})
    assert len(sols) == 1, rejects
    assert sols[0]["1LDSBuffer"] == 0


@pytest.mark.parametrize("prefetchGlobalRead", [1, 2], ids=["pgr1", "pgr2"])
def test_tdm_rejects_1ldsbuffer_1(tmp_path, monkeypatch, capsys, prefetchGlobalRead):
    # The next tile's tensor loads would write the only LDS buffer while other
    # waves are still reading it.
    sols, rejects = _derive(tmp_path, monkeypatch, capsys,
                            PrefetchGlobalRead=prefetchGlobalRead, **{"1LDSBuffer": 1})
    assert not sols
    assert _REJECT_REASON in rejects
