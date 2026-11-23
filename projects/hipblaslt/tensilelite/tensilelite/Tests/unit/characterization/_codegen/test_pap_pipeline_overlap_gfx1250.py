# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only assembly regressions for PAP synchronization and SGPR allocation."""
from pathlib import Path

import pytest
import yaml
from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config

pytestmark = pytest.mark.unit
_BASE_CONFIG = (Path(__file__).resolve().parent / "data/test_data/_designed/gfx1250"
                / "dataparallel_pap_address_pipeline.yaml")


def _config(tmp_path, **overrides):
    """Derive one code-generation case from the existing tracked PAP fixture."""
    config = yaml.safe_load(_BASE_CONFIG.read_text())
    forks = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for entry in forks:
        for key in entry.keys() & overrides.keys():
            entry[key] = [overrides[key]]
    path = tmp_path / "pap_codegen.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _assert_classic_pap_overlaps_nll(lines):
    marker = next(i for i, line in enumerate(lines) if 'PAP descriptor in high LDS bank' in line)
    issues = [i for i in range(marker) if lines[i].startswith('tensor_load_to_lds ')]
    issue = issues[-2]  # data and MX-scale producers
    # WAR: peers' reads of the refilled bank complete before PAP overwrites it.
    wait = max(i for i in range(issue) if lines[i].startswith('s_barrier_wait -1'))
    signal = max(i for i in range(wait) if lines[i].startswith('s_barrier_signal -1'))
    assert not any(line.startswith(('ds_', 'tensor_')) for line in lines[signal:issue])
    nll_end = next(i for i in range(marker, len(lines)) if lines[i].startswith('label_toPGR1end_OrdNLL:'))
    assert not any(line.startswith('s_wait_tensorcnt') for line in lines[issue:nll_end])
    entry = next(i for i, line in enumerate(lines) if line.startswith('label_PersistentLoopStart:'))
    first_read = next(i for i in range(entry, len(lines)) if lines[i].startswith('ds_load_'))
    assert any(line.startswith('s_wait_tensorcnt') for line in lines[entry:first_read])
    return issue, signal, wait, nll_end, entry, first_read


def test_classic_pap_load_is_not_drained_in_no_load_loop(tmp_path):
    # The NLL reads one bank while PAP refills the other. A PAP load tagged with
    # the NLL's token makes the token planner drain it before the NLL's reads.
    results = emit_kernels_from_config(_config(tmp_path), limit=1, arch='gfx1250')
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        lines = src.splitlines()
        issue, signal, wait, nll_end, entry, first_read = _assert_classic_pap_overlaps_nll(lines)
        corruptions = [
            [line for i, line in enumerate(lines) if i not in (signal, wait)],
            lines[:nll_end] + ['s_wait_tensorcnt 0'] + lines[nll_end:],
            [line for i, line in enumerate(lines)
             if not (entry < i < first_read and line.startswith('s_wait_tensorcnt'))],
        ]
        for corrupted in corruptions:
            with pytest.raises(AssertionError):
                _assert_classic_pap_overlaps_nll(corrupted)
