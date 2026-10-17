# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native classic PAP descriptor snapshots retain K0 across store and entry."""

import re

import pytest
from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config
from test_pap_pipeline_overlap_gfx1250 import _config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("cluster", [[1, 1], [2, 4]])
def test_descriptor_snapshot_captures_k0_and_skips_next_entry_rebuild(tmp_path, cluster):
    results = emit_kernels_from_config(
        _config(tmp_path, AssertSummationElementMultiple=256, ClusterDim=cluster),
        limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        (tmp_path / "descriptor_snapshot.s").write_text(src)
        lines = src.splitlines()
        saves = [i for i, line in enumerate(lines) if "PAP retain next K0 descriptor" in line]
        restores = [i for i, line in enumerate(lines) if "PAP activate retained K0 descriptor" in line]
        assert len(saves) == len(restores) == 28
        saved = [re.search(r"v_writelane_b32\s+(v[^,]+),[^,]+,\s*(\d+)", lines[i]).groups()
                 for i in saves]
        restored = [re.search(r"v_readlane_b32\s+[^,]+,\s*(v[^,]+),\s*(\d+)", lines[i]).groups()
                    for i in restores]
        assert len({register for register, _ in saved}) == 1
        assert [int(lane) for _, lane in saved] == list(range(4, 32))
        assert restored == saved, "descriptor restores must use the saved VGPR and lanes"
        issue0 = next(i for i, line in enumerate(lines) if "PAP PGR2 group0 actually issued" in line)
        group1 = next(i for i, line in enumerate(lines) if "PAP group1 positive DepthU increment" in line)
        assert issue0 < saves[0] < saves[-1] < group1
        assert src.count("PAP snapshot descriptor LDS address in bank zero") == 2
        cold = next(i for i, line in enumerate(lines) if line.startswith("label_PapColdEntryDescriptors:"))
        ready = next(i for i, line in enumerate(lines) if line.startswith("label_PapEntryDescriptorsReady:"))
        assert restores[-1] < cold < ready
        assert any("TDM" in line for line in lines[cold:ready])
        assert "PAP retained descriptors belong to an issued tile?" in src
        assert "PAP consume retained descriptor snapshot" in src
        assert "PAP second PGR group already issued?" in src
        assert "PAP skip redundant first local reads" in src
        assert "PAP activate retained WorkGroup0" in src
        assert max(map(int, re.findall(r"\.amdhsa_next_free_sgpr\s+(\d+)", src))) <= 106
