# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Provider-owned repository discovery supplies portable rocKE test inputs."""

from pathlib import Path
import subprocess

import pytest

PROVIDER = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "inputs", ["present", "absent", "template_only", "bundles_only"]
)
@pytest.mark.parametrize("overrides", ["none", "both", "template", "empty"])
def test_provider_discovery_preserves_explicit_inputs(tmp_path, inputs, overrides):
    root = tmp_path / "repository"
    template = (
        root
        / "projects/hipdnn/tools/IngestorGenerator/configs/templates/attention_sdpa_fwd.graph.json"
    )
    bundles = root / "dnn-providers/integration-tests/integration-test-bundles"
    if inputs in ("present", "template_only"):
        template.parent.mkdir(parents=True)
        template.write_text("{}")
    if inputs in ("present", "bundles_only"):
        bundles.mkdir(parents=True)
    setup = {
        "none": "",
        "both": 'set(ROCKE_HIPDNN_SDPA_TEMPLATE "caller")\nset(ROCKE_HIPDNN_SDPA_BUNDLES_DIR "caller-dir")\n',
        "template": 'set(ROCKE_HIPDNN_SDPA_TEMPLATE "caller")\n',
        "empty": 'set(ROCKE_HIPDNN_SDPA_TEMPLATE "")\nset(ROCKE_HIPDNN_SDPA_BUNDLES_DIR "")\n',
    }[overrides]
    script = tmp_path / "provider.cmake"
    script.write_text(
        setup + f'include("{(PROVIDER / "cmake/RockeTestAssets.cmake").as_posix()}")\n'
        f'hipkernelprovider_configure_rocke_test_assets("{root.as_posix()}")\n'
        'file(WRITE "${CMAKE_CURRENT_LIST_DIR}/provider-inputs.txt" "${ROCKE_HIPDNN_SDPA_TEMPLATE}\\n${ROCKE_HIPDNN_SDPA_BUNDLES_DIR}")\n'
    )
    subprocess.run(
        ["cmake", "-P", str(script)], check=True, capture_output=True, text=True
    )
    actual = (tmp_path / "provider-inputs.txt").read_text().split("\n")
    expected = (
        ["caller", "caller-dir"]
        if overrides == "both"
        else (
            ["caller", ""]
            if overrides == "template"
            else (
                [template.as_posix(), bundles.as_posix()]
                if overrides == "none" and inputs == "present"
                else ["", ""]
            )
        )
    )
    assert actual == expected


def test_provider_wires_discovery_before_the_sdk_subdirectory():
    parent = (PROVIDER / "CMakeLists.txt").read_text()
    assert "include(RockeTestAssets)" in parent
    assert parent.index(
        "hipkernelprovider_configure_rocke_test_assets("
    ) < parent.index("    add_subdirectory(rocke)\n")
