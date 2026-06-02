# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Regression tests for the dnn-benchmarking setup helper."""

import subprocess
from pathlib import Path


SETUP_SCRIPT = Path(__file__).resolve().parents[3] / "setup.sh"


def test_setup_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], check=True)


def test_setup_checks_provider_artifacts_independently() -> None:
    script = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert 'MIOPEN_PLUGIN="$PLUGIN_DIR/libmiopen_plugin.so"' in script
    assert 'HIP_KERNEL_PLUGIN="$PLUGIN_DIR/libhip_kernel_provider.so"' in script
    assert 'HIPBLASLT_PLUGIN="$PLUGIN_DIR/libhipblaslt_plugin.so"' in script

    hipdnn_block = 'if needs_install "$HIPDNN_CONFIG"; then'
    miopen_block = '[ "$BUILT_HIPDNN" -eq 1 ] || needs_install "$MIOPEN_PLUGIN"'
    hipblaslt_block = '[ "$BUILT_HIPDNN" -eq 1 ] || needs_install "$HIPBLASLT_PLUGIN"'
    hip_kernel_block = '[ "$BUILT_HIPDNN" -eq 1 ] || needs_install "$HIP_KERNEL_PLUGIN"'

    assert hipdnn_block in script
    assert miopen_block in script
    assert hipblaslt_block in script
    assert hip_kernel_block in script

    assert script.index(hipdnn_block) < script.index(miopen_block)
    assert script.index(hipdnn_block) < script.index(hipblaslt_block)
    assert script.index(hipdnn_block) < script.index(hip_kernel_block)
