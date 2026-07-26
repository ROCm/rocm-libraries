# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.Common.Architectures import gfxToIsa
from Tensile.Toolchain.Assembly import replaceAssemblyTarget, targetArchForIsa

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "target",
    ["gfx90c", "gfx90c:xnack+", "gfx90c:xnack-"],
)
def test_target_arch_for_isa_preserves_requested_features(target):
    assert targetArchForIsa(gfxToIsa(target), [target]) == target


def test_target_arch_for_isa_uses_bare_fallback_without_matching_request():
    assert targetArchForIsa(gfxToIsa("gfx90c"), ["gfx942:xnack+"]) == "gfx90c"


def test_target_arch_for_isa_uses_bare_target_for_multiple_variants():
    assert targetArchForIsa(
        gfxToIsa("gfx90c"),
        ["gfx90c:xnack+", "gfx90c:xnack-"],
    ) == "gfx90c"


def test_replace_assembly_target_preserves_xnack_feature():
    isa = gfxToIsa("gfx90c")
    source = '.amdgcn_target "amdgcn-amd-amdhsa--gfx90c"\n.text\n'
    assert replaceAssemblyTarget(source, isa, "gfx90c:xnack+") == (
        '.amdgcn_target "amdgcn-amd-amdhsa--gfx90c:xnack+"\n.text\n'
    )
