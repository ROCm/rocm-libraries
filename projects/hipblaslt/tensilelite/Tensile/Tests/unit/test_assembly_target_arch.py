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


@pytest.mark.parametrize("target", ["gfx90c:xnack+", "gfx90c:xnack-"])
def test_replace_assembly_target_preserves_xnack_feature(target):
    isa = gfxToIsa("gfx90c")
    source = '.amdgcn_target "amdgcn-amd-amdhsa--gfx90c"\n.text\n'
    assert replaceAssemblyTarget(source, isa, target) == (
        f'.amdgcn_target "amdgcn-amd-amdhsa--{target}"\n.text\n'
    )


def test_replace_assembly_target_leaves_bare_source_unchanged():
    isa = gfxToIsa("gfx90c")
    source = '.amdgcn_target "amdgcn-amd-amdhsa--gfx90c"\n.text\n'
    assert replaceAssemblyTarget(source, isa, "gfx90c") == source


@pytest.mark.parametrize(
    "source",
    [
        ".text\n",
        (
            '.amdgcn_target "amdgcn-amd-amdhsa--gfx90c"\n'
            '.amdgcn_target "amdgcn-amd-amdhsa--gfx90c"\n'
        ),
    ],
)
def test_replace_assembly_target_rejects_ambiguous_source(source):
    with pytest.raises(RuntimeError, match=r"Expected one gfx90c .* found (0|2)"):
        replaceAssemblyTarget(source, gfxToIsa("gfx90c"), "gfx90c:xnack+")
