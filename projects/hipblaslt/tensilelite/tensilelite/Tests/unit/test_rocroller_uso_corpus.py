# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Only verified, row-uniform rocRoller kernels may declare uniform summation order.

Uniform summation order is on by default, and under it hipBLASLt refuses every
hand-written custom kernel unless its ``CustomKernel`` block declares
``uniformSummationOrder: true``. The declaration is a claim the library cannot
check at run time: that the kernel sums every row of D in the same order. A
wrong claim is a silent correctness bug, so a checked-in rocRoller kernel may
carry it only when it is listed in ``data/rocroller_uso_verified.txt``, the
kernels shown to produce bitwise-identical rows of D for identical rows of A.

The flag is written twice, in the ``custom.config`` of the ``.s`` (what the
library build reads) and in the solution's ``CustomKernel`` block of the logic
file. The two must agree, or the logic file misdescribes what ships.
"""

from pathlib import Path

import pytest
import yaml

from logic_corpus import needs_logic_corpus
from rocroller_logic_corpus import corpus, rocroller_logic_files
from tensilelite.CustomKernels import readCustomKernelConfig

pytestmark = pytest.mark.unit

RR_KERNEL_PREFIX = "RR_GEMM_"

_HERE = Path(__file__).resolve().parent
_VERIFIED = _HERE / "data" / "rocroller_uso_verified.txt"
_KERNELS = _HERE.parents[1] / "CustomKernels" / "rocroller"


def _verified():
    names = set()
    for line in _VERIFIED.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


@pytest.fixture(scope="module")
def shipped():
    """CustomKernel block of every shipped rocRoller kernel, from its custom.config."""
    kernels = {}
    for path in sorted(_KERNELS.glob(RR_KERNEL_PREFIX + "*.s")):
        config = readCustomKernelConfig(path.stem, str(_KERNELS))
        kernels[path.stem] = config["CustomKernel"]
    assert kernels, f"no {RR_KERNEL_PREFIX}*.s kernels under {_KERNELS}"
    return kernels


@pytest.fixture(scope="module")
def logic_blocks():
    """CustomKernel block of every rocRoller solution in the gfx950 logic files."""
    blocks = {}
    for path in rocroller_logic_files():
        doc = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        for solution in doc["Solutions"]:
            name = (solution.get("CustomKernel") or {}).get("name", "")
            if name.startswith(RR_KERNEL_PREFIX):
                blocks.setdefault(name, []).append((path.name, solution["CustomKernel"]))
    assert blocks, f"no {RR_KERNEL_PREFIX} solutions under {corpus()}"
    return blocks


def test_every_flagged_kernel_is_verified(shipped):
    verified = _verified()
    unverified = sorted(
        name for name, ck in shipped.items()
        if ck.get("uniformSummationOrder") and name not in verified
    )
    assert not unverified, (
        f"kernels declare uniformSummationOrder without being in {_VERIFIED.name}: {unverified}"
    )


def test_every_verified_kernel_ships_flagged(shipped):
    verified = _verified()
    missing = sorted(verified - shipped.keys())
    assert not missing, f"{_VERIFIED.name} lists kernels that are not shipped: {missing}"
    unflagged = sorted(name for name in verified if not shipped[name].get("uniformSummationOrder"))
    assert not unflagged, f"verified kernels do not declare uniformSummationOrder: {unflagged}"


@needs_logic_corpus
def test_logic_files_agree_with_custom_config(shipped, logic_blocks):
    disagree = []
    for name, entries in logic_blocks.items():
        assert name in shipped, f"logic references a kernel that is not shipped: {name}"
        expected = bool(shipped[name].get("uniformSummationOrder", False))
        for logic_file, ck in entries:
            if bool(ck.get("uniformSummationOrder", False)) != expected:
                disagree.append(f"{logic_file}: {name}")
    assert not disagree, f"logic CustomKernel.uniformSummationOrder differs from custom.config: {disagree}"
