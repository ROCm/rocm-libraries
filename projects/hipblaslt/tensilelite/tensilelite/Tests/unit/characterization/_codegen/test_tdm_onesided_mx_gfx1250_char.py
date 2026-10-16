################################################################################
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 TDM kernels with MX scales on one tensor only (CPU-only characterization).

Emits one kernel per emitting cell, from a TN F8 x F8 config with MX block 32 and E8
scales on A only (MXSA) or on B only (MXSB): TDMFuse=0 on one, two and four waves, and
TDMFuse 1-3 on two and four waves (Solution rejects them on one wave). Each kernel is
checked at the register level:

* the dead scale's address SGPR is never read;
* the live scale's descriptor is programmed: an instruction writes the Group1 registers
  of the set carrying it, its own set on TDMFuse=0 and its data tensor's set on
  TDMFuse 1-3;
* every live descriptor set issues a tensor_load_to_lds inside the main loop, and no
  wave issues a descriptor that only the dead scale owns.

SGPR names resolve through the kernel's ``.set`` table, so a member that aliases its set
owner's registers counts as that set.

CPU-only: the emit harness runs Python+rocisa codegen; nothing is assembled or launched.
"""

import re

import pytest
import yaml

from config_harness import derive_states, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

# 64x64 on one wave and on waves [2,1]; 64x128 on waves [2,2].
_MATRIX_INSTRUCTION = {
    1: [16, 16, 128, 1, 1, 4, 4, 1, 1],
    2: [16, 16, 128, 1, 1, 2, 4, 2, 1],
    4: [16, 16, 128, 1, 1, 2, 4, 2, 2],
}

_CELLS = [(fuse, scale, numWaves)
          for fuse in (0, 1, 2, 3)
          for numWaves in ((1, 2, 4) if fuse == 0 else (2, 4))
          for scale in ("MXSA", "MXSB")]
_CELL_IDS = ["tdmfuse%d-%s-w%d" % cell for cell in _CELLS]

# The data tensor whose descriptor set the lone scale rides on TDMFuse 1-3.
_SCALE_SET_OWNER = {(1, "MXSA"): "A", (1, "MXSB"): "B", (2, "MXSA"): "A",
                    (2, "MXSB"): "A", (3, "MXSA"): "B", (3, "MXSB"): "B"}

_GROUP1_SGPRS = 8

_FORK = {
    "DepthU": 256,
    "TDMInst": 3,
    "TDMSplit": False,
    "PrefetchGlobalRead": 2,
    "ScheduleIterAlg": 4,
    "StaggerU": 0,
    "TransposeLDS": -1,
    "LdsPadA": -1,
    "LdsPadB": -1,
    "LdsBlockSizePerPadA": -1,
    "LdsBlockSizePerPadB": -1,
    "LdsPadMetadata": 0,
    "LDSTrInst": False,
    "PrefetchLocalRead": 1,
    "StoreRemapVectorWidth": 0,
    "GlobalSplitU": 1,
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "GlobalReadVectorWidthA": -1,
    "GlobalReadVectorWidthB": -1,
    "VectorWidthA": -1,
    "VectorWidthB": -1,
    "LocalReadVectorWidth": -1,
    "1LDSBuffer": 0,
    "SourceSwap": False,
    "DirectToVgprSparseMetadata": False,
    "WorkGroupMapping": 1,
    "StoreVectorWidth": -1,
    "DirectToVgprA": False,
    "DirectToVgprB": False,
    "WavefrontSize": 32,
    "ExpandPointerSwap": False,
}


def _config(fuse, scale, numWaves):
    problemType = {
        "OperationType": "GEMM",
        "MacDataTypeA": "F8",
        "MacDataTypeB": "F8",
        "DataType": "F8",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "Activation": False,
        "SupportUserArgs": True,
        "UseBias": 0,
        "MXBlockA": 32 if scale == "MXSA" else 0,
        "MXBlockB": 32 if scale == "MXSB" else 0,
        "DataTypeMXSA": "E8",
        "DataTypeMXSB": "E8",
    }
    fork = dict(_FORK, MatrixInstruction=_MATRIX_INSTRUCTION[numWaves], TDMFuse=fuse)
    return {
        "GlobalParameters": {"PrintSolutionRejectionReason": True},
        "BenchmarkProblems": [[problemType, {
            "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
            "ForkParameters": [{key: [value]} for key, value in fork.items()],
        }]],
    }


_SET = re.compile(r"^\s*\.set\s+(\w+)\s*,\s*(\S+)", re.M)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_SGPR_START = re.compile(r"^s\[([^\]:]+)")
# Mnemonic prefixes whose first operand is a source: they write no register operand.
_NO_DESTINATION = ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_setpc", "v_cmpx",
                   "tensor_", "s_store", "buffer_store", "global_store", "flat_store",
                   "scratch_store", "ds_store", "ds_write")


class _Kernel:
    """One emitted kernel: its instructions, label positions and .set table."""

    def __init__(self, source):
        self.symbols = {}
        for name, value in _SET.findall(source):
            self.symbols.setdefault(name, value)
        self.instructions = []
        self.labels = {}
        for line in _BLOCK_COMMENT.sub("", source).splitlines():
            code = line.split("//", 1)[0].strip()
            if not code or code.startswith("."):
                continue
            if code.endswith(":"):
                self.labels.setdefault(code[:-1], len(self.instructions))
                continue
            mnemonic, _, rest = code.partition(" ")
            self.instructions.append(
                (mnemonic, [op.strip() for op in rest.split(",") if op.strip()]))

    def index(self, expr, depth=0):
        """The SGPR index `expr` names (``sgprtdmAGroup1+2``, ``52``), or None."""
        total = 0
        for term in expr.split("+"):
            term = term.strip()
            if term.isdigit():
                total += int(term)
                continue
            if term not in self.symbols or depth > 16:
                return None
            value = self.index(self.symbols[term], depth + 1)
            if value is None:
                return None
            total += value
        return total

    def operandIndex(self, operand):
        match = _SGPR_START.match(operand)
        return self.index(match.group(1)) if match else None

    def setIndex(self, *names):
        """The SGPR index of the first of `names` the kernel defines, or None."""
        for name in names:
            if "sgpr" + name in self.symbols:
                return self.index("sgpr" + name)
        return None

    def reads(self, name):
        """Instructions reading SGPR `name`, as a source or as an operand nothing writes."""
        pattern = re.compile(r"\bsgpr%s\b" % re.escape(name))
        return [(mnemonic, operands) for mnemonic, operands in self.instructions
                if any(pattern.search(op) for op in
                       (operands if mnemonic.startswith(_NO_DESTINATION) else operands[1:]))]

    def tdmWrites(self, first, count):
        """Instructions whose destination is a TDM register starting in first..first+count-1."""
        hits = []
        for mnemonic, operands in self.instructions:
            if not operands or mnemonic.startswith(_NO_DESTINATION) \
                    or "sgprtdm" not in operands[0]:
                continue
            index = self.operandIndex(operands[0])
            if index is not None and first <= index < first + count:
                hits.append((mnemonic, operands))
        return hits

    def tdmIssues(self, begin=0, end=None):
        """The Group0 SGPR index of each tensor_load_to_lds in instructions[begin:end]."""
        return [self.operandIndex(operands[0])
                for mnemonic, operands in self.instructions[begin:end]
                if mnemonic.startswith("tensor_load_to_lds") and operands]

    def mainLoop(self):
        begin = self.labels.get("label_LoopBeginL")
        end = self.labels.get("label_LoopEndL")
        assert begin is not None and end is not None and begin < end, \
            "no label_LoopBeginL ... label_LoopEndL main loop"
        return begin, end


def _carriers(fuse, scale, numWaves, tc):
    """Tensors whose name may program the descriptor set carrying `tc`, its own first."""
    if fuse != 0:
        return [tc, _SCALE_SET_OWNER[fuse, scale]] if tc == scale else [tc]
    if numWaves == 1:
        return [tc]
    return [tc, "MXSA", "MXSB"] if tc == scale else [tc, "A"]


def _liveSets(kernel, fuse, scale, numWaves):
    """{live tensor: Group0 SGPR index of the descriptor set carrying it}."""
    return {tc: kernel.setIndex(*("tdm%sGroup0" % name
                                  for name in _carriers(fuse, scale, numWaves, tc)))
            for tc in ("A", "B", scale)}


def _dead(scale):
    return "MXSB" if scale == "MXSA" else "MXSA"


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    """Derive and emit each cell's kernel once, on first use."""
    root = tmp_path_factory.mktemp("tdm_onesided_mx")
    kernels = {}

    def emit(fuse, scale, numWaves):
        cell = (fuse, scale, numWaves)
        if cell not in kernels:
            path = root / ("tdmfuse%d_%s_w%d.yaml" % cell)
            path.write_text(yaml.safe_dump(_config(*cell)))
            states = derive_states(path, arch=_ARCH, limit_solutions=1)
            assert len(states) == 1, "%s derives no solution" % (cell,)
            assert (states[0]["TDMFuse"], states[0]["NumWaves"]) == (fuse, numWaves), cell
            results = emit_kernels_from_config(path, limit=1, arch=_ARCH)
            assert len(results) == 1, cell
            base, source, err = results[0]
            assert err == 0, (cell, base)
            assert "gfx1250" in source, (cell, base)
            kernels[cell] = _Kernel(source)
        return kernels[cell]

    return emit


@pytest.mark.parametrize("fuse, scale, numWaves", _CELLS, ids=_CELL_IDS)
def test_the_dead_scale_address_is_never_read(emitted, fuse, scale, numWaves):
    kernel = emitted(fuse, scale, numWaves)
    reads = kernel.reads("Address" + _dead(scale))
    assert reads == [], reads[:4]


@pytest.mark.parametrize("fuse, scale, numWaves", _CELLS, ids=_CELL_IDS)
def test_the_live_scale_descriptor_is_programmed(emitted, fuse, scale, numWaves):
    kernel = emitted(fuse, scale, numWaves)
    if fuse == 0:
        group1 = kernel.setIndex(*("tdm%sGroup1" % name
                                   for name in _carriers(fuse, scale, numWaves, scale)))
    else:
        owner = _SCALE_SET_OWNER[fuse, scale]
        group1 = kernel.setIndex("tdm%sGroup1" % owner)
        own = kernel.setIndex("tdm%sGroup1" % scale)
        assert own in (None, group1), "%s does not ride %s's descriptor set" % (scale, owner)
    assert group1 is not None, "no Group1 registers carry %s" % scale
    assert kernel.tdmWrites(group1, _GROUP1_SGPRS), \
        "nothing programs the Group1 registers carrying %s" % scale


@pytest.mark.parametrize("fuse, scale, numWaves", _CELLS, ids=_CELL_IDS)
def test_every_live_descriptor_set_issues_inside_the_main_loop(emitted, fuse, scale, numWaves):
    kernel = emitted(fuse, scale, numWaves)
    sets = _liveSets(kernel, fuse, scale, numWaves)
    assert None not in sets.values(), sets
    # One wave gives each live tensor a descriptor of its own; on more waves A and B share
    # one set (TDMFuse=0) or the scale shares its data tensor's set (TDMFuse 1-3).
    assert len(set(sets.values())) == (3 if numWaves == 1 else 2), sets
    begin, end = kernel.mainLoop()
    issued = set(kernel.tdmIssues(begin, end))
    missing = {tc: index for tc, index in sets.items() if index not in issued}
    assert not missing, (missing, sorted(i for i in issued if i is not None))


@pytest.mark.parametrize("fuse, scale, numWaves", _CELLS, ids=_CELL_IDS)
def test_no_wave_issues_a_descriptor_only_the_dead_scale_owns(emitted, fuse, scale, numWaves):
    kernel = emitted(fuse, scale, numWaves)
    deadSet = kernel.setIndex("tdm%sGroup0" % _dead(scale))
    if deadSet is None or deadSet in _liveSets(kernel, fuse, scale, numWaves).values():
        return
    assert deadSet not in kernel.tdmIssues(), deadSet


@pytest.mark.parametrize("numWaves", [2, 4])
def test_tdmfuse0_loads_a_lone_mxsb_through_the_scale_set(emitted, numWaves):
    """MXSA leads the TDMFuse=0 scale set; with it dead the set still loads MXSB."""
    kernel = emitted(0, "MXSB", numWaves)
    scaleSet = kernel.setIndex("tdmMXSBGroup0", "tdmMXSAGroup0")
    assert scaleSet is not None
    assert scaleSet in kernel.tdmIssues(), scaleSet
