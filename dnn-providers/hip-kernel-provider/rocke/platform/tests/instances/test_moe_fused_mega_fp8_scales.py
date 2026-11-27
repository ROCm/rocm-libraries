# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The fp8 fused-MoE kernel scales every gate and up output with the
activation scale of its own row.

Stage 1 of ``moe_fused_mega_fp8`` computes, per sorted A row ``m`` and
intermediate column ``n``,

    gate[m, n] = sum_g AScale[m, g] * WGateScale[g, n // 128] * dot_g(m, n)

(``up`` the same with ``WUp``), where ``dot_g`` is the fp8 dot product of A
row ``m`` and weight row ``n`` over the 128-wide hidden group ``g``. A lane of
the 16x16 MFMA loads A from one row but holds four outputs in other rows
(``MfmaAtom.lane_to_output``, the mapping the kernel stores them with), so it
must read the A scale of each output's row, not of the row it loads A from.

Checks:

* On the host, on the kernel's LLVM IR for the spec the dispatcher ships (its
  scheduling hint pinned to the default) for gfx950 (the K=128 atom) and for
  gfx942 (the K=32 atom), from the Python engine and, with the
  ``rocke_engine`` binding, from the C++ engine's own builder (the two texts
  must be the same byte for byte): every gate and up output that Pass A
  stores gets, in each loop iteration ``g``, exactly one A scale, read at
  ``row * stride_a_scale + g``, and one weight scale, read at
  ``g * stride_w + col // 128``, where ``(row, col)`` is the cell the output
  is stored to, for every lane and group of a 3 x 2 grid of workgroups (the
  indices are affine in the workgroup ids, so more than one workgroup along
  each axis shows a wrong tile base); and the per-output factors of a lane
  come from one packed ``<4 x float>`` multiply.
* On a gfx942 or gfx950 GPU, the whole kernel (gate and up, SiLU, the
  dynamic fp8 requantization of the hidden activation, down and the weighted
  reduce into Y), with A scales that differ from row to row within every MFMA
  tile, matches a float reference within the runbook GEMM tolerance. The
  reference requantizes the hidden activation to the e4m3 encoding the target
  converts to (FNUZ on gfx942, OCP on gfx950), scaled to its largest value.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest
from rocke.core.backend import import_engine
from rocke.dispatch.families.moe import MoeRequest, dispatch_moe
from rocke.helpers import compile_kernel
from rocke.helpers.fp8_encoding import fp8_element_format
from rocke.helpers.quant import quant_max_abs
from rocke.instances.common.moe_fused_mega_fp8 import (
    AMAX_FLOOR,
    GROUP_K,
    FusedMegaKernelSpecFp8,
    build_moe_fused_mega_gemm_fp8,
    moe_fused_mega_fp8_grid,
    moe_fused_mega_fp8_signature,
)
from rocke.tests.engines import engine_modules_available, lower_with_engine
from rocke.tests.fp8_gemm_checks import (
    EXACT_OPERAND_VALUES,
    GEMM_TOLERANCE,
    fp8_host_codec,
    mfma_operand_bytes,
)
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers
from rocke.tests.llvm_ir import KernelIR

_ARCHS = ("gfx942", "gfx950")

# The smallest shape the shipped specs tile: one 256-wide intermediate tile
# and one 256-wide hidden output tile, two 128-wide groups along each.
_HIDDEN = 256
_INTERMEDIATE = 256

# The routed problem: each token goes to two of three experts.
_TOKENS = 30
_EXPERTS = 3


def _shipped_spec(arch: str) -> FusedMegaKernelSpecFp8:
    """The fp8 spec the MoE dispatcher selects on ``arch`` for this shape,
    with the loop scheduling hint pinned to the default, ``iglp1``.

    The dispatched spec leaves ``sched_cadence`` unset; then the Python builder
    takes the hint from the ``ROCKE_FP8_SCHED`` environment variable while the
    C++ builder uses ``iglp1``. Both builders read the field, so pinning it
    makes them build the same kernel whatever that variable says."""
    request = MoeRequest(
        num_tokens=_TOKENS,
        hidden=_HIDDEN,
        intermediate=_INTERMEDIATE,
        num_experts=_EXPERTS,
        top_k=2,
        arch=arch,
        dtype="fp8",
    )
    return dataclasses.replace(dispatch_moe(request).spec, sched_cadence="iglp1")


def _lowered_llvm(spec: FusedMegaKernelSpecFp8, arch: str) -> str:
    """The kernel's LLVM IR from every engine this run can drive: the Python
    builder lowered by the Python engine and, with the ``rocke_engine``
    binding, the C++ builder lowered by the C++ engine; the texts must be
    identical."""
    kernel = build_moe_fused_mega_gemm_fp8(spec, arch=arch)
    python_ir = lower_with_engine(kernel, arch, engine="python")
    if engine_modules_available():
        fields = {
            field: value
            for field, value in dataclasses.asdict(spec).items()
            if value is not None
        }
        native_ir = import_engine().moe_fused_mega_fp8_lower_llvm(fields, arch=arch)
        assert (
            native_ir == python_ir
        ), f"the C++ builder emitted different LLVM IR for {spec.name} on {arch}"
    return python_ir


# ---------------------------------------------------------------------------
# Host check on the lowered kernel.
#
# It reads the kernel's LLVM IR and evaluates the address of every scale load
# feeding a stored gate / up output, and of the store itself, for every lane,
# workgroup and loop iteration, so it follows the kernel: any change to how
# the kernel picks a scale changes what it checks.
# ---------------------------------------------------------------------------

_LDS_STORE = re.compile(r"^\s*store float (%[\w.]+), ptr addrspace\(3\) (%[\w.]+)")
# The SiLU(gate) * up chain between the stage 1 accumulators and the store.
_ACTIVATION_OPS = ("fmul", "fadd", "llvm.exp2", "llvm.amdgcn.rcp")


class _KernelIR(KernelIR):
    """The fused-MoE kernel's LLVM IR, with the queries this check makes of
    it."""

    def hidden_stores(self, tile_m: int, tile_n: int) -> list[tuple[str, str, str]]:
        """Pass A's stores of ``silu(gate) * up`` into the ``(tile_m, tile_n)``
        f32 LDS tile: (stored value, row index, column index)."""
        tile_gep = f"getelementptr inbounds [{tile_m} x [{tile_n} x float]]"
        found = []
        for value, address in (
            match.groups() for match in map(_LDS_STORE.match, self.lines) if match
        ):
            if self.definitions[address].startswith(tile_gep):
                row, col = self.operands(address)[-2:]
                found.append((value, row, col))
        return found

    def accumulator_elements(self, value: str) -> list[tuple[str, int]]:
        """The distinct ``(stage 1 loop phi, element)`` pairs a stored
        activation is computed from: the ``extractelement`` reads its
        SiLU * up chain makes (``gate`` is read twice, by the sigmoid and the
        product)."""
        found, stack, seen = set(), [value], set()
        while stack:
            current = stack.pop()
            if current in seen or not current.startswith("%"):
                continue
            seen.add(current)
            op = self.opcode(current)
            if op == "extractelement":
                vector, index = self.operands(current)
                found.add((self.copied(vector), int(index)))
            elif op in _ACTIVATION_OPS:
                stack.extend(self.operands(current))
            else:
                raise AssertionError(f"not modelled: {self.definitions[current]}")
        return sorted(found)


@pytest.mark.parametrize("arch", _ARCHS)
def test_each_stage1_output_is_scaled_with_its_own_row(arch):
    """Every gate and up output is scaled, in every hidden group, with the A
    scale of the row it is stored to and the weight scale of its column.

    In the IR: Pass A stores ``silu(gate[i]) * up[i]`` to the LDS tile at
    ``(row, col)``; ``gate`` and ``up`` are results of the stage 1 group loop,
    each iteration ``g`` of which adds ``group * scale`` with one
    ``llvm.fmuladd``. Element ``i`` of ``scale`` must be the product of one
    read of ``AScale``, at ``(block_m * tile_m + row) * stride_a_scale + g``,
    and one read of the gate (or up) weight scale, at ``g * stride_w +
    (block_n * tile_n + col) // 128``, and ``scale`` is one packed vector
    multiply. The strides get distinct values so a swapped stride shows."""
    spec = _shipped_spec(arch)
    ir = _KernelIR(_lowered_llvm(spec, arch))
    groups = _HIDDEN // GROUP_K
    a_stride = 5
    weight_strides = {"%WGateScale": 7, "%WUpScale": 11}
    block_m = np.arange(3)[None, :, None, None]
    block_n = np.arange(2)[None, None, :, None]
    work_items = {
        ir.call_named("llvm.amdgcn.workitem.id.x"): np.arange(spec.block_size)[
            :, None, None, None
        ],
        ir.call_named("llvm.amdgcn.workgroup.id.y"): block_m,
        ir.call_named("llvm.amdgcn.workgroup.id.x"): block_n,
        "%stride_a_scale": a_stride,
        "%stride_gate_scale": weight_strides["%WGateScale"],
        "%stride_up_scale": weight_strides["%WUpScale"],
    }
    group = np.arange(groups)[None, None, None, :]

    stores = ir.hidden_stores(spec.tile_m, spec.tile_n_inter)
    outputs_per_lane = spec.gate_up_atom().c_per_lane
    assert len(stores) == spec.mfmas_m * spec.mfmas_n * outputs_per_lane, stores
    for value, row_index, col_index in stores:
        row = block_m * spec.tile_m + ir.evaluate(row_index, work_items)
        col = block_n * spec.tile_n_inter + ir.evaluate(col_index, work_items)
        accumulators = ir.accumulator_elements(value)
        assert len(accumulators) == 2, accumulators  # gate and up
        weight_params = set()
        for phi, element in accumulators:
            _, carried = ir.incoming(phi)
            assert ir.opcode(carried) == "llvm.fmuladd", ir.definitions[carried]
            _, scale, addend = ir.operands(carried)
            assert ir.copied(addend) == phi, ir.definitions[carried]
            # One packed multiply makes a lane's per-output dequant factors.
            assert ir.definitions[ir.copied(scale)].startswith(
                f"fmul <{outputs_per_lane} x float>"
            ), ir.definitions[ir.copied(scale)]
            (counter,) = ir.header_i32_phis(phi)
            inputs = {**work_items, counter: group}
            reads = {}
            for load in ir.loads_feeding(scale, element):
                param, index = ir.read_location(load, inputs)
                reads.setdefault(param, []).append(index)
            (weight_param,) = set(reads) - {"%AScale"}
            weight_params.add(weight_param)
            (a_index,) = reads["%AScale"]
            (w_index,) = reads[weight_param]
            shape = np.broadcast(a_index, w_index, row, col, group).shape
            for got, want, what in (
                (a_index, row * a_stride + group, "A scale"),
                (
                    w_index,
                    group * weight_strides[weight_param] + col // GROUP_K,
                    f"{weight_param} scale",
                ),
            ):
                np.testing.assert_array_equal(
                    np.broadcast_to(got, shape),
                    np.broadcast_to(want, shape),
                    err_msg=f"element {element} of {phi}: {what}",
                )
        assert weight_params == {"%WGateScale", "%WUpScale"}, weight_params


# ---------------------------------------------------------------------------
# GPU check.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpu_arch():
    """The current GPU's arch if this check is enabled for it; skip
    otherwise, or fail when ``ROCKE_REQUIRE_<ARCH>=1`` names an arch the GPU
    is not."""
    return current_device_arch_in(_ARCHS, subject="the fp8 fused-MoE kernel")


@dataclasses.dataclass(frozen=True)
class _MoeCase:
    """One routed problem in the sorted, tile-padded layout the kernel reads.

    ``a`` is ``(blocks * tile_m, hidden)``, zero in padding rows, and
    ``a_scale`` ``(blocks * tile_m, hidden / 128)``; the weights are
    ``(experts, out, contraction)`` and their scales ``(experts,
    contraction / 128, out / 128)``, indexed ``[e, g, n // 128]`` as the
    kernel reads them. ``token`` is -1 in padding rows."""

    a: np.ndarray
    a_scale: np.ndarray
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray
    gate_scale: np.ndarray
    up_scale: np.ndarray
    down_scale: np.ndarray
    token: np.ndarray
    weight: np.ndarray
    block_expert: np.ndarray
    tokens: int


def _routed_case(tile_m: int, *, tokens: int, experts: int, seed: int) -> _MoeCase:
    """Route token ``t`` to experts ``t % experts`` and ``(t + 1) % experts``
    and pad each expert's rows to whole ``tile_m`` blocks, so some blocks are
    full and some partial. Every A row has its own scale in every group, so
    the rows of one MFMA tile all differ."""
    rng = np.random.default_rng(seed)
    hidden_groups = _HIDDEN // GROUP_K
    inter_groups = _INTERMEDIATE // GROUP_K
    rows_of = {expert: [] for expert in range(experts)}
    for token_id in range(tokens):
        rows_of[token_id % experts].append((token_id, 0.75))
        rows_of[(token_id + 1) % experts].append((token_id, 0.25))
    token, weight, block_expert = [], [], []
    for expert, rows in rows_of.items():
        blocks = -(-len(rows) // tile_m)
        padded = rows + [(-1, 0.0)] * (blocks * tile_m - len(rows))
        token += [token_id for token_id, _ in padded]
        weight += [route_weight for _, route_weight in padded]
        block_expert += [expert] * blocks
    token = np.array(token, dtype=np.int32)
    a = rng.choice(EXACT_OPERAND_VALUES, size=(token.size, _HIDDEN))
    a[token < 0] = 0.0

    def weights(out: int, contraction: int) -> np.ndarray:
        return rng.choice(EXACT_OPERAND_VALUES, size=(experts, out, contraction))

    def weight_scales(groups: int, out_groups: int) -> np.ndarray:
        return rng.uniform(1 / 32, 1 / 8, size=(experts, groups, out_groups))

    return _MoeCase(
        a=a,
        a_scale=rng.uniform(0.25, 4.0, size=(token.size, hidden_groups)),
        w_gate=weights(_INTERMEDIATE, _HIDDEN),
        w_up=weights(_INTERMEDIATE, _HIDDEN),
        w_down=weights(_HIDDEN, _INTERMEDIATE),
        gate_scale=weight_scales(hidden_groups, inter_groups),
        up_scale=weight_scales(hidden_groups, inter_groups),
        down_scale=weight_scales(inter_groups, hidden_groups),
        token=token,
        weight=np.array(weight),
        block_expert=np.array(block_expert, dtype=np.int32),
        tokens=tokens,
    )


def _grouped_matmul(
    operand: np.ndarray,
    weights: np.ndarray,
    operand_scale: np.ndarray,
    weight_scale: np.ndarray,
) -> np.ndarray:
    """``sum_g (operand_g @ weights_g.T) * operand_scale[:, g] *
    weight_scale[g, n // 128]`` over the 128-wide contraction groups ``g``, in
    float64."""
    out = np.zeros((operand.shape[0], weights.shape[0]))
    out_group = np.arange(weights.shape[0]) // GROUP_K
    for group in range(operand.shape[1] // GROUP_K):
        k_slice = slice(group * GROUP_K, (group + 1) * GROUP_K)
        out += (
            (operand[:, k_slice] @ weights[:, k_slice].T)
            * operand_scale[:, group, None]
            * weight_scale[group, out_group]
        )
    return out


def _reference(case: _MoeCase, tile_m: int, arch: str) -> np.ndarray:
    """``Y`` as the kernel defines it on ``arch``, per ``tile_m``-row block of
    one expert: gate and up with each row's own A scale; ``h = silu(gate) *
    up``; ``h`` requantized to the e4m3 encoding ``arch`` converts to, with one
    scale per 128-wide intermediate group, ``amax / largest value`` over the
    block's rows (amax floored at ``AMAX_FLOOR``), in f32 as the kernel does;
    down with that scale and the down weight scale; then ``Y[token] += weight
    * out`` for the block's real rows."""
    codec = fp8_host_codec(fp8_element_format("fp8e4m3", arch))
    fp8_values, fp8_round = codec.decode_table(), codec.round_to_codes
    fp8_max = np.float32(quant_max_abs("fp8e4m3", arch=arch))
    y = np.zeros((case.tokens, _HIDDEN))
    for block, expert in enumerate(case.block_expert):
        rows = slice(block * tile_m, (block + 1) * tile_m)
        a, a_scale = case.a[rows], case.a_scale[rows]
        gate = _grouped_matmul(a, case.w_gate[expert], a_scale, case.gate_scale[expert])
        up = _grouped_matmul(a, case.w_up[expert], a_scale, case.up_scale[expert])
        with np.errstate(over="ignore"):
            hidden = (gate / (1.0 + np.exp(-gate)) * up).astype(np.float32)
        hidden_groups = hidden.reshape(tile_m, -1, GROUP_K)
        amax = np.maximum(
            np.abs(hidden_groups).max(axis=(0, 2)), np.float32(AMAX_FLOOR)
        )
        h_scale = amax * (np.float32(1.0) / fp8_max)
        codes = fp8_round(hidden_groups * (np.float32(1.0) / h_scale)[:, None])
        h_q = fp8_values[codes].reshape(tile_m, -1).astype(np.float64)
        h_scale_rows = np.broadcast_to(
            h_scale.astype(np.float64), (tile_m, h_scale.size)
        )
        out = _grouped_matmul(
            h_q, case.w_down[expert], h_scale_rows, case.down_scale[expert]
        )
        for token_id, route_weight, row_out in zip(
            case.token[rows], case.weight[rows], out
        ):
            if token_id >= 0:
                y[token_id] += route_weight * row_out
    return y


def _run_on_gpu(arch: str, spec: FusedMegaKernelSpecFp8, case: _MoeCase) -> np.ndarray:
    """Run the kernel :func:`_lowered_llvm` checks on the GPU; return ``Y``."""
    artifact = compile_kernel(build_moe_fused_mega_gemm_fp8(spec, arch=arch), arch=arch)
    assert artifact.llvm_text == _lowered_llvm(spec, arch)
    hidden_groups = _HIDDEN // GROUP_K
    inter_groups = _INTERMEDIATE // GROUP_K
    host = {
        "A": mfma_operand_bytes(arch, "fp8e4m3", case.a),
        "WGate": mfma_operand_bytes(arch, "fp8e4m3", case.w_gate),
        "WUp": mfma_operand_bytes(arch, "fp8e4m3", case.w_up),
        "WDown": mfma_operand_bytes(arch, "fp8e4m3", case.w_down),
        "AScale": case.a_scale.astype(np.float32),
        "WGateScale": case.gate_scale.astype(np.float32),
        "WUpScale": case.up_scale.astype(np.float32),
        "WDownScale": case.down_scale.astype(np.float32),
        "SortedTokenIds": case.token,
        "SortedWeights": case.weight.astype(np.float32),
        "BlockExpertIds": case.block_expert,
        "Y": np.zeros((case.tokens, _HIDDEN), dtype=np.float32),
    }
    host = {name: np.ascontiguousarray(array) for name, array in host.items()}
    launch_with_buffers(
        artifact,
        moe_fused_mega_fp8_signature(spec),
        host,
        grid=moe_fused_mega_fp8_grid(case.block_expert.size, _INTERMEDIATE, spec),
        block=(spec.block_size, 1, 1),
        scalars={
            "M": case.token.size,
            "N": _INTERMEDIATE,
            "K": _HIDDEN,
            "H_out": _HIDDEN,
            "stride_a": _HIDDEN,
            "stride_b_gate": _INTERMEDIATE * _HIDDEN,
            "stride_b_up": _INTERMEDIATE * _HIDDEN,
            "stride_b_down": _HIDDEN * _INTERMEDIATE,
            "stride_a_scale": hidden_groups,
            "stride_gate_scale": inter_groups,
            "stride_up_scale": inter_groups,
            "stride_down_scale": hidden_groups,
            "stride_gate_scale_e": hidden_groups * inter_groups,
            "stride_up_scale_e": hidden_groups * inter_groups,
            "stride_down_scale_e": inter_groups * hidden_groups,
            "slot_size": spec.tile_m,
            "tokens": case.tokens,
        },
    )
    return host["Y"]


@pytest.mark.gpu
def test_row_scales_on_gpu(gpu_arch):
    """With an A scale per row that differs within every MFMA tile, the
    kernel's ``Y`` matches the reference. A kernel that scales a lane's four
    outputs with the scale of the row the lane loads A from gets three of
    every four gate and up outputs wrong, by up to a factor of 16."""
    spec = _shipped_spec(gpu_arch)
    case = _routed_case(spec.tile_m, tokens=_TOKENS, experts=_EXPERTS, seed=3128)
    got = _run_on_gpu(gpu_arch, spec, case)
    want = _reference(case, spec.tile_m, gpu_arch)
    assert np.isfinite(got).all()
    error = np.abs(got - want).max()
    assert error <= GEMM_TOLERANCE * np.abs(want).max(), error
