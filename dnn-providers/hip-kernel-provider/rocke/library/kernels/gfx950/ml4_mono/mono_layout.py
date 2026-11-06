# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ML4 mono layout loaded from the aiter source tree.

Unlike the Qwen family, ML4 is hosted in aiter and has no ATOM package to
import.  This module therefore keeps the source path and the formulas used by
aiter's ``layout.py``/``kernel.py`` together.  The formulas are intentionally
small and explicit: they make the ABI sidecar reproducible on a workstation
without importing torch or compiling FlyDSL.
"""

from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

GRID = 256
THREADS = 512
WAVES = THREADS // 64
PAIR = 8
HIDDEN = 6144
KV_LORA = 512
PE_DIM = 64
# ``variant.py`` defaults to the rope layer.  The no-PE layer loader can opt
# into 128 NoPE columns; keeping the default here matches aiter's default
# source import.
NOPE_DIM = 64
V_DIM = 128
INDEX_DIM = 128
INDEX_HEADS = 32
N_EXPERTS = 128
TOP_K = 4
MOE_SLOTS = TOP_K + 1
INTER = 1536  # TP4 ML4 shard; TP8 callers may pass 768 explicitly.
QKV_A_ROWS = KV_LORA + PE_DIM
QKV_A_TILE = 16
ROUTER_TILE = 8
ROW_TILE = 32
UG_TILE = 16
UK_TILE = 128
UV_TILE = 64
SPLIT_KEYS = 64
INDEX_KEYS_PER_TASK = 64
INDEX_TILE = 16
XQ_BLOCKS = HIDDEN // 128
XQ_GROUPS = HIDDEN // 32

# The launch ABI is the order in build_glm5_monokernel and launch().  Keep
# step/rank/layer in this list: a linked stage sees the same kernarg contract
# as the FlyDSL kernel, even when a particular body does not use all fields.
KERNEL_PARAMS = (
    ("h_in", "i64"),
    ("x_out", "i64"),
    ("cur_pos", "i64"),
    ("positions", "i64"),
    ("slot_mapping", "i64"),
    ("sparse_kv_indptr", "i64"),
    ("kv_cache", "i64"),
    ("pe_cache", "i64"),
    ("indices", "i64"),
    ("rope_cos", "i64"),
    ("rope_sin", "i64"),
    ("g_in", "i64"),
    ("g_q", "i64"),
    ("g_kv", "i64"),
    ("g_post", "i64"),
    ("w_qkv_a", "i64"),
    ("s_qkv_a", "i64"),
    ("w_q_b", "i64"),
    ("s_q_b", "i64"),
    ("w_uk", "i64"),
    ("s_uk", "i64"),
    ("w_uv", "i64"),
    ("s_uv", "i64"),
    ("w_o", "i64"),
    ("s_o", "i64"),
    ("w_r", "i64"),
    ("bias", "i64"),
    ("w_ug", "i64"),
    ("s_ug", "i64"),
    ("w_dn", "i64"),
    ("s_dn", "i64"),
    ("scratch", "i64"),
    ("sym", "i64"),
    ("peers", "i64"),
    ("timeline_buf", "i64"),
    ("step", "i64"),
    ("rank", "i32"),
    ("layer", "i32"),
)
LAYER_ARGS = tuple(name for name, _ in KERNEL_PARAMS)


def _align(size: int, alignment: int = 256) -> int:
    return (size + alignment - 1) // alignment * alignment


def _source_constants(path: Path, names: set[str]) -> dict[str, object]:
    """Read literal top-level assignments for provenance checks."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


def _find_layout_dir(root: str | os.PathLike[str] | None) -> Path:
    root = root or os.environ.get("AITER_ROOT") or os.environ.get("AITER_ML4_ROOT")
    if not root:
        raise FileNotFoundError(
            "aiter ML4 layout is unavailable; pass --aiter-root or set AITER_ROOT"
        )
    p = Path(root)
    candidates = [
        p,
        p / "aiter" / "ops" / "flydsl" / "kernels" / "mono" / "mistral_large4" / "gfx950",
        p / "ops" / "flydsl" / "kernels" / "mono" / "mistral_large4" / "gfx950",
    ]
    for candidate in candidates:
        if (candidate / "layout.py").is_file() and (candidate / "kernel.py").is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find aiter/ops/flydsl/kernels/mono/mistral_large4/gfx950 "
        f"under {p}"
    )


def _mxfp4_scale_index(row: int, col: int, cols: int) -> tuple[int, int]:
    """Return the source-layout scale word and byte lane for documentation."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, e = col // 8, col % 4
    return (((r32 * ((cols + 7) // 8) + c8) * 4 + e) * 16 + b, a)


@dataclass(frozen=True)
class MonoLayout:
    """A source-backed ML4 layout and schedule view."""

    source_dir: Path
    source_digest: str
    grid: int = GRID
    threads: int = THREADS
    max_rows: int = 12
    hidden: int = HIDDEN
    inter: int = INTER
    experts: int = N_EXPERTS
    top_k: int = TOP_K
    nope_layer: bool = False
    q_heads: int = 16
    kv_heads: int = 2
    head_dim: int = 128
    scratch_by_s: dict[int, dict[str, int]] = field(default_factory=dict)
    symmetric_by_s: dict[int, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.grid != 256 or self.threads != 512:
            raise ValueError("ML4 rocKE stages require the 256 CTA x 512 thread mono")

    @property
    def waves(self) -> int:
        return self.threads // 64

    @property
    def moe_slots(self) -> int:
        return self.top_k + 1

    def tag(self, step: int, layer: int) -> int:
        """The ML4 mono epoch tag (``step * 128 + layer + 1``)."""
        return step * 128 + layer + 1

    def peer_slot(self, rank: int, npes: int) -> int:
        """Rank-local peer slot used by ML4's symmetric peer reductions."""
        if npes <= 0 or not 0 <= rank < npes:
            raise ValueError(f"rank must be in [0, {npes}), got {rank}")
        return rank

    def layout(
        self,
        samples: int,
        heads: int | None = None,
        npes: int = 4,
        sparse_attention_topk: int = 2048,
        *,
        with_indexer: bool = False,
        index_max_seq: int = 4096,
        indexer_cp: bool = False,
        output_heads: int | None = None,
        dcp_size: int = 1,
        native_fp4_mfma: bool = True,
        inter: int | None = None,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """The aiter ``layout()`` result: scratch and symmetric offsets."""
        if not 1 <= samples <= self.max_rows:
            raise ValueError(f"samples must be in 1..{self.max_rows}, got {samples}")
        heads = self.q_heads if heads is None else heads
        inter = self.inter if inter is None else inter
        output_heads = heads if output_heads is None else output_heads
        split_count = sparse_attention_topk // self._sparse_keys(samples, heads)
        nope_dim = 128 if self.nope_layer else 64
        xq_groups = self.hidden // (32 if native_fp4_mfma else 128)
        pair_bytes = 8
        items = [
            ("q_a", 0),
            ("q_an", samples * self.hidden // 2 * pair_bytes),
            ("iq_an", samples * self.hidden // 2 * pair_bytes),
            ("kv_a", samples * (KV_LORA + PE_DIM) * pair_bytes),
            ("kvnew", samples * KV_LORA * pair_bytes),
            ("penew", samples * PE_DIM * pair_bytes),
            ("q_nope", samples * heads * nope_dim * pair_bytes),
            ("q_pe", samples * heads * PE_DIM * pair_bytes),
            ("q_lat", samples * heads * KV_LORA * pair_bytes),
            ("sp_acc", samples * split_count * heads * KV_LORA * pair_bytes),
            ("sp_m", samples * split_count * heads * pair_bytes),
            ("sp_l", samples * split_count * heads * pair_bytes),
            ("o", samples * output_heads * V_DIM * pair_bytes),
            ("a", samples * self.hidden * pair_bytes),
            ("scores", samples * self.experts * pair_bytes),
            ("xq", samples * self.hidden // 4 * pair_bytes),
            ("xqs", samples * xq_groups * pair_bytes),
            ("sel", samples * self.moe_slots * pair_bytes),
            ("prob", samples * self.moe_slots * pair_bytes),
            ("mid", samples * self.moe_slots * inter * pair_bytes),
            ("ugp", GRID * samples * 2 * UG_TILE * pair_bytes),
            ("xqd", samples * self.hidden * 4),
        ]
        if with_indexer:
            items += [
                ("index_k", samples * INDEX_DIM * pair_bytes),
                ("index_k_new", samples * INDEX_DIM // 2 * pair_bytes),
                ("index_ready", samples * pair_bytes),
                ("index_w", samples * INDEX_HEADS * pair_bytes),
                ("index_q", samples * INDEX_HEADS * INDEX_DIM // 2 * pair_bytes),
                ("index_scores", samples * index_max_seq * pair_bytes),
                ("indices", samples * sparse_attention_topk * 4),
                ("indices_ready", samples * pair_bytes),
            ]
        offset, scratch = 0, {}
        for name, size in items:
            scratch[name] = offset
            offset += _align(size)
        scratch["_bytes"] = offset

        part = npes * samples * self.hidden * pair_bytes
        region = 2 * part
        symmetric = {
            "attn": 0,
            "ffn": region,
            "_part_stride": part,
            "_bytes": 2 * region,
        }
        return scratch, symmetric

    def scratch(self, samples: int, **kwargs) -> dict[str, int]:
        return self.layout(samples, **kwargs)[0]

    def stage_tasks(
        self,
        samples: int,
        heads: int | None = None,
        sparse_attention_topk: int = 2048,
        *,
        expert_mxfp4: bool = True,
        inter: int | None = None,
        npes: int = 4,
        indexer_cp: bool = False,
        split_head_merge: bool = True,
    ) -> tuple[tuple[str, int], ...]:
        """The aiter stage task table, including the routed MoE suffix."""
        heads = self.q_heads if heads is None else heads
        inter = self.inter if inter is None else inter
        split_count = sparse_attention_topk // self._sparse_keys(samples, heads)
        tasks = [
            ("qkv_a", QKV_A_ROWS // QKV_A_TILE),
            ("q_norm", samples),
            ("cache", 1),
            (
                "q_b",
                heads * ((128 if self.nope_layer else 64) + PE_DIM) // 16,
            ),
            ("uk", heads * KV_LORA // UK_TILE),
            ("split", samples * (1 if split_head_merge else heads // WAVES) * split_count),
            ("uv", samples * (heads * V_DIM // UV_TILE)),
            ("o", self.hidden // ROW_TILE),
            ("router", samples * (self.experts // ROUTER_TILE)),
            # FlyDSL's UG8 task owns eight intermediate rows, not one
            # UG_TILE (16-row) tile.  This is deliberately independent of
            # UG_TILE, which describes the LDS/output tile elsewhere.
            ("ug", self.top_k * (inter // 8)),
            ("down", self.hidden // self._dn_tile(samples)),
        ]
        return tuple(tasks)

    def stage_bases(self, samples: int, **kwargs) -> dict[str, int]:
        bases, offset = {}, 0
        for name, count in self.stage_tasks(samples, **kwargs):
            bases[name] = offset % self.grid
            offset += count
        return bases

    def stage_table(self, samples: int, **kwargs) -> dict[str, tuple[int, int]]:
        return {
            name: (count, self.stage_bases(samples, **kwargs)[name])
            for name, count in self.stage_tasks(samples, **kwargs)
        }

    def arena_words(
        self,
        samples: int,
        *,
        heads: int | None = None,
        sparse_attention_topk: int = 2048,
        inter: int | None = None,
        native_fp4_mfma: bool = True,
        moe_dedup: bool = False,
        qkv_sample_tile: int = 6,
        with_indexer: bool = False,
        index_max_seq: int = 4096,
    ) -> int:
        """Reproduce ``WORK_WORDS`` from ML4's ``build_glm5_monokernel``."""
        heads = self.q_heads if heads is None else heads
        inter = self.inter if inter is None else inter
        split_keys = self._sparse_keys(samples, heads)
        qk_dim = KV_LORA + PE_DIM
        qs, ks, ps = qk_dim // 2 + 4, KV_LORA // 2 + 4, PE_DIM // 2 + 4
        split_x_words = heads * qs + split_keys * ks + split_keys * ps
        sample_tile = min(samples, qkv_sample_tile)
        xq_stride = self.hidden // 4 + 4
        xqs_stride = self.hidden // (32 if native_fp4_mfma else 128) + 1
        n_split = sparse_attention_topk // split_keys
        index_words = min(index_max_seq, 16384) if with_indexer else 0
        down_x_words = samples * self.moe_slots * inter // (
            4 if native_fp4_mfma else 2
        )
        x_words = max(
            sample_tile * self.hidden // 2,
            samples * self.hidden // 2,
            samples * xq_stride,
            split_x_words,
            index_words,
            down_x_words,
        )
        misc_words = max(
            8 + samples * xqs_stride,
            samples * self.moe_slots * (inter // (32 if native_fp4_mfma else 128)),
            n_split + 2,
        )
        lds_keys = max(264 if with_indexer else 0, split_keys, samples * self.moe_slots)
        red_words = WAVES * 64 * 4
        out_words = max(samples * ROW_TILE, samples * 2 * UG_TILE)
        group_words = (
            samples * self.top_k * (2 + 2 * ((samples * 4 + 31) // 32)) + 1
            if moe_dedup and samples > 1
            else 0
        )
        return x_words + misc_words + lds_keys + red_words + out_words + group_words

    def _sparse_keys(self, samples: int, heads: int) -> int:
        return 32 if heads > 2 * WAVES or (heads == WAVES and samples > 4) else 64

    def _dn_tile(self, samples: int) -> int:
        return 32 if samples == 1 else self.hidden // GRID


def from_aiter(aiter_root: str | os.PathLike[str] | None = None) -> MonoLayout:
    """Load and validate the ML4 layout sources by path.

    The loader does not import ``aiter`` (and therefore does not require torch).
    It reads the two source files for provenance and uses the formulas mirrored
    above for the static layout values.
    """
    source_dir = _find_layout_dir(aiter_root)
    layout_py = source_dir / "layout.py"
    kernel_py = source_dir / "kernel.py"
    text = layout_py.read_bytes() + kernel_py.read_bytes()
    digest = hashlib.sha256(text).hexdigest()
    layout_text = layout_py.read_text(encoding="utf-8")
    kernel_text = kernel_py.read_text(encoding="utf-8")
    if "def layout(" not in layout_text or "def stage_tasks(" not in layout_text:
        raise ValueError(f"{layout_py}: missing layout()/stage_tasks()")
    if "def build_glm5_monokernel(" not in kernel_text:
        raise ValueError(f"{kernel_py}: not the ML4 mono kernel source")
    # Literal checks catch accidentally pointing at a similarly named kernel.
    constants = _source_constants(layout_py, {"BLOCKS", "THREADS", "ROUTER_TILE", "UG_TILE"})
    if constants.get("BLOCKS", GRID) != GRID or constants.get("THREADS", THREADS) != THREADS:
        raise ValueError(f"{layout_py}: unexpected ML4 grid geometry")
    return MonoLayout(source_dir=source_dir, source_digest=digest)


def from_path(path: str | os.PathLike[str]) -> MonoLayout:
    """Alias used by callers that describe the source as a file path."""
    return from_aiter(path)


__all__ = [
    "GRID",
    "THREADS",
    "WAVES",
    "KERNEL_PARAMS",
    "LAYER_ARGS",
    "MonoLayout",
    "from_aiter",
    "from_path",
]
