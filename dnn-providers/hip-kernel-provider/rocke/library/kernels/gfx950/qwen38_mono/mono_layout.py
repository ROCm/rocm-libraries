# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ATOM mono layout constants the rocKE stage functions are built against
(the roadmap's ``atom_compat`` mailbox layout).

A stage function linked into the FlyDSL mono kernel reads and writes the
mono's own mailboxes, state tables and caches, so it is specialized on the
mono's layout: scratch region offsets (``layout.SC``), the request-row
stride (``layout.ROW_PAIRS``), the stage task counts, the QSA state-table
order and the model geometry. :func:`from_atom` reads them, read-only, from
the ATOM checkout in use (``atom.models.qwen4_exp_mono.kernels.layout`` /
``config`` and ``stages.qsa``); nothing here is a second copy of them, apart
from the fixture tests build with explicit values.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field

__all__ = ["MonoLayout", "from_atom", "LAYER_ARGS"]

# ATOM ``kernels/kernel.py`` LAYER_ARGS: the mono layer kernel's arguments
# after ``step`` and ``layer``. The stage-function ABI passes all of them;
# ``from_atom`` checks this copy against ATOM's.
LAYER_ARGS = (
    "scratch", "ple", "h_in", "hc_norm_w", "w_down", "s_down", "w_up", "s_up", "w_inj",
    "w_qkvz", "s_qkvz", "conv_w", "conv_in", "conv_out", "ba_w", "a_log", "dt_bias",
    "norm_w", "rec_in", "rec_out", "w_qsa", "s_qsa", "qsa", "w_out", "s_out", "m_norm_w",
    "m_w_down", "m_s_down", "m_w_up", "m_s_up", "m_w_inj", "r_w", "e_w_gu", "e_s_gu",
    "sh_w_gu", "sh_s_gu", "e_w_dn", "e_s_dn", "sh_w_dn", "sh_s_dn", "h_out",
)  # fmt: skip


@dataclass(frozen=True)
class MonoLayout:
    """The constants a stage function needs (byte offsets, pair indices)."""

    grid: int
    threads: int
    max_rows: int
    row_pairs: int  # pairs between request rows of a mailbox region
    sc: dict = field(default_factory=dict)  # region -> byte offset in scratch
    tasks: dict = field(default_factory=dict)  # stage -> tasks (rows-1 layout)
    # model geometry
    q_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    index_dim: int = 128
    index_heads: int = 4
    rotary_dim: int = 64
    index_compress: int = 4
    # QSA
    qsa_state: tuple = ()  # the ``qsa`` table's entry order
    page: int = 16  # tokens per cache page
    eps_bits: int = 0x358637BD  # fl(1e-6)

    @property
    def waves(self) -> int:
        return self.threads // 64

    @property
    def head_words(self) -> int:
        return self.head_dim // 2

    @property
    def index_words(self) -> int:
        return self.index_dim // 2

    @property
    def rope_words(self) -> int:
        return self.rotary_dim // 2

    # pair offsets of the fused QSA projection (``qp``), ATOM ``stages/qsa.py``
    @property
    def p_ik(self) -> int:
        return self.index_heads * self.index_dim // 2

    @property
    def p_q(self) -> int:
        return self.p_ik + self.index_dim // 2

    @property
    def p_k(self) -> int:
        return self.p_q + self.q_heads * self.head_dim

    @property
    def p_v(self) -> int:
        return self.p_k + self.kv_heads * self.head_dim // 2

    def state_index(self, name: str) -> int:
        return self.qsa_state.index(name)


def _import_atom(name: str, atom_root: str | None):
    if atom_root is None:
        return importlib.import_module(name)
    # Load the module file by path under stub packages: no ``atom/__init__``
    # (which imports the serving engine) and no FlyDSL needed for layout/config.
    for i in range(1, name.count(".") + 1):
        pkg = ".".join(name.split(".")[:i])
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [os.path.join(atom_root, *pkg.split("."))]
            sys.modules[pkg] = m
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(atom_root, *name.split(".")) + ".py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _literals(path: str, names) -> dict:
    """Module-level ``NAME = <literal>`` assignments of a source file."""
    import ast

    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in names:
                out[t.id] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    if missing:
        raise RuntimeError(f"{path}: no literal {sorted(missing)}")
    return out


def from_atom(atom_root: str | None = None) -> MonoLayout:
    """The layout of the ATOM checkout in use (``import atom``), or of the
    checkout at ``atom_root`` loaded by file path (layout/config only)."""
    base = "atom.models.qwen4_exp_mono.kernels"
    cfg = _import_atom(base + ".config", atom_root)
    lay = _import_atom(base + ".layout", atom_root)
    C = cfg.DEFAULT
    if atom_root is None:
        kern = importlib.import_module(base + ".kernel")
        qsa = importlib.import_module(base + ".stages.qsa")
        if tuple(kern.LAYER_ARGS) != LAYER_ARGS:
            raise RuntimeError("ATOM LAYER_ARGS differ from the stage-function ABI")
        state, page = tuple(qsa.STATE), qsa.PAGE
    else:
        # ``stages/qsa.py`` imports FlyDSL: read its STATE / PAGE literals.
        consts = _literals(
            os.path.join(atom_root, *base.split("."), "stages", "qsa.py"),
            ("STATE", "PAGE"),
        )
        state, page = tuple(consts["STATE"]), consts["PAGE"]
        kargs = _literals(
            os.path.join(atom_root, *base.split("."), "kernel.py"), ("LAYER_ARGS",)
        )["LAYER_ARGS"]
        if tuple(kargs) != LAYER_ARGS:
            raise RuntimeError("ATOM LAYER_ARGS differ from the stage-function ABI")
    return MonoLayout(
        grid=cfg.GRID,
        threads=cfg.THREADS,
        max_rows=lay.MAX_ROWS,
        row_pairs=lay.ROW_PAIRS,
        sc=dict(lay.SC),
        tasks=dict(lay.TASKS),
        q_heads=C.q_heads,
        kv_heads=C.kv_heads,
        head_dim=C.head_dim,
        index_dim=C.index_dim,
        index_heads=C.index_heads,
        rotary_dim=C.rotary_dim,
        index_compress=C.index_compress,
        qsa_state=state,
        page=page,
    )
