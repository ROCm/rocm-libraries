# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The FlyDSL mono's QSA layout, as the Q3-Q7 stage functions see it.

Everything comes from the ATOM checkout in use (``layout.py`` / ``config.py``
of ``atom.models.qwen4_exp_mono.kernels``, which import no FlyDSL; ``STATE`` /
``PAGE`` as literals of ``stages/qsa.py``). The constants ATOM computes inside
``stages/qsa.py`` and ``stages/qsa_fast.py`` (FlyDSL modules) are recomputed
here from the same inputs with the same formulas; :func:`check_against_atom`
compares them with the ATOM modules where FlyDSL imports (the cluster
container), and the generator records the result.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..mono_layout import _import_atom, _literals

BASE = "atom.models.qwen4_exp_mono.kernels"


@dataclass(frozen=True)
class QsaLayout:
    rows: int
    seq: bool
    grid: int
    threads: int
    max_rows: int
    row_pairs: int
    scratch_bytes: int
    sc: dict = field(default_factory=dict)
    tasks: dict = field(default_factory=dict)  # this build's (``for_rows``)
    lds: dict = field(default_factory=dict)  # word offsets in the mono arena
    stride: dict = field(default_factory=dict)
    lds_words: int = 0
    state: tuple = ()
    page: int = 16
    # model
    q_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    index_dim: int = 128
    index_heads: int = 4
    rotary_dim: int = 64
    index_compress: int = 4
    index_budget: int = 2048
    # layout.py QSA constants
    score_tile: int = 32
    score_tiles: int = 65
    ss_row: int = 2080
    qsa_list: int = 2051
    k_stride: int = 132
    pair: int = 8

    # ---------------------------------------------------------- qsa.py
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

    @property
    def page_blocks(self) -> int:
        return self.page // self.index_compress

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

    @property
    def sel(self) -> int:
        return self.index_budget // self.index_compress  # 512

    @property
    def group(self) -> int:
        return self.q_heads // self.kv_heads  # 12

    @property
    def pp_row(self) -> int:
        return self.ss_row // 2

    @property
    def qn_specs(self) -> int:
        return (self.group + 1) * self.head_words // 2

    def qn_seq(self, rows: int) -> int:
        return (self.group + rows) * self.head_words // 2

    @property
    def selh_from_qn(self) -> int:
        return (self.sc["selh"] - self.sc["qn"]) // self.pair

    @property
    def q7_rounds(self) -> int:
        return -(-self.qsa_list // 128)  # 17

    @property
    def q1_index_tasks(self) -> int:
        return (self.p_q + 31) // 32  # 10

    def state_index(self, name: str) -> int:
        return self.state.index(name)

    # ------------------------------------------------------- qsa_fast.py
    @property
    def max_blocks(self) -> int:
        return 1 << 16

    @property
    def score_shift(self) -> int:
        return self.max_rows * self.scratch_bytes

    @property
    def fast_chunk(self) -> int:
        return 8192

    @property
    def chunks(self) -> int:
        return self.max_blocks // self.fast_chunk  # 8

    @property
    def q4_tasks(self) -> int:
        return self.chunks + 1

    @property
    def fsc(self) -> dict:
        """Byte offsets of Q3 / Q4's mailboxes from the row's second half
        (``qsa_fast._FAST_REGIONS``)."""
        regions = {"fisch": self.grid, "fch": self.chunks, "fcand": self.chunks * self.sel * 2}
        out, off = {}, 0
        for name, pairs in regions.items():
            out[name] = off
            off += (pairs * self.pair + 255) // 256 * 256
        assert off <= self.sc["isc"]
        return out

    @property
    def task_rounds(self) -> dict:
        return {k: -(-v // self.grid) for k, v in self.tasks.items()}


# qsa_fast's Q3 / Q4 constants (pure arithmetic of the values above)
UC = 8
Q4_BATCH = 8
RADIX = ((0, 24), (0, 16), (0, 8), (0, 0), (1, 8), (1, 0))
H_WORDS = 256
COPIES = 8


def q4_lds(sel: int, threads: int) -> dict:
    """Word offsets inside ``q4_keys`` (``qsa_fast``: CNT, CK, CB, H0, ...)."""
    cnt = 3 * H_WORDS
    ck = cnt + 256
    cb = ck + 2 * sel
    h0_stride = H_WORDS + 4
    h0 = cb + sel
    h0_clear = -(-COPIES * h0_stride // threads) * threads
    return {
        "CNT": cnt, "CK": ck, "CB": cb, "H0_STRIDE": h0_stride, "H0": h0,
        "H0_CLEAR": h0_clear, "LDS_WORDS": h0 + h0_clear,
    }  # fmt: skip


def from_atom(rows: int = 1, seq: bool = False, atom_root: str | None = None) -> QsaLayout:
    """The QSA layout of a ``rows``-row build (``seq``: an MTP verify build)."""
    cfg = _import_atom(BASE + ".config", atom_root)
    lay = _import_atom(BASE + ".layout", atom_root)
    C = cfg.DEFAULT
    L = lay.for_rows(rows, seq) if rows > 1 else lay.for_rows(1)
    if atom_root is None:
        import importlib

        try:
            q = importlib.import_module(BASE + ".stages.qsa")
            state, page = tuple(q.STATE), q.PAGE
        except ImportError:  # no FlyDSL here: the literals
            atom_root_guess = os.path.dirname(os.path.dirname(os.path.dirname(cfg.__file__)))
            state, page = _qsa_literals(os.path.dirname(atom_root_guess))
    else:
        state, page = _qsa_literals(atom_root)
    return QsaLayout(
        rows=rows,
        seq=seq,
        grid=cfg.GRID,
        threads=cfg.THREADS,
        max_rows=lay.MAX_ROWS,
        row_pairs=lay.ROW_PAIRS,
        scratch_bytes=lay.SCRATCH_BYTES,
        sc=dict(lay.SC),
        tasks=dict(L.tasks),
        lds=dict(L.lds),
        stride=dict(L.stride),
        lds_words=L.lds_words,
        state=state,
        page=page,
        q_heads=C.q_heads,
        kv_heads=C.kv_heads,
        head_dim=C.head_dim,
        index_dim=C.index_dim,
        index_heads=C.index_heads,
        rotary_dim=C.rotary_dim,
        index_compress=C.index_compress,
        index_budget=C.index_budget,
        score_tile=lay.SCORE_TILE,
        score_tiles=lay.SCORE_TILES,
        ss_row=lay.SS_ROW,
        qsa_list=lay.QSA_LIST,
        k_stride=lay.K_STRIDE,
        pair=lay.PAIR,
    )


def _qsa_literals(atom_root: str):
    path = os.path.join(atom_root, *BASE.split("."), "stages", "qsa.py")
    d = _literals(path, ("STATE", "PAGE"))
    return tuple(d["STATE"]), d["PAGE"]


def check_against_atom(lay: QsaLayout) -> dict:
    """Compare the recomputed constants with ATOM's FlyDSL modules (needs
    FlyDSL). Returns ``{name: (ours, atom)}`` of every mismatch (empty: all
    equal); raises ImportError where FlyDSL is missing."""
    import importlib

    q = importlib.import_module(BASE + ".stages.qsa")
    f = importlib.import_module(BASE + ".stages.qsa_fast")
    pairs = {
        "P_IK": (lay.p_ik, q.P_IK), "P_Q": (lay.p_q, q.P_Q), "P_K": (lay.p_k, q.P_K),
        "P_V": (lay.p_v, q.P_V), "HEAD_WORDS": (lay.head_words, q.HEAD_WORDS),
        "INDEX_WORDS": (lay.index_words, q.INDEX_WORDS), "ROPE_WORDS": (lay.rope_words, q.ROPE_WORDS),
        "PAGE_BLOCKS": (lay.page_blocks, q.PAGE_BLOCKS), "SEL": (lay.sel, q.SEL),
        "GROUP": (lay.group, q.GROUP), "PP_ROW": (lay.pp_row, q.PP_ROW),
        "QN_SPECS": (lay.qn_specs, q.QN_SPECS), "SELH_FROM_QN": (lay.selh_from_qn, q.SELH_FROM_QN),
        "Q7_ROUNDS": (lay.q7_rounds, q.Q7_ROUNDS), "Q1_INDEX_TASKS": (lay.q1_index_tasks, q.Q1_INDEX_TASKS),
        "SCORE_SHIFT": (lay.score_shift, f.SCORE_SHIFT), "FAST_CHUNK": (lay.fast_chunk, f.FAST_CHUNK),
        "CHUNKS": (lay.chunks, f.CHUNKS), "Q4_TASKS": (lay.q4_tasks, f.Q4_TASKS),
        "FSC": (lay.fsc, dict(f.FSC)), "UC": (UC, f.UC), "Q4_BATCH": (Q4_BATCH, f.Q4_BATCH),
        "RADIX": (RADIX, tuple(f.RADIX)), "MAX_BLOCKS": (lay.max_blocks, f.MAX_BLOCKS),
    }  # fmt: skip
    for k in range(2, lay.max_rows + 1):
        pairs[f"QN_SEQ[{k}]"] = (lay.qn_seq(k), q.QN_SEQ[k])
    ql = q4_lds(lay.sel, lay.threads)
    for name in ("CNT", "CK", "CB", "H0_STRIDE", "H0", "H0_CLEAR", "LDS_WORDS"):
        pairs[name] = (ql[name], getattr(f, name))
    return {k: v for k, v in pairs.items() if v[0] != v[1]}
