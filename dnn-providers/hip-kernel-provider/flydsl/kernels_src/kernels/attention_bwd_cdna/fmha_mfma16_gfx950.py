# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""The 16-rows-per-wave MFMA family: lane maps, addressing, operand shapes.

**Published B3.5 probe result.** Shared by the dK/dV and dQ backward kernels so
the map is derived once; `tooling/probe_tr16_lanemap_gfx950.py` is what
measured everything here, and it is the thing to re-run if any of it is
doubted.

--- The question this answers ----------------------------------------------

B3.5 needs 16 rows per wave, which needs a 16-row MFMA A operand out of an LDS
tile staged `[token][d]`. The addendum's warning was that AITER's kernels using
`v_mfma_f32_16x16x16` *exclusively* issue **zero** `ds_read_b64_tr_b16`, and the
ones with hundreds of transpose reads use the wide-K shapes -- so the transpose
might simply not serve a 16-row operand, which would force both orientations to
be staged at exactly the widths where LDS is tightest.

**It does serve it, for both `16x16x16` and `16x16x32`, from one staged tile.**
Measured end to end against a host reference at head_dim 64, 96, 128, 192 and
256, at both staging granules, 0 wrong out of 256 accumulator elements every
time. AITER's correlation is with its own layout choice, not with the hardware.

--- What the instruction actually does -------------------------------------

Measured with every lane given a distinct address and LDS holding a pattern
that names its own `(token, d)`, so the dump identifies its own source:

    O[j][i] = M[16*(j//16) + 4*i + ((j % 16)//4)][j % 4]

where `M[p][q]` is the `q`-th of the four 16-bit elements lane `p` addressed.
Equivalently, with `p = 4a + b` inside a 16-lane group and element `q`, the
result lands at lane `4b + q`, element `a`. Two facts fall out, and both were
checked rather than assumed:

- **Every lane's own address is honoured** -- the hardware does not derive a
  block from one lane's address.
- **No traffic crosses a 16-lane group.** 0 of 256 output elements did.

So one read produces, per 16-lane group: **16 output lanes, each holding one
value of whatever the address varies with `lane % 4` and the element index, and
four values of whatever it varies with `(lane % 16) // 4`.** That is exactly
the A-operand shape of `v_mfma_f32_16x16x16` -- and two reads give
`16x16x32`'s.

--- The maps, for a V-shaped LDS tile --------------------------------------

`A[m][k]`, `m` the head-dim column and `k` the staged token:

| shape | reads | per lane | `m` | `k` |
|---|---|---|---|---|
| `32x32x16` (the forward's) | 2 | 8 | `dc*32 + lane%32` | `16*sub + 4*(lane//32) + [0,1,2,3,8,9,10,11][i]` |
| `16x16x16` | **1** | 4 | `c*16 + lane%16` | `4*(lane//16) + i` |
| `16x16x32` | 2 | 8 | `c*16 + lane%16` | `8*(lane//16) + i` |

**The 16-row maps carry no permutation on `k`.** The 32-row one does, and every
consumer of it has to match -- `_pack_p_v8_slices` slices the score accumulator
in exactly that order and the two line up by coincidence. At 16 rows the
contraction index is the identity, so a `P`/`dS` operand only has to be laid
out in plain order.

The B operand and the accumulator have the same shape with `m` and `n`
exchanged, which the end-to-end probe confirms:

    B[k][n]: lane holds n = lane % 16,  k = quad*(lane//16) + i
    D[m][n]: lane holds n = lane % 16,  m = 4*(lane//16) + i     (4 f32)

**A lane's four accumulator elements are four *contiguous* rows.** At 32 rows
they are `8*(i//4) + 4*(lane//32) + (i%4)`, which is why the 32-row body needs
`_score_column_runs` to group the LSE and delta reads into four spans of four.
At 16 rows one `buffer_load_dwordx4` at row `4*(lane//16)` is the whole thing.

--- The trap ----------------------------------------------------------------

`tok_off` is **mixed-radix**, so `g * tok_off(q)` is not `tok_off(g*q)`. It
happens to hold at `quad = 8` with `SMEM_N_RPT = 8`, because eight tokens is
exactly one granule slot -- so a naive linear group term makes `16x16x32` work
and `16x16x16` silently wrong, with lanes 32..63 addressing past the end of the
tile and reading zeros. Finite, half-right, no diagnostic. `tok_off_dyn` is the
form to use for anything scaled by a runtime lane term.
"""

import flydsl.expr as fx

__all__ = [
    "MFMA16_M",
    "a16_chunk_offset",
    "a16_read_base",
    "acc16_row_base",
    "lds_elem",
    "tok_off",
    "tok_off_dyn",
]

# Rows (and columns) one `16x16x*` MFMA covers. A wave owns this many KV rows
# in the dK/dV kernel and this many Q rows in dQ.
MFMA16_M = 16


def tok_off(traits, t):
    """LDS element offset of advancing the staged token index by a **constant** `t`.

    `fmha_traits_gfx950`'s unified formula: a line holds `512 // granule` token
    slots, and slot `s` of line `n` is token `s * SMEM_N_RPT + n`.
    """
    return (t // traits.SMEM_N_RPT) * traits.D_128B_SIZE + (
        t % traits.SMEM_N_RPT
    ) * traits.SMEM_V_LINE_STRIDE


def tok_off_dyn(traits, t):
    """`tok_off` for a **runtime** token count. See the module docstring's trap."""
    n = fx.Index(traits.SMEM_N_RPT)
    return (t // n) * fx.Index(traits.D_128B_SIZE) + (t % n) * fx.Index(
        traits.SMEM_V_LINE_STRIDE
    )


def lds_elem(traits, t, d):
    """LDS element index of staged `(token, d)`, relative to the tile's base."""
    g = traits.D_128B_SIZE
    return (
        tok_off(traits, t)
        + (d // g) * traits.SMEM_N_RPT * traits.SMEM_V_LINE_STRIDE
        + (d % g)
    )


def a16_read_base(traits, lane, quad):
    """Per-lane LDS element base for a 16-row A operand's transpose read.

    `quad` is tokens per k-group: 4 for `16x16x16`, 8 for `16x16x32`. Add
    `a16_chunk_offset(traits, c)` for the `c`-th 16-wide output chunk, and for
    `16x16x32` read a second time at `+ tok_off(traits, 4)` elements.

    Three terms, and each names one axis of the pre-transpose layout:
    `lane // 16` is the k-group, `(lane % 16) // 4` is one of four consecutive
    tokens, `lane % 4` is which four of the sixteen d columns this lane
    fetches. The transpose turns the last into the operand's `m` axis.
    """
    return (
        tok_off_dyn(traits, (lane // fx.Index(MFMA16_M)) * fx.Index(quad))
        + ((lane % fx.Index(MFMA16_M)) // fx.Index(4))
        * fx.Index(traits.SMEM_V_LINE_STRIDE)
        + (lane % fx.Index(4)) * fx.Index(4)
    )


def a16_chunk_offset(traits, c):
    """LDS element offset of the `c`-th 16-wide d chunk. Compile-time `c`."""
    return lds_elem(traits, 0, c * MFMA16_M)


def acc16_row_base(lane_div_16):
    """First accumulator row a lane holds: its four are `base .. base + 3`.

    The whole of the 16-row LSE/delta addressing. At 32 rows the same thing
    needs a four-run table (`_score_column_runs`) because the rows a lane holds
    are not contiguous.
    """
    return lane_div_16 * fx.Index(4)
