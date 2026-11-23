# Not vendored: written for this provider; it has no upstream.
"""Row and offset arithmetic of the tail-zero kernels, without FlyDSL.

The tail-zero kernel (`fmha_bwd_tailzero_gfx950.py`; `../attention_bwd_rdna/fmha_bwd_tailzero_gfx11.py` for gfx11) writes zero
to the logical rows `seq_len[b] <= row < S` of each dense gradient output of a padded graph (`padding_mask` set), over the
output's own `(batch, head, seq)` element strides and a contiguous last dimension. Every function here is plain `+ - * //`
arithmetic and comparisons, so the kernel evaluates it on FlyDSL values and `written_elements` evaluates the same functions on
Python integers: the host tests check the addressing the object runs.

Launch: block `BLOCK_THREADS`; grid `(grid_blocks(units), planes_of(...), 1)`. A plane is one `(batch, head)` of dQ, dK or dV:
`plane = b * heads_total + r`, with `r` in `[0, Hq)` for dQ, `[Hq, Hq + Hk)` for dK and `[Hq + Hk, Hq + Hk + Hv)` for dV
(`heads_total = Hq + Hk + Hv`; `Hv` is `Hk` unless the V-head variant passes its own count). A thread owns one unit: `vec_of(elem_bytes)` consecutive columns of one tail row of one plane.
`mask` bit 0 / 1 / 2 selects dQ / dK / dV; an output whose bit is clear is not written (a ragged output's storage past a length
belongs to a gap or to the next sequence).

**One copy.** The gfx950 and the gfx11 kernels both import this file.
"""

BLOCK_THREADS = 256
VECTOR_BYTES = 16  # one 128-bit access
MASK_DQ = 1
MASK_DK = 2
MASK_DV = 4


def vec_of(elem_bytes):
    """Columns one thread owns: one 16-byte vector of elements."""
    return VECTOR_BYTES // elem_bytes


def heads_total(hq, hkv, hv=None):
    """Planes per batch; dV has `hv` heads, `hkv` when `hv` is None."""
    return hq + hkv + (hkv if hv is None else hv)


def units_per_row(dim, vec):
    """Vectors per row of a `dim`-column tensor (the last one may be partial)."""
    return (dim + vec - 1) // vec


def units_of(seq, dim, vec):
    """Upper bound on the units of one tensor's tail rows: every row of the `seq`-row extent."""
    return seq * units_per_row(dim, vec)


def grid_blocks(units):
    """Blocks along grid.x for a `units`-unit extent."""
    return (units + BLOCK_THREADS - 1) // BLOCK_THREADS


def batch_of(plane, heads):
    """The batch a plane belongs to; `heads` is `heads_total`."""
    return plane // heads


def row_of_unit(index, nvec, vec):
    """`(row offset inside the tail, first column)` of unit `index`; `nvec` is `units_per_row`."""
    q = index // nvec
    return q, (index - q * nvec) * vec


def element_offset(head_offset, row, col, stride_seq):
    """Element offset of `(row, col)` from a plane's origin `head_offset` (last dim contiguous)."""
    return head_offset + row * stride_seq + col


def written_elements(
    seq_len_q,
    seq_len_kv,
    hq,
    hkv,
    sq,
    skv,
    dqk,
    dv,
    strides,
    mask,
    elem_bytes,
    hv=None,
):
    """The element offsets the kernel writes, as `{(tensor, offset)}` with tensor 0 / 1 / 2 for dQ / dK / dV.

    Evaluates the kernel's own arithmetic over its whole grid on Python integers. `seq_len_q` and `seq_len_kv` are the
    `[B]` lengths; `strides` is `((b, h, s) of dQ, (b, h, s) of dK, (b, h, s) of dV)` in elements; offsets are from each
    tensor's base pointer. `hv` is dV's head count, `hkv` when None.
    """
    vec = vec_of(elem_bytes)
    batch = len(seq_len_q)
    hv = hkv if hv is None else hv
    heads = heads_total(hq, hkv, hv)
    extents = ((sq, dqk, hq, 0), (skv, dqk, hkv, hq), (skv, dv, hv, hq + hkv))
    units = max(units_of(s, d, vec) for s, d, _, _ in extents)
    written = set()
    for plane in range(batch * heads):
        b = batch_of(plane, heads)
        r = plane - b * heads
        for tensor, (seq, dim, count, first) in enumerate(extents):
            if not (first <= r < first + count) or not mask & (1 << tensor):
                continue
            head = r - first
            length = (seq_len_q if tensor == 0 else seq_len_kv)[b]
            nvec = units_per_row(dim, vec)
            sb, sh, ss = strides[tensor]
            origin = b * sb + head * sh
            for block in range(grid_blocks(units)):
                for tid in range(BLOCK_THREADS):
                    index = block * BLOCK_THREADS + tid
                    q, col0 = row_of_unit(index, nvec, vec)
                    row = length + q
                    if row >= seq:
                        continue
                    for col in range(col0, min(col0 + vec, dim)):
                        written.add((tensor, element_offset(origin, row, col, ss)))
    return written
