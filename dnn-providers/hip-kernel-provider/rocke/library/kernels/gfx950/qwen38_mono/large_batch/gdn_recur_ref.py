# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host references for probe P-L1 (``gdn_recur``): inputs, an fp64 reference, a float32 reference of the textbook
formula, and an exact-order float32 emulation of the LB kernel (``emulate_lb``) that reproduces its bits.

Layouts (one layer; the harness stacks layers):
* ``state``: ``[rows, hv, DV, DK]`` (ATOM's ``[v, k]``), float32 (or bf16 bits as uint16 for ``state="bf16"``);
* ``qkvz``: ``[rows, row_elems]`` bf16 bits (uint16): q ``[hk*128]``, k ``[hk*128]``, v ``[hv*128]``, z ``[hv*128]``;
* ``gates``: ``[rows, hv, 2]`` float32 (g = log decay, beta);
* ``normw``: ``[128]`` bf16 bits;
* outputs ``y``: ``[rows, hv*128]`` bf16 bits; new state as the input layout (per snapshot row).

``emulate_lb`` mirrors the kernel lane by lane: fma chains in the lane's element order (exact fused multiply-add,
:func:`fma32`), the DPP butterflies (pairwise trees), the per-chunk / per-wave partial sums in their fixed order.
The hardware transcendentals (``v_rsq``, ``v_exp``, ``v_rcp``) are not reproducible on the host, so the emulation
takes them from a ``dbg`` build's dump and :func:`check_dbg` checks each against fp64 (a few ulp). Numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DK = DV = 128
EPS = np.float32(1e-6)
RSQ128 = np.array([0x3DB504F3], np.uint32).view(np.float32)[0]
LOG2E = np.float32(1.4426950408889634)
DBG_W = 136


# ------------------------------------------------------------------ bit helpers


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (np.asarray(u16, np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round to nearest even (finite inputs; NaN kept quiet)."""
    u = np.asarray(x, np.float32).view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    nan = np.isnan(np.asarray(x, np.float32))
    r = np.where(nan, (u >> 16) | 0x40, r)
    return r.astype(np.uint16)


def fma32(a, b, c) -> np.ndarray:
    """Exact float32 fused multiply-add (one rounding), vectorized: the float64 product of two float32 values is
    exact; the float64 sum's rounding error is recovered by TwoSum, and the one case where rounding the float64 sum
    to float32 rounds twice (the sum lands exactly on a float32 midpoint) is corrected toward the error's sign."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    c = np.asarray(c, np.float32)
    p = a.astype(np.float64) * b.astype(np.float64)
    c64 = c.astype(np.float64)
    s = p + c64
    bb = s - p
    err = (p - (s - bb)) + (c64 - bb)
    r = s.astype(np.float32)
    d = s - r.astype(np.float64)
    cand = (d != 0) & (err != 0)
    if np.any(cand):
        toward = np.where(d > 0, np.float32(np.inf), np.float32(-np.inf))
        nb = np.nextafter(r, toward)
        half = (nb.astype(np.float64) - r.astype(np.float64)) / 2
        fix = cand & (d == half) & (np.sign(err) == np.sign(d))
        r = np.where(fix, nb, r)
    return r


def butterfly(x: np.ndarray, axis: int, offsets) -> np.ndarray:
    """``x[i] + x[i ^ off]`` for each offset in turn along ``axis`` (float32): the DPP / swizzle / permlane
    butterflies of the kernel. Every index holds the same bits at the end (commutative pairs)."""
    x = np.asarray(x, np.float32)
    n = x.shape[axis]
    idx = np.arange(n)
    for off in offsets:
        x = (x + np.take(x, idx ^ off, axis=axis)).astype(np.float32)
    return x


def ulp_f32(x: np.ndarray) -> np.ndarray:
    x = np.abs(np.asarray(x, np.float32))
    return (np.nextafter(x, np.float32(np.inf)) - x).astype(np.float64)


def ulp_bf16(x: np.ndarray) -> np.ndarray:
    """bf16 ulp at |x| (of the bf16 grid around x)."""
    x = np.abs(np.asarray(x, np.float64))
    e = np.floor(np.log2(np.maximum(x, 1e-38)))
    return np.exp2(e - 7)


# ------------------------------------------------------------------ inputs


@dataclass
class Geo:
    hv: int
    hk: int

    @property
    def gqa(self):
        return self.hv // self.hk

    @property
    def key(self):
        return self.hk * DK

    @property
    def val(self):
        return self.hv * DV

    @property
    def row_elems(self):
        return 2 * self.key + 2 * self.val


def make_inputs(geo: Geo, rows: int, seed: int, state: str = "f32", state_scale: float = 0.5):
    """One layer's inputs. q / k / v / z ~ N(0, 1) (bf16), state ~ N(0, state_scale^2), g = -exp(U(-4, 1)) (decay
    0.066-0.98), beta = sigmoid(N(0, 1)), norm weight ~ N(1, 0.1^2) (bf16)."""
    rng = np.random.default_rng(seed)
    qkvz = f32_to_bf16(rng.standard_normal((rows, geo.row_elems)).astype(np.float32))
    st = (rng.standard_normal((rows, geo.hv, DV, DK)) * state_scale).astype(np.float32)
    if state == "bf16":
        st = f32_to_bf16(st)
    g = -np.exp(rng.uniform(-4.0, 1.0, (rows, geo.hv)))
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((rows, geo.hv))))
    gates = np.stack([g, beta], axis=-1).astype(np.float32)
    normw = f32_to_bf16((1.0 + 0.1 * rng.standard_normal(DV)).astype(np.float32))
    return dict(qkvz=qkvz, state=st, gates=gates, normw=normw)


def split_qkvz(geo: Geo, qkvz: np.ndarray):
    x = bf16_to_f32(qkvz)
    q = x[:, : geo.key].reshape(-1, geo.hk, DK)
    k = x[:, geo.key : 2 * geo.key].reshape(-1, geo.hk, DK)
    v = x[:, 2 * geo.key : 2 * geo.key + geo.val].reshape(-1, geo.hv, DV)
    z = x[:, 2 * geo.key + geo.val :].reshape(-1, geo.hv, DV)
    return q, k, v, z


def state_f32(st: np.ndarray) -> np.ndarray:
    return bf16_to_f32(st) if st.dtype == np.uint16 else st


# ------------------------------------------------------------------ references (formula order)


def reference(geo: Geo, inp: dict, chain: int = 1, dtype=np.float64):
    """The gated delta rule in ``dtype`` (float64 = the reference, float32 = the textbook float32 order). Returns
    (y ``[rows, val]`` float, state snapshots ``[rows, hv, DV, DK]``: row r's state after its own step; with
    ``chain`` > 1 the rows of a request chain from the request's first row's input state)."""
    q, k, v, z = split_qkvz(geo, inp["qkvz"])
    S0 = state_f32(inp["state"]).astype(dtype)
    g = inp["gates"][..., 0].astype(dtype)
    beta = inp["gates"][..., 1].astype(dtype)
    w = bf16_to_f32(inp["normw"]).astype(dtype)
    rows = q.shape[0]
    kh = np.arange(geo.hv) // geo.gqa
    eps = dtype(1e-6)
    y = np.zeros((rows, geo.hv, DV), dtype)
    snaps = np.zeros((rows, geo.hv, DV, DK), dtype)
    for r0 in range(0, rows, chain):
        S = S0[r0].copy()
        for t in range(chain):
            r = r0 + t
            qq = q[r, kh].astype(dtype)  # [hv, DK]
            kk = k[r, kh].astype(dtype)
            qn = qq / np.sqrt((qq * qq).sum(-1, keepdims=True) + eps) / np.sqrt(dtype(DK))
            kn = kk / np.sqrt((kk * kk).sum(-1, keepdims=True) + eps)
            S = S * np.exp(g[r])[:, None, None]
            kv = np.einsum("hvk,hk->hv", S, kn)
            delta = beta[r][:, None] * (v[r].astype(dtype) - kv)
            S = S + delta[:, :, None] * kn[:, None, :]
            o = np.einsum("hvk,hk->hv", S, qn)
            rr = 1.0 / np.sqrt((o * o).mean(-1, keepdims=True) + eps)
            sig = 1.0 / (1.0 + np.exp(-z[r].astype(dtype)))
            y[r] = o * rr * w[None, :] * sig
            snaps[r] = S
    return y.reshape(rows, geo.val), snaps


def reference_torch32(geo: Geo, inp: dict, chain: int = 1, device: str = "cuda"):
    """The float32 torch reference (formula order, torch's own reductions) on ``device``."""
    import torch

    q, k, v, z = (torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in split_qkvz(geo, inp["qkvz"]))
    S0 = torch.from_numpy(np.ascontiguousarray(state_f32(inp["state"]))).to(device)
    g = torch.from_numpy(inp["gates"][..., 0].copy()).to(device)
    beta = torch.from_numpy(inp["gates"][..., 1].copy()).to(device)
    w = torch.from_numpy(bf16_to_f32(inp["normw"])).to(device)
    rows = q.shape[0]
    kh = torch.arange(geo.hv, device=device) // geo.gqa
    y = torch.zeros((rows, geo.hv, DV), dtype=torch.float32, device=device)
    snaps = torch.zeros((rows, geo.hv, DV, DK), dtype=torch.float32, device=device)
    for r0 in range(0, rows, chain):
        S = S0[r0].clone()
        for t in range(chain):
            r = r0 + t
            qq, kk = q[r, kh], k[r, kh]
            qn = qq * torch.rsqrt((qq * qq).sum(-1, keepdim=True) + 1e-6) / (DK**0.5)
            kn = kk * torch.rsqrt((kk * kk).sum(-1, keepdim=True) + 1e-6)
            S = S * torch.exp(g[r])[:, None, None]
            kv = torch.einsum("hvk,hk->hv", S, kn)
            delta = beta[r][:, None] * (v[r] - kv)
            S = S + delta[:, :, None] * kn[:, None, :]
            o = torch.einsum("hvk,hk->hv", S, qn)
            rr = torch.rsqrt((o * o).mean(-1, keepdim=True) + 1e-6)
            y[r] = o * rr * w[None, :] * torch.sigmoid(z[r])
            snaps[r] = S
    return y.reshape(rows, geo.val).cpu().numpy(), snaps.cpu().numpy()


# ------------------------------------------------------------------ the LB kernel's exact order


@dataclass
class LbOrder:
    """The orders the LB kernel's knobs set: lanes per v-row (map=lane), the map, waves per unit, state dtype,
    chain."""

    lanes: int = 4
    waves: int = 8
    state: str = "f32"
    chain: int = 1
    map: str = "lane"

    @property
    def ve(self):
        return 4 if self.state == "f32" else 8

    @property
    def lr(self):  # lanes per v-row
        return DK // self.ve if self.map != "lane" else self.lanes

    @property
    def vpi(self):  # map=row: v-rows per access
        return 64 // self.lr

    @property
    def chunk_rows(self):
        return 16 if self.map != "lane" else 64 // self.lanes

    @property
    def chunks(self):
        return (DV // self.waves) // self.chunk_rows


def lane_k_order(o: LbOrder) -> np.ndarray:
    """``[lr, per]``: the k index of each element of lane j's in-order fma chain over one v-row (map=lane:
    interleaved 16-byte accesses ``ve (j + lanes i) + e``; map=row: the lane's one access ``ve j + e``)."""
    L = o.lr
    per = DK // L
    out = np.zeros((L, per), np.int64)
    for j in range(L):
        m = 0
        for i in range(per // o.ve):
            for e in range(o.ve):
                out[j, m] = o.ve * (j + L * i) + e
                m += 1
    return out


def _norm_sums(x: np.ndarray, ve: int) -> np.ndarray:
    """``x``: ``[..., 128]``: each lane's fma chain over its ``ve`` consecutive values (q in lanes 0-31 /
    k in 32-63 for map=lane; the lane's own quad for map=row), then the butterfly over the ``128 / ve`` lanes;
    returns the total ``[...]`` (every lane's bits)."""
    xl = x.reshape(*x.shape[:-1], DK // ve, ve)
    s = (xl[..., 0] * xl[..., 0]).astype(np.float32)
    for e in range(1, ve):
        s = fma32(xl[..., e], xl[..., e], s)
    n = DK // ve
    s = butterfly(s, -1, [1 << i for i in range(n.bit_length() - 1)])
    return s[..., 0]


def emulate_lb(geo: Geo, inp: dict, order: LbOrder, dbg: np.ndarray):
    """The LB kernel's result, bit for bit, given its ``dbg`` dump ``[rows, hv, DBG_W]`` (float32: inv_q, inv_k,
    decay, inv_rms, beta, -, -, -, sig(z)[128]). Returns (y bf16 bits ``[rows, val]``, new state
    ``[rows, hv, DV, DK]`` in the storage dtype for the snapshot rows (others zero), and the per-row quantities
    the dbg check needs: the q / k sums of squares and the RMS totals)."""
    q, k, v, z = split_qkvz(geo, inp["qkvz"])
    rows = q.shape[0]
    G, L, W = order.chain, order.lr, order.waves
    kh = np.arange(geo.hv) // geo.gqa
    korder = lane_k_order(order)  # [L, per]
    S_in = state_f32(inp["state"])
    beta_all = inp["gates"][..., 1].astype(np.float32)
    w = bf16_to_f32(inp["normw"])
    y = np.zeros((rows, geo.hv, DV), np.uint16)
    snaps = np.zeros((rows, geo.hv, DV, DK), np.float32)
    sums = np.zeros((rows, geo.hv, 3), np.float32)  # q sumsq, k sumsq, rms total
    for r0 in range(0, rows, G):
        S = S_in[r0].copy()  # [hv, DV, DK]
        for t in range(G):
            r = r0 + t
            nve = order.ve if order.map != "lane" else 4
            sums[r, :, 0] = _norm_sums(q[r, kh], nve)
            sums[r, :, 1] = _norm_sums(k[r, kh], nve)
            invq = dbg[r, :, 0].astype(np.float32)
            invk = dbg[r, :, 1].astype(np.float32)
            decay = dbg[r, :, 2].astype(np.float32)
            beta = beta_all[r]
            qh = (q[r, kh] * invq[:, None]).astype(np.float32)  # [hv, DK]
            khat = (k[r, kh] * invk[:, None]).astype(np.float32)
            sv = (S * decay[:, None, None]).astype(np.float32)  # [hv, DV, DK]
            # kv: per lane j an fma chain over korder[j], then the in-group butterfly over j
            svl = sv[:, :, korder]  # [hv, DV, L, per]
            khl = khat[:, korder]  # [hv, L, per]
            acc = (svl[..., 0] * khl[:, None, :, 0]).astype(np.float32)
            for m in range(1, korder.shape[1]):
                acc = fma32(svl[..., m], khl[:, None, :, m], acc)
            kv = butterfly(acc, -1, [1 << s for s in range(L.bit_length() - 1)])[..., 0]  # [hv, DV]
            delta = (beta[:, None] * (v[r] - kv).astype(np.float32)).astype(np.float32)
            Sn = fma32(delta[:, :, None], khat[:, None, :], sv)
            snl = Sn[:, :, korder]
            qhl = qh[:, korder]
            acc = (snl[..., 0] * qhl[:, None, :, 0]).astype(np.float32)
            for m in range(1, korder.shape[1]):
                acc = fma32(snl[..., m], qhl[:, None, :, m], acc)
            o = butterfly(acc, -1, [1 << s for s in range(L.bit_length() - 1)])[..., 0]  # [hv, DV]
            # stored copy: bf16 storage rounds (RNE); the chain's next step keeps the f32 Sn in registers
            snaps[r] = Sn if order.state == "f32" else bf16_to_f32(f32_to_bf16(Sn))
            S = Sn
            # RMS: per wave ws, per chunk c: the chunk's partial (map=lane: the over-group butterfly of o^2;
            # map=row: per lane (vsub) the fma chain over its accesses i of o^2 at v-row i*vpi + vsub, then the
            # butterfly over vsub), then the chunk sum, then the waves in order
            CR, C = order.chunk_rows, order.chunks
            if order.map == "lane":
                o2 = (o * o).astype(np.float32).reshape(geo.hv, W, C, CR)
                sc = butterfly(o2, -1, [1 << s for s in range(CR.bit_length() - 1)])[..., 0]  # [hv, W, C]
            elif order.map == "rowi":
                # wave ws holds v-rows i*8*vpi + ws*vpi + vsub (one chunk)
                VPI = order.vpi
                ov = o.reshape(geo.hv, CR // VPI, 8, VPI)  # [hv, i, ws, vsub]
                acc = (ov[:, 0] * ov[:, 0]).astype(np.float32)  # [hv, ws, vsub]
                for i in range(1, CR // VPI):
                    acc = fma32(ov[:, i], ov[:, i], acc)
                sc = butterfly(acc, -1, [1 << s for s in range(VPI.bit_length() - 1)])[..., 0][:, :, None]  # [hv, W, C=1]
            else:
                VPI = order.vpi
                ov = o.reshape(geo.hv, W, C, CR // VPI, VPI)  # [.., i, vsub]
                acc = (ov[..., 0, :] * ov[..., 0, :]).astype(np.float32)
                for i in range(1, CR // VPI):
                    acc = fma32(ov[..., i, :], ov[..., i, :], acc)
                sc = butterfly(acc, -1, [1 << s for s in range(VPI.bit_length() - 1)])[..., 0]
            P = sc[:, :, 0]
            for c in range(1, C):
                P = (P + sc[:, :, c]).astype(np.float32)
            tot = P[:, 0]
            for ws in range(1, W):
                tot = (tot + P[:, ws]).astype(np.float32)
            sums[r, :, 2] = tot
            inv = dbg[r, :, 3].astype(np.float32)
            sig = dbg[r, :, 8 : 8 + DV].astype(np.float32)
            yv = ((o * inv[:, None]).astype(np.float32) * w[None, :]).astype(np.float32)
            yv = (yv * sig).astype(np.float32)
            y[r] = f32_to_bf16(yv)
    return y.reshape(rows, geo.val), snaps, sums


def check_dbg(geo: Geo, inp: dict, dbg: np.ndarray, sums: np.ndarray, max_ulp: float = 4.0) -> dict:
    """Each dumped transcendental against fp64 from the emulated (exact) operand: inv_q = rsq(sq + eps) * rsq128,
    inv_k = rsq(sk + eps), decay = exp2(f32(g log2 e)), inv_rms = rsq(f32(tot / 128) + eps), sig = rcp(1 +
    exp2(f32(-z log2 e))). Returns the max error in ulp of each."""
    q, k, v, z = split_qkvz(geo, inp["qkvz"])
    g = inp["gates"][..., 0].astype(np.float32)
    out = {}

    def ulps(got, want):
        got = np.asarray(got, np.float32)
        return float(np.max(np.abs(got.astype(np.float64) - want) / ulp_f32(want.astype(np.float32))))

    sq = (sums[..., 0] + EPS).astype(np.float32)
    out["inv_q"] = ulps(dbg[..., 0], (1.0 / np.sqrt(sq.astype(np.float64))).astype(np.float32).astype(np.float64)
                        * np.float64(RSQ128))
    sk = (sums[..., 1] + EPS).astype(np.float32)
    out["inv_k"] = ulps(dbg[..., 1], 1.0 / np.sqrt(sk.astype(np.float64)))
    gl = (g * LOG2E).astype(np.float32)
    out["decay"] = ulps(dbg[..., 2], np.exp2(gl.astype(np.float64)))
    tr = (sums[..., 2] * np.float32(1.0 / DV)).astype(np.float32)
    tr = (tr + EPS).astype(np.float32)
    out["inv_rms"] = ulps(dbg[..., 3], 1.0 / np.sqrt(tr.astype(np.float64)))
    out["beta"] = float(np.max(np.abs(dbg[..., 4] - inp["gates"][..., 1])))
    zl = ((np.float32(0.0) - z) * LOG2E).astype(np.float32)
    e = np.exp2(zl.astype(np.float64)).astype(np.float32)
    den = (np.float32(1.0) + e).astype(np.float32)
    out["sig"] = ulps(dbg[..., 8 : 8 + DV], 1.0 / den.astype(np.float64))
    out["ok"] = all(out[x] <= max_ulp for x in ("inv_q", "inv_k", "decay", "inv_rms", "sig")) and out["beta"] == 0.0
    return out


# ------------------------------------------------------------------ tolerance metrics


def tol_metrics(y_bits: np.ndarray, snaps: np.ndarray, y64: np.ndarray, s64: np.ndarray, rows_mask=None) -> dict:
    """Output error vs the fp64 reference: per (row, head) the max |error| over the head's 128 outputs relative to
    the head's max |y| (``y_head_rel_max``, the gate metric: an element-wise ulp measure explodes on outputs near
    zero), plus the element-wise bf16-ulp max and the fraction above 1 ulp (informational). State error relative
    to the largest |state| (max, rms). Over the snapshot rows given by ``rows_mask`` (y: every row)."""
    yk = bf16_to_f32(y_bits).astype(np.float64) if y_bits.dtype == np.uint16 else np.asarray(y_bits, np.float64)
    m = rows_mask if rows_mask is not None else slice(None)
    ey = np.abs(yk - y64) / ulp_bf16(y64)
    rows = yk.shape[0]
    yh = yk.reshape(rows, -1, DV)
    rh = np.asarray(y64, np.float64).reshape(rows, -1, DV)
    eh = np.abs(yh - rh).max(-1) / np.maximum(np.abs(rh).max(-1), 1e-30)
    sk = snaps[m].astype(np.float64)
    ref = s64[m]
    scale = float(np.max(np.abs(ref))) or 1.0
    es = np.abs(sk - ref) / scale
    return dict(
        y_head_rel_max=float(eh.max()),
        y_ulp_max=float(ey.max()),
        y_frac_gt1=float((ey > 1.0).mean()),
        s_rel_max=float(es.max()),
        s_rel_rms=float(np.sqrt((es**2).mean())),
        finite=bool(np.isfinite(yk).all() and np.isfinite(sk).all()),
    )
