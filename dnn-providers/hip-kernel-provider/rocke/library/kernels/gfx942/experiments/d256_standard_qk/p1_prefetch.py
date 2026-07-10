# PROTOTYPE (AICK-1495 next-step): register double-buffer PREFETCH on the QK inner k-loop.
# Base = 04_realworkload_bf16_gqa_vs_aotriton.py (bf16 + GQA 16/2 + causal). Knobs via env:
#   PF   = QK-loop software-pipeline prefetch depth (0 = baseline mfma_k_loop; 1 = depth-1 reg double-buffer)
#   DUMP = write the built hsaco to /tmp/k.hsaco (for llvm-objdump resource inspection)
import os, sys, torch, math
import kernels.common.attention_unified as au

au._RESOLVED_ATTENTION_ARCH = "gfx942"
from rocke.core.ir import F16, BF16, F32, I32, IRBuilder, PtrType
from rocke.helpers.atoms import MfmaAtom
from rocke.helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    mfma_atom_for_dtype,
    load_a_row_major_contiguous,
    load_b_col_strided_scalars,
    mfma_k_loop,
)
from rocke.helpers import SignatureBuilder
from rocke import compile_kernel
from rocke.runtime import (
    KernelLauncher,
    LaunchConfig,
    time_launches,
    synchronize_and_release,
)

D = 256
LOG2E = 1.4426950408889634
SQ = int(sys.argv[1]) if len(sys.argv) > 1 else 512
SK = SQ
NQB = SQ // 32
H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
HKV = 2
GQ = H // HKV
PF = int(os.environ.get("PF", "2"))  # best-config default: QK-unroll (measured winner)
VLDS = int(
    os.environ.get("VLDS", "0")
)  # (a) stage V tile in LDS via wide coalesced HBM load, read B-operand from LDS
ILV = int(
    os.environ.get("ILV", "0")
)  # (b) interleave PV nt-tile MFMAs in groups of ILV to space acc-read hazards
WPE = int(
    os.environ.get("WPE", "0")
)  # amdgpu-waves-per-eu occupancy hint: force allocator to fit N waves/EU (spills if needed)
SMX = int(
    os.environ.get("SMX", "0")
)  # softmax: fold log2e into S scale + causal phase-split (mask only the diagonal kv-tile)


def build():
    at = MfmaAtom.bf16_32x32x8()
    ND = D // 32
    CPL = at.c_per_lane
    b = IRBuilder("stdqk_pf")
    b.kernel.attrs["max_workgroup_size"] = 64
    if WPE:
        b.kernel.attrs["waves_per_eu"] = WPE
    Q = b.param("Q", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    K = b.param("K", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    V = b.param("V", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    O = b.param("O", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    lane = b.thread_id_x()
    ld = decode_mfma_lanes(b, at, lane)
    zb = b.const_i32(0)
    qblk = b.block_id_x()
    qbase = b.mul(qblk, b.const_i32(32))
    head = b.block_id_y()
    hoq = b.mul(head, b.const_i32(SQ))
    hok = b.mul(
        b.div(head, b.const_i32(GQ)), b.const_i32(SK)
    )  # GQA: kv_head=q_head//GQ
    S_lds = b.smem_alloc(
        F32, [32, 33], name_hint="Slds"
    )  # +1 col pad: break stride-32 bank conflict
    m_lds = b.smem_alloc(F32, [32], name_hint="mlds")
    l_lds = b.smem_alloc(F32, [32], name_hint="llds")
    c_lds = b.smem_alloc(F32, [32], name_hint="clds")
    if VLDS:
        V_lds = b.smem_alloc(
            BF16, [32 * D], name_hint="Vlds"
        )  # 16KB tile-local V (k,n) staging
    scale = b.const_f32(1.0 / math.sqrt(D))
    l2e = b.const_f32(LOG2E)
    ninf = b.const_f32(-1e30)
    row = b.mod(lane, b.const_i32(32))
    qabs = b.add(qbase, row)
    b.smem_store_vN(m_lds, [row], ninf, 1)
    b.smem_store_vN(l_lds, [row], b.const_f32(0.0), 1)
    b.sync()
    iter_args = [(f"acc{n}", b.const_f32(0.0)) for n in range(ND * CPL)]
    kvloop = b.scf_for_iter(
        zb, b.add(qblk, b.const_i32(1)), b.const_i32(1), iter_args, iv_name="kv"
    )
    with kvloop as (kv, carry):
        kvb = b.mul(kv, b.const_i32(32))

        def la(bb, kt):
            return load_a_row_major_contiguous(
                bb,
                A=Q,
                atom=at,
                lane_decode=ld,
                m_tile_base=bb.add(qbase, hoq),
                k_tile_base=bb.mul(kt, bb.const_i32(at.k)),
                K=D,
            )

        def lb(bb, kt):
            return load_a_row_major_contiguous(
                bb,
                A=K,
                atom=at,
                lane_decode=ld,
                m_tile_base=bb.add(kvb, hok),
                k_tile_base=bb.mul(kt, bb.const_i32(at.k)),
                K=D,
            )

        if PF >= 2:
            # Python-unrolled QK: expose all D/atom.k steps so the compiler can hoist
            # the (independent) K/Q loads ahead of the dependent MFMA chain -> batch the
            # 32 vmcnt(0) drains into a few. PF>=3 also groups loads in blocks of PF.
            nqk = D // at.k
            acc = at.zero_acc(b)
            G = PF if PF >= 3 else nqk
            for g0 in range(0, nqk, G):
                grp = list(range(g0, min(g0 + G, nqk)))
                av = [la(b, b.const_i32(kt)) for kt in grp]
                bv = [lb(b, b.const_i32(kt)) for kt in grp]
                for a_i, b_i in zip(av, bv):
                    acc = at.emit(b, a_i, b_i, acc)
            s_acc = acc
        elif PF >= 1:
            # Software-pipelined QK: carry (acc, Q-frag, K-frag); issue step kt+1's loads
            # (into registers) before the step-kt MFMA so the L2/HBM load latency overlaps
            # compute. kt+1 is clamped to the last step so the trailing (unused) prefetch
            # never reads out of bounds.
            nqk = D // at.k
            a0 = la(b, zb)
            b0 = lb(b, zb)
            qk = b.scf_for_iter(
                zb,
                b.const_i32(nqk),
                b.const_i32(1),
                [("accqk", at.zero_acc(b)), ("acur", a0), ("bcur", b0)],
                iv_name="ktqk",
            )
            with qk as (kt, (acc_v, a_cur, b_cur)):
                ktn = b.add(kt, b.const_i32(1))
                ktc = b.select(
                    b.cmp_lt(ktn, b.const_i32(nqk)), ktn, b.const_i32(nqk - 1)
                )
                a_nx = la(b, ktc)
                b_nx = lb(b, ktc)
                b.scf_yield(at.emit(b, a_cur, b_cur, acc_v), a_nx, b_nx)
            s_acc = qk.results[0]
        else:
            s_acc = mfma_k_loop(
                b, K=D, atom=at, load_a=la, load_b=lb, iv_name="ktqk", acc_name="accqk"
            )
        # SMX: fold (scale*log2e) into the S write so exp2/corr drop their per-element *l2e.
        sfac = b.const_f32((1.0 / math.sqrt(D)) * LOG2E) if SMX else scale
        for i in range(CPL):
            r, c = at.lane_to_output(b, lane, i)
            b.smem_store_vN(S_lds, [r, c], b.fmul(b.vec_extract(s_acc, i), sfac), 1)
        b.sync()

        def _softmax(masked, use_l2e):
            m_old = b.vec_extract(b.smem_load_vN(m_lds, row, dtype=F32, n=1), 0)
            l_old = b.vec_extract(b.smem_load_vN(l_lds, row, dtype=F32, n=1), 0)
            mx = m_old
            for jj in range(8):
                v4 = b.smem_load_vN(S_lds, row, b.const_i32(4 * jj), dtype=F32, n=4)
                for e in range(4):
                    j = 4 * jj + e
                    ve = b.vec_extract(v4, e)
                    v = (
                        b.select(b.cmp_gt(b.add(kvb, b.const_i32(j)), qabs), ninf, ve)
                        if masked
                        else ve
                    )
                    mx = b.fmax(v, mx)
            corr = (
                b.exp2(b.fmul(b.fsub(m_old, mx), l2e))
                if use_l2e
                else b.exp2(b.fsub(m_old, mx))
            )
            ssum = b.const_f32(0.0)
            for jj in range(8):
                v4 = b.smem_load_vN(S_lds, row, b.const_i32(4 * jj), dtype=F32, n=4)
                ps = []
                for e in range(4):
                    j = 4 * jj + e
                    ve = b.vec_extract(v4, e)
                    v = (
                        b.select(b.cmp_gt(b.add(kvb, b.const_i32(j)), qabs), ninf, ve)
                        if masked
                        else ve
                    )
                    p = (
                        b.exp2(b.fmul(b.fsub(v, mx), l2e))
                        if use_l2e
                        else b.exp2(b.fsub(v, mx))
                    )
                    ps.append(p)
                    ssum = b.fadd(ssum, p)
                b.smem_store_vN(
                    S_lds, [row, b.const_i32(4 * jj)], b.vec_pack(ps, F32), 4
                )
            b.smem_store_vN(m_lds, [row], mx, 1)
            b.smem_store_vN(l_lds, [row], b.fadd(b.fmul(l_old, corr), ssum), 1)
            b.smem_store_vN(c_lds, [row], corr, 1)

        with b.scf_if(b.cmp_lt(lane, b.const_i32(32))):
            if SMX:
                # causal phase-split: below-diagonal kv-tiles are fully valid (skip the
                # per-element mask select); only the diagonal tile kv==qblk is masked.
                with b.scf_if(b.cmp_lt(kv, qblk)):
                    _softmax(masked=False, use_l2e=False)
                with b.scf_if(b.cmp_eq(kv, qblk)):
                    _softmax(masked=True, use_l2e=False)
            else:
                _softmax(masked=True, use_l2e=True)
        b.sync()
        # (a) VLDS: cooperative wide coalesced V[tile] HBM->LDS (16 x dwordx4/lane), one drain
        if VLDS:
            vbase = b.mul(b.add(hok, kvb), b.const_i32(D))
            for i in range(16):
                off = b.add(b.const_i32(i * 512), b.mul(lane, b.const_i32(8)))
                b.smem_store_vN(
                    V_lds,
                    [off],
                    b.global_load_vN(V, b.add(vbase, off), BF16, 8, align=16),
                    8,
                )
            b.sync()
        new_acc = [None] * (ND * CPL)

        def lpa(bb, kt):
            qn = bb.add(zb, ld.m_in_atom)
            kbase = bb.add(
                bb.mul(kt, bb.const_i32(at.k)),
                bb.mul(ld.k_blk, bb.const_i32(at.a_per_lane)),
            )
            el = [
                bb.cast_f32_to(
                    bb.vec_extract(
                        bb.smem_load_vN(
                            S_lds, qn, bb.add(kbase, bb.const_i32(j)), dtype=F32, n=1
                        ),
                        0,
                    ),
                    BF16,
                )
                for j in range(at.a_per_lane)
            ]
            return bb.vec_pack(el, BF16)

        def lvb_g(bb, kt, nb):
            return load_b_col_strided_scalars(
                bb,
                B=V,
                atom=at,
                lane_decode=ld,
                n_tile_base=nb,
                k_tile_base=bb.add(bb.add(hok, kvb), bb.mul(kt, bb.const_i32(at.k))),
                N=D,
            )

        def lvb_l(bb, kt, nb):
            ncol = bb.add(nb, ld.n_in_atom)
            kb = bb.add(
                bb.mul(kt, bb.const_i32(at.k)),
                bb.mul(ld.k_blk, bb.const_i32(at.b_per_lane)),
            )
            el = [
                bb.vec_extract(
                    bb.smem_load_vN(
                        V_lds,
                        bb.add(
                            bb.mul(bb.add(kb, bb.const_i32(j)), bb.const_i32(D)), ncol
                        ),
                        dtype=BF16,
                        n=1,
                    ),
                    0,
                )
                for j in range(at.b_per_lane)
            ]
            return bb.vec_pack(el, BF16)

        lvb = lvb_l if VLDS else lvb_g
        cr_i = [
            b.vec_extract(
                b.smem_load_vN(c_lds, at.lane_to_output(b, lane, i)[0], dtype=F32, n=1),
                0,
            )
            for i in range(CPL)
        ]
        if ILV >= 1 or VLDS:
            # (b) unrolled+interleaved PV: within a group of G nt-tiles, issue one MFMA per
            # tile at each k-step so consecutive MFMAs target distinct accs (hazard-free).
            G = ILV if ILV >= 1 else 1
            nkp = 32 // at.k
            for g0 in range(0, ND, G):
                grp = list(range(g0, min(g0 + G, ND)))
                accs = [at.zero_acc(b) for _ in grp]
                for kt in range(nkp):
                    a = lpa(b, b.const_i32(kt))
                    for gi, nt in enumerate(grp):
                        accs[gi] = at.emit(
                            b,
                            a,
                            lvb(b, b.const_i32(kt), b.const_i32(nt * 32)),
                            accs[gi],
                        )
                for gi, nt in enumerate(grp):
                    for i in range(CPL):
                        new_acc[nt * CPL + i] = b.fma(
                            carry[nt * CPL + i], cr_i[i], b.vec_extract(accs[gi], i)
                        )
        else:
            for nt in range(ND):
                nb = b.const_i32(nt * 32)
                pv = mfma_k_loop(
                    b,
                    K=32,
                    atom=at,
                    load_a=lpa,
                    load_b=lambda bb, kt, nb=nb: lvb(bb, kt, nb),
                    iv_name=f"ktpv{nt}",
                    acc_name=f"accpv{nt}",
                )
                for i in range(CPL):
                    new_acc[nt * CPL + i] = b.fma(
                        carry[nt * CPL + i], cr_i[i], b.vec_extract(pv, i)
                    )
        b.scf_yield(*new_acc)
    final = kvloop.results
    for nt in range(ND):
        nbase = b.const_i32(nt * 32)
        for i in range(CPL):
            r, c = at.lane_to_output(b, lane, i)
            lv = b.vec_extract(b.smem_load_vN(l_lds, r, dtype=F32, n=1), 0)
            addr = b.add(
                b.mul(b.add(b.add(hoq, qbase), r), b.const_i32(D)), b.add(nbase, c)
            )
            b.global_store(O, addr, b.fdiv(final[nt * CPL + i], lv), align=4)
    b.ret()
    return b.kernel


art = compile_kernel(build(), arch="gfx942")
print(f"built PF={PF}", flush=True)
if os.environ.get("DUMP"):
    open("/tmp/k.hsaco", "wb").write(art.hsaco)
    print("dumped /tmp/k.hsaco", len(art.hsaco), "bytes")
torch.manual_seed(0)
q = torch.randn(H, SQ, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(HKV, SK, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(HKV, SK, D, device="cuda", dtype=torch.bfloat16)
o = torch.zeros(H, SQ, D, device="cuda", dtype=torch.float32)
sig = (
    SignatureBuilder()
    .ptr("Q", "bf16")
    .ptr("K", "bf16")
    .ptr("V", "bf16")
    .ptr("O", "f32")
    .build()
)
L = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)
hs = torch.cuda.current_stream().cuda_stream
cfg = LaunchConfig(grid=(NQB, H, 1), block=(64, 1, 1), stream=hs)
L({"Q": q, "K": k, "V": v, "O": o}, config=cfg)
torch.cuda.synchronize()
ke = k.repeat_interleave(GQ, dim=0)
ve = v.repeat_interleave(GQ, dim=0)
ref = torch.nn.functional.scaled_dot_product_attention(
    q.float()[None], ke.float()[None], ve.float()[None], is_causal=True
)[0]
err = (o - ref).abs().max().item()
print(
    f"GQA-CAUSAL SQ={SQ} H={H} PF={PF} blocks={NQB*H} max_abs_err={err:.4e}  {'CORRECT' if err<0.2 else 'WRONG'}"
)


def once():
    L({"Q": q, "K": k, "V": v, "O": o}, config=cfg)


ms = time_launches(once, warmup=10, iters=50, stream=hs)
synchronize_and_release(hs)
flop = 2.0 * (2.0 * SQ * SK * D) * 0.5 * H
print(f"time={ms*1e3:.1f}us  TF/s(causal)={flop/(ms*1e-3)/1e12:.2f}")
