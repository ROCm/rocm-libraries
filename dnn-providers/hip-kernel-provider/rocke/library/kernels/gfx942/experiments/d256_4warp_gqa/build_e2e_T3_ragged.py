import sys, math, os
from rocke.core.ir import BF16, F32, I32, IRBuilder, PtrType
from rocke.helpers.atoms import MfmaAtom
from rocke.helpers.mfma_gemm_inner import decode_mfma_lanes
from rocke import compile_kernel
from rocke.helpers import SignatureBuilder
from rocke.runtime import KernelLauncher, LaunchConfig, time_launches, synchronize_and_release
import torch
RESOLVED_ATTENTION_ARCH = "gfx942"
at = MfmaAtom.bf16_32x32x8()
APL, BPL, CPL, K = at.a_per_lane, at.b_per_lane, at.c_per_lane, at.k
HD = 256; BN = 64; NKEYT = BN//32; NK = HD//K; NDdim = HD//32; NKpv = BN//K
BS = 16; BPT = BN//BS
NPHYS = 32           # per-seq physical blocks (max klen = NPHYS*BS = 512)
H = 16; HKV = 2; GQAG = H//HKV
# ragged config (host sets these before build via globals)
NSEQ = 3; TOTALQ = 928; TOTALQB = 10

def build():
    b = IRBuilder("e2eRagged"); b.kernel.attrs["max_workgroup_size"] = 256
    Q  = b.param("Q",  PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Kp = b.param("K",  PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Vp = b.param("V",  PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    C  = b.param("C",  PtrType(F32,  "global"), noalias=True, writeonly=True, align=16)
    BT = b.param("BT", PtrType(I32,  "global"), noalias=True, readonly=True, align=16)
    KL = b.param("KL", PtrType(I32,  "global"), noalias=True, readonly=True, align=16)  # per-seq seqlen_k [NSEQ]
    CUQ= b.param("CUQ",PtrType(I32,  "global"), noalias=True, readonly=True, align=16)  # cu_seqlens_q [NSEQ+1]
    SID= b.param("SID",PtrType(I32,  "global"), noalias=True, readonly=True, align=16)  # seq per global qblock [TOTALQB]
    LQ = b.param("LQ", PtrType(I32,  "global"), noalias=True, readonly=True, align=16)  # local qblock [TOTALQB]
    tid = b.thread_id_x(); wid = b.div(tid, b.const_i32(64)); lane = b.mod(tid, b.const_i32(64)); ld = decode_mfma_lanes(b, at, lane)
    wq = b.mul(wid, b.const_i32(32))
    bid = b.block_id_x(); qhead = b.mod(bid, b.const_i32(H)); kvh = b.div(qhead, b.const_i32(GQAG)); gqb = b.div(bid, b.const_i32(H))
    def li(param, rsrc, idx):  # load I32 scalar param[idx]
        r = b.buffer_rsrc(param, b.const_i32((TOTALQB+NSEQ+2)*4))
        return b.vec_extract(b.buffer_load_vN(r, b.mul(idx, b.const_i32(4)), b.const_i32(0), I32, 1), 0)
    sid = li(SID, None, gqb)
    lqb = li(LQ, None, gqb)
    cuq_s = li(CUQ, None, sid); cuq_s1 = li(CUQ, None, b.add(sid, b.const_i32(1)))
    qlen = b.sub(cuq_s1, cuq_s)
    klen = li(KL, None, sid)
    qbase = b.mul(lqb, b.const_i32(128))           # in-seq query position of this block
    qstart = b.add(cuq_s, qbase)                    # packed row of first query
    V_lds = b.smem_alloc(BF16, [64, 256], name_hint="Vlds")
    qr = b.buffer_rsrc(Q, b.const_i32(TOTALQ*H*HD*2))
    kr = b.buffer_rsrc(Kp, b.const_i32(NSEQ*NPHYS*BS*HKV*HD*2)); vr = b.buffer_rsrc(Vp, b.const_i32(NSEQ*NPHYS*BS*HKV*HD*2))
    btr = b.buffer_rsrc(BT, b.const_i32(NSEQ*NPHYS*4))
    def phys_key(kv, keytile):
        lblk = b.add(b.mul(sid, b.const_i32(NPHYS)), b.add(b.mul(kv, b.const_i32(BPT)), b.div(keytile, b.const_i32(BS))))
        pb = b.vec_extract(b.buffer_load_vN(btr, b.mul(lblk, b.const_i32(4)), b.const_i32(0), I32, 1), 0)
        return b.add(b.mul(pb, b.const_i32(BS)), b.mod(keytile, b.const_i32(BS)))
    sc = b.const_f32((1.0/math.sqrt(HD))*1.4426950408889634); ninf = b.const_f32(-1e30); zf = b.const_f32(0.0)
    def bperm(v):
        partner = b.mul(b.xor(lane, b.const_i32(32)), b.const_i32(4)); return b.bitcast(b.ds_bpermute(partner, b.bitcast(v, I32)), F32)
    iters = [("m", ninf), ("l", zf)] + [(f"a{nt}", at.zero_acc(b)) for nt in range(NDdim)]
    context_off = b.sub(klen, qlen)  # prefix already in KV cache; qlen!=klen (chunked prefill / decode)
    causal_t = b.div(b.add(b.add(context_off, qbase), b.const_i32(128 + BN - 1)), b.const_i32(BN))  # ceil((ctx+qmax+1)/BN)
    klen_t = b.div(b.add(klen, b.const_i32(BN-1)), b.const_i32(BN))
    kvend = b.select(b.cmp_lt(causal_t, klen_t), causal_t, klen_t)
    loop = b.scf_for_iter(b.const_i32(0), kvend, b.const_i32(1), iters, iv_name="kv")
    with loop as (kv, carry):
        m_old = carry[0]; l_old = carry[1]; accs = list(carry[2:])
        for c in range(8):
            lin = b.add(b.mul(tid, b.const_i32(64)), b.const_i32(c*8)); key = b.div(lin, b.const_i32(256)); hd = b.mod(lin, b.const_i32(256))
            pk = phys_key(kv, key); vsrc = b.mul(b.add(b.mul(b.add(b.mul(pk, b.const_i32(HKV)), kvh), b.const_i32(HD)), hd), b.const_i32(2))
            b.smem_store_vN(V_lds, [key, hd], b.buffer_load_vN(vr, vsrc, b.const_i32(0), BF16, 8), 8)
        b.sync()
        S_T = [at.zero_acc(b) for _ in range(NKEYT)]
        pk_kt = [phys_key(kv, b.add(b.const_i32(kt*32), ld.m_in_atom)) for kt in range(NKEYT)]
        for h in range(NK):
            koff = b.add(b.mul(b.const_i32(h), b.const_i32(K)), b.mul(ld.k_blk, b.const_i32(APL)))
            qi = b.add(b.mul(b.add(qstart, b.add(wq, ld.n_in_atom)), b.const_i32(H)), qhead)  # per-block packed Q offset
            q = b.buffer_load_vN(qr, b.mul(b.add(b.mul(qi, b.const_i32(HD)), koff), b.const_i32(2)), b.const_i32(0), BF16, BPL)
            for kt in range(NKEYT):
                kf = b.buffer_load_vN(kr, b.mul(b.add(b.mul(b.add(b.mul(pk_kt[kt], b.const_i32(HKV)), kvh), b.const_i32(HD)), koff), b.const_i32(2)), b.const_i32(0), BF16, APL)
                S_T[kt] = at.emit(b, kf, q, S_T[kt])
        Sm = [[None]*CPL for _ in range(NKEYT)]
        for kt in range(NKEYT):
            for i in range(CPL):
                rr, cc = at.lane_to_output(b, lane, i)
                key_g = b.add(b.add(b.mul(kv, b.const_i32(BN)), b.const_i32(kt*32)), rr)
                q_g = b.add(context_off, b.add(qbase, b.add(wq, cc)))  # absolute query pos = context_off + in-seq pos
                m_causal = b.cmp_gt(key_g, q_g); m_varlen = b.cmp_ge(key_g, klen)
                Sm[kt][i] = b.select(b.lor(m_causal, m_varlen), ninf, b.vec_extract(S_T[kt], i))
        local = ninf
        for kt in range(NKEYT):
            for i in range(CPL): local = b.fmax(local, b.fmul(Sm[kt][i], sc))
        m_new = b.fmax(m_old, b.fmax(local, bperm(local)))
        alpha = b.exp2(b.fsub(m_old, m_new))
        P = [[None]*CPL for _ in range(NKEYT)]; lsum = zf
        for kt in range(NKEYT):
            for i in range(CPL):
                p = b.exp2(b.fsub(b.fmul(Sm[kt][i], sc), m_new)); lsum = b.fadd(lsum, p); P[kt][i] = b.cast_f32_to(p, BF16)
        l_new = b.fadd(b.fmul(l_old, alpha), b.fadd(lsum, bperm(lsum)))
        Bp = [b.vec_pack([P[kk//4][(kk%4)*4+j] for j in range(BPL)], BF16) for kk in range(NKpv)]
        newaccs = []
        for nt in range(NDdim):
            pv = at.zero_acc(b)
            for kk in range(NKpv):
                va = b.vec_pack([b.vec_extract(b.smem_load_vN(V_lds, b.add(b.mul(b.const_i32(kk), b.const_i32(K)), b.add(b.mul(ld.k_blk, b.const_i32(APL)), b.const_i32(j))), b.add(b.mul(b.const_i32(nt), b.const_i32(32)), ld.m_in_atom), dtype=BF16, n=1), 0) for j in range(APL)], BF16)
                pv = at.emit(b, va, Bp[kk], pv)
            na = b.vec_pack([b.fma(b.vec_extract(accs[nt], i), alpha, b.vec_extract(pv, i)) for i in range(CPL)], F32)
            newaccs.append(na)
        b.scf_yield(m_new, l_new, *newaccs)
    m_f = loop.results[0]; l_f = loop.results[1]; accs_f = loop.results[2:]
    recip = b.rcp_fast(l_f)
    for nt in range(NDdim):
        for i in range(CPL):
            r, c = at.lane_to_output(b, lane, i); dim = b.add(b.mul(b.const_i32(nt), b.const_i32(32)), r)
            q_inseq = b.add(qbase, b.add(wq, c))
            oi = b.add(b.mul(b.add(qstart, b.add(wq, c)), b.const_i32(H)), qhead)
            val = b.fmul(b.vec_extract(accs_f[nt], i), recip)
            with b.scf_if(b.cmp_lt(q_inseq, qlen)):
                b.global_store(C, b.add(b.mul(oi, b.const_i32(HD)), dim), val, align=4)
    b.ret(); return b.kernel

if __name__ == "__main__":
    torch.manual_seed(0)
    qlens = [300, 1, 100]; klens = [300, 500, 500]   # ragged: seq0 prefill(q==k), seq1 decode(q=1,k=500), seq2 chunked(q=100,k=500)
    NSEQ = len(qlens); TOTALQ = sum(qlens)
    qblocks = [(q+127)//128 for q in qlens]; TOTALQB = sum(qblocks)
    globals()["NSEQ"] = NSEQ; globals()["TOTALQ"] = TOTALQ; globals()["TOTALQB"] = TOTALQB
    cu_q = [0]
    for q in qlens: cu_q.append(cu_q[-1]+q)
    SIDl, LQl = [], []
    for i, nb in enumerate(qblocks):
        for l in range(nb): SIDl.append(i); LQl.append(l)
    scale = 1.0/math.sqrt(HD)
    art = compile_kernel(build(), arch="gfx942"); open("/home/AMD/avirgoel/wk/eRag.hsaco", "wb").write(art.hsaco); print("BUILT", art.kernel_name, "NSEQ", NSEQ, "TOTALQ", TOTALQ, "TOTALQB", TOTALQB, flush=True)
    Kk = torch.randn(NSEQ*NPHYS, BS, HKV, HD, device="cuda", dtype=torch.bfloat16)*0.3
    Vv = torch.randn(NSEQ*NPHYS, BS, HKV, HD, device="cuda", dtype=torch.bfloat16)*0.3
    Qp = torch.randn(TOTALQ, H, HD, device="cuda", dtype=torch.bfloat16)*0.3
    bt = torch.arange(NSEQ*NPHYS, dtype=torch.int32, device="cuda")
    Klog = Kk.view(NSEQ*NPHYS, BS, HKV, HD)[bt.long()].reshape(NSEQ*NPHYS*BS, HKV, HD)
    Vlog = Vv.view(NSEQ*NPHYS, BS, HKV, HD)[bt.long()].reshape(NSEQ*NPHYS*BS, HKV, HD)
    Cc = torch.zeros(TOTALQ, H, HD, device="cuda", dtype=torch.float32)
    cuq = torch.tensor(cu_q, dtype=torch.int32, device="cuda")
    klt = torch.tensor(klens, dtype=torch.int32, device="cuda")
    sidt = torch.tensor(SIDl, dtype=torch.int32, device="cuda"); lqt = torch.tensor(LQl, dtype=torch.int32, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    L = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name,
        signature=SignatureBuilder().ptr("Q","bf16").ptr("K","bf16").ptr("V","bf16").ptr("C","f32").ptr("BT","i32").ptr("KL","i32").ptr("CUQ","i32").ptr("SID","i32").ptr("LQ","i32").build())
    def run(): L({"Q":Qp,"K":Kk,"V":Vv,"C":Cc,"BT":bt,"KL":klt,"CUQ":cuq,"SID":sidt,"LQ":lqt}, config=LaunchConfig(grid=(TOTALQB*H,1,1), block=(256,1,1), stream=stream))
    run(); torch.cuda.synchronize()
    # validate each seq: causal self-attention over its own K
    for i in range(NSEQ):
        q0, q1 = cu_q[i], cu_q[i+1]; ql = qlens[i]; kl = klens[i]; kbase = i*NPHYS*BS
        Qi = Qp[q0:q1].float(); Ki = Klog[kbase:kbase+kl].float(); Vi = Vlog[kbase:kbase+kl].float()
        qpos = (kl - ql) + torch.arange(ql, device="cuda"); kpos = torch.arange(kl, device="cuda"); mask = kpos[None,:] > qpos[:,None]  # abs q pos = context_off + i
        err = 0.0; err_sdpa = 0.0
        import torch.nn.functional as Fnn
        for h in range(H):
            kvh = h//GQAG
            Sm = (Qi[:,h,:]@Ki[:,kvh,:].T)*scale
            r = torch.softmax(Sm.masked_fill(mask, float("-inf")), -1)@Vi[:,kvh,:]
            err = max(err, (Cc[q0:q1,h,:]-r).abs().max().item())
            # INDEPENDENT: SDPA with EXPLICIT bottom-right causal mask (flash/vLLM/AITER convention; NOT is_causal which is top-left)
            br = (torch.arange(kl, device="cuda")[None,:] <= ((kl-ql)+torch.arange(ql, device="cuda"))[:,None])
            rs = Fnn.scaled_dot_product_attention(Qi[:,h,:].unsqueeze(0), Ki[:,kvh,:].unsqueeze(0), Vi[:,kvh,:].unsqueeze(0), attn_mask=br.unsqueeze(0), scale=scale).squeeze(0)
            err_sdpa = max(err_sdpa, (Cc[q0:q1,h,:]-rs).abs().max().item())
        print(f"seq{i} qlen={ql} klen={kl} manual={err:.2e} SDPA-indep={err_sdpa:.2e}", flush=True)
    ms = time_launches(run, warmup=10, iters=50, stream=stream)
    print(f"RAGGED {NSEQ} seqs, {TOTALQB} qblocks, time={ms*1e3:.1f}us", flush=True)
    synchronize_and_release(stream)
