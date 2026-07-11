import sys, math; sys.path.insert(0,"/home/AMD/avirgoel/wk")
from rocke.core.ir import BF16, F32, I32, IRBuilder, PtrType
from rocke.helpers.atoms import MfmaAtom
from rocke.helpers.mfma_gemm_inner import decode_mfma_lanes
from rocke import compile_kernel
from rocke.helpers import SignatureBuilder
from rocke.runtime import KernelLauncher, LaunchConfig
import torch
at=MfmaAtom.bf16_32x32x8(); APL,BPL,CPL,K=at.a_per_lane,at.b_per_lane,at.c_per_lane,at.k
HD=256; BN=64; NKEYT=BN//32; NK=HD//K; NDdim=HD//32; NKpv=BN//K; NUM_KV=128  # SQ=8192: 8192 keys
SQ=8192; NQB=SQ//128
BS=16; BPT=BN//BS  # paged: block_size=16, 4 blocks per 64-key tile
NPHYS=NUM_KV*BPT  # physical blocks per seq
H=16; HKV=2; GQAG=H//HKV  # GQA-16/2
def build():
    b=IRBuilder("e2eT3gqaw4"); b.kernel.attrs["max_workgroup_size"]=256  # 4 warps
    Q=b.param("Q",PtrType(BF16,"global"),noalias=True,readonly=True,align=16)
    Kp=b.param("K",PtrType(BF16,"global"),noalias=True,readonly=True,align=16)
    Vp=b.param("V",PtrType(BF16,"global"),noalias=True,readonly=True,align=16)
    C=b.param("C",PtrType(F32,"global"),noalias=True,writeonly=True,align=16)
    BT=b.param("BT",PtrType(I32,"global"),noalias=True,readonly=True,align=16)
    KL=b.param("KL",PtrType(I32,"global"),noalias=True,readonly=True,align=16)
    CUQ=b.param("CUQ",PtrType(I32,"global"),noalias=True,readonly=True,align=16)  # cu_seqlens_q  # varlen: seqlen_k
    tid=b.thread_id_x(); wid=b.div(tid,b.const_i32(64)); lane=b.mod(tid,b.const_i32(64)); ld=decode_mfma_lanes(b,at,lane)
    wq=b.mul(wid,b.const_i32(32))  # this warp's 32-query base
    bid=b.block_id_x(); qhead=b.mod(bid,b.const_i32(H)); kvh=b.div(qhead,b.const_i32(GQAG)); qblock=b.div(bid,b.const_i32(H))
    cuqr=b.buffer_rsrc(CUQ,b.const_i32(8)); cuq0=b.vec_extract(b.buffer_load_vN(cuqr,b.const_i32(0),b.const_i32(0),I32,1),0)
    qbase=b.mul(qblock,b.const_i32(128)); qstart=b.add(cuq0,qbase)
    V_lds=b.smem_alloc(BF16,[64,256],name_hint="Vlds")
    qr=b.buffer_rsrc(Q,b.const_i32(SQ*H*HD*2)); kr=b.buffer_rsrc(Kp,b.const_i32(NPHYS*BS*HKV*HD*2)); vr=b.buffer_rsrc(Vp,b.const_i32(NPHYS*BS*HKV*HD*2))
    btr=b.buffer_rsrc(BT,b.const_i32(NPHYS*4))
    def phys_key(kv,keytile):  # logical key-in-tile -> physical key via block table
        lblk=b.add(b.mul(kv,b.const_i32(BPT)),b.div(keytile,b.const_i32(BS)))
        pb=b.vec_extract(b.buffer_load_vN(btr,b.mul(lblk,b.const_i32(4)),b.const_i32(0),I32,1),0)
        return b.add(b.mul(pb,b.const_i32(BS)),b.mod(keytile,b.const_i32(BS)))
    sc=b.const_f32((1.0/math.sqrt(HD))*1.4426950408889634); ninf=b.const_f32(-1e30); zf=b.const_f32(0.0)
    # qbase computed per-block above
    klr=b.buffer_rsrc(KL,b.const_i32(4)); klen=b.vec_extract(b.buffer_load_vN(klr,b.const_i32(0),b.const_i32(0),I32,1),0)  # varlen KV bound
    def bperm(v):
        partner=b.mul(b.xor(lane,b.const_i32(32)),b.const_i32(4)); return b.bitcast(b.ds_bpermute(partner,b.bitcast(v,I32)),F32)
    iters=[("m",ninf),("l",zf)]+[(f"a{nt}",at.zero_acc(b)) for nt in range(NDdim)]
    kvend=b.mul(b.add(qblock,b.const_i32(1)),b.const_i32(2*BN//64))  # 2 tiles per q-block (causal bound)
    loop=b.scf_for_iter(b.const_i32(0),kvend,b.const_i32(1),iters,iv_name="kv")
    with loop as (kv, carry):
        m_old=carry[0]; l_old=carry[1]; accs=list(carry[2:])
        for c in range(8):  # cooperative PAGED V load: logical key -> physical via block table
            lin=b.add(b.mul(tid,b.const_i32(64)),b.const_i32(c*8)); key=b.div(lin,b.const_i32(256)); hd=b.mod(lin,b.const_i32(256))
            pk=phys_key(kv,key); vsrc=b.mul(b.add(b.mul(b.add(b.mul(pk,b.const_i32(HKV)),kvh),b.const_i32(HD)),hd),b.const_i32(2))
            b.smem_store_vN(V_lds,[key,hd],b.buffer_load_vN(vr,vsrc,b.const_i32(0),BF16,8),8)
        b.sync()
        S_T=[at.zero_acc(b) for _ in range(NKEYT)]
        pk_kt=[phys_key(kv,b.add(b.const_i32(kt*32),ld.m_in_atom)) for kt in range(NKEYT)]  # paged K phys keys
        for h in range(NK):
            koff=b.add(b.mul(b.const_i32(h),b.const_i32(K)),b.mul(ld.k_blk,b.const_i32(APL)))
            qi=b.add(b.mul(b.add(qstart,b.add(wq,ld.n_in_atom)),b.const_i32(H)),qhead)  # FIX: per-block Q offset
            q=b.buffer_load_vN(qr,b.mul(b.add(b.mul(qi,b.const_i32(HD)),koff),b.const_i32(2)),b.const_i32(0),BF16,BPL)
            for kt in range(NKEYT):
                kf=b.buffer_load_vN(kr,b.mul(b.add(b.mul(b.add(b.mul(pk_kt[kt],b.const_i32(HKV)),kvh),b.const_i32(HD)),koff),b.const_i32(2)),b.const_i32(0),BF16,APL)
                S_T[kt]=at.emit(b,kf,q,S_T[kt])
        Sm=[[None]*CPL for _ in range(NKEYT)]  # causal (key>query) + varlen (key>=seqlen_k) mask
        for kt in range(NKEYT):
            for i in range(CPL):
                rr,cc=at.lane_to_output(b,lane,i)
                key_g=b.add(b.add(b.mul(kv,b.const_i32(BN)),b.const_i32(kt*32)),rr)
                q_g=b.add(qbase,b.add(wq,cc))  # qbase = qblock*128 (in-seq pos)
                m_causal=b.cmp_gt(key_g,q_g); m_varlen=b.cmp_ge(key_g,klen)
                Sm[kt][i]=b.select(b.lor(m_causal,m_varlen),ninf,b.vec_extract(S_T[kt],i))
        local=ninf
        for kt in range(NKEYT):
            for i in range(CPL): local=b.fmax(local,b.fmul(Sm[kt][i],sc))
        m_new=b.fmax(m_old,b.fmax(local,bperm(local)))
        alpha=b.exp2(b.fsub(m_old,m_new))
        P=[[None]*CPL for _ in range(NKEYT)]; lsum=zf
        for kt in range(NKEYT):
            for i in range(CPL):
                p=b.exp2(b.fsub(b.fmul(Sm[kt][i],sc),m_new)); lsum=b.fadd(lsum,p); P[kt][i]=b.cast_f32_to(p,BF16)
        l_new=b.fadd(b.fmul(l_old,alpha),b.fadd(lsum,bperm(lsum)))
        Bp=[b.vec_pack([P[kk//4][(kk%4)*4+j] for j in range(BPL)],BF16) for kk in range(NKpv)]
        newaccs=[]
        for nt in range(NDdim):
            pv=at.zero_acc(b)
            for kk in range(NKpv):
                va=b.vec_pack([b.vec_extract(b.smem_load_vN(V_lds,b.add(b.mul(b.const_i32(kk),b.const_i32(K)),b.add(b.mul(ld.k_blk,b.const_i32(APL)),b.const_i32(j))),b.add(b.mul(b.const_i32(nt),b.const_i32(32)),ld.m_in_atom),dtype=BF16,n=1),0) for j in range(APL)],BF16)
                pv=at.emit(b,va,Bp[kk],pv)
            na=b.vec_pack([b.fma(b.vec_extract(accs[nt],i),alpha,b.vec_extract(pv,i)) for i in range(CPL)],F32)
            newaccs.append(na)
        b.scf_yield(m_new,l_new,*newaccs)
    m_f=loop.results[0]; l_f=loop.results[1]; accs_f=loop.results[2:]
    recip=b.rcp_fast(l_f)
    for nt in range(NDdim):
        for i in range(CPL):
            r,c=at.lane_to_output(b,lane,i); dim=b.add(b.mul(b.const_i32(nt),b.const_i32(32)),r)
            oi=b.add(b.mul(b.add(qstart,b.add(wq,c)),b.const_i32(H)),qhead)
            b.global_store(C,b.add(b.mul(oi,b.const_i32(HD)),dim),b.fmul(b.vec_extract(accs_f[nt],i),recip),align=4)
    b.ret(); return b.kernel
art=compile_kernel(build(),arch="gfx942"); open("/home/AMD/avirgoel/wk/e2eT3gqa.hsaco","wb").write(art.hsaco); print("BUILT",art.kernel_name,flush=True)
torch.manual_seed(0)
import os
Kk=torch.randn(NPHYS,BS,HKV,HD,device="cuda",dtype=torch.bfloat16)*0.3
Vv=torch.randn(NPHYS,BS,HKV,HD,device="cuda",dtype=torch.bfloat16)*0.3
Q=torch.randn(SQ,H,HD,device="cuda",dtype=torch.bfloat16)*0.3
bt=(torch.randperm(NPHYS) if os.environ.get("SCATTER") else torch.arange(NPHYS)).to(torch.int32).cuda()
Klog=Kk.view(NPHYS,BS,HKV,HD)[bt.long()].reshape(NPHYS*BS,HKV,HD)
Vlog=Vv.view(NPHYS,BS,HKV,HD)[bt.long()].reshape(NPHYS*BS,HKV,HD)
Cc=torch.zeros(SQ,H,HD,device="cuda",dtype=torch.float32)
cuq=torch.tensor([0,SQ],dtype=torch.int32,device="cuda"); klt=torch.tensor([SQ],dtype=torch.int32,device="cuda")
scale=1.0/math.sqrt(HD); stream=torch.cuda.current_stream().cuda_stream
L=KernelLauncher(hsaco=art.hsaco,kernel_name=art.kernel_name,signature=SignatureBuilder().ptr("Q","bf16").ptr("K","bf16").ptr("V","bf16").ptr("C","f32").ptr("BT","i32").ptr("KL","i32").ptr("CUQ","i32").build())
def ours(): L({"Q":Q,"K":Kk,"V":Vv,"C":Cc,"BT":bt,"KL":klt,"CUQ":cuq},config=LaunchConfig(grid=(NQB*H,1,1),block=(256,1,1),stream=stream))
ours(); torch.cuda.synchronize()
# validate a few query-blocks (causal, per-block)
kpos=torch.arange(SQ,device="cuda")
for qb in [0,16,31]:
    qpos=qb*128+torch.arange(128,device="cuda"); mask=kpos[None,:]>qpos[:,None]
    err=0.0
    for h in range(H):
        kvh=h//GQAG; Sm=(Q[qb*128:(qb+1)*128,h,:].float()@Klog[:,kvh,:].float().T)*scale
        r=torch.softmax(Sm.masked_fill(mask,float("-inf")),-1)@Vlog[:,kvh,:].float()
        err=max(err,(Cc[qb*128:(qb+1)*128,h,:]-r).abs().max().item())
    print(f"OURS qblock{qb} max_abs={err:.2e}",flush=True)
from rocke.runtime import time_launches, synchronize_and_release
ms=time_launches(ours,warmup=10,iters=50,stream=stream)
flop=2.0*(2.0*SQ*SQ*HD)*0.5*H  # causal ~half
print(f"OURS SQ={SQ} time={ms*1e3:.1f}us TF/s={flop/ms/1e12:.1f}",flush=True)
try:
    from aiter.ops.triton.attention.unified_attention import unified_attention
    out_ai=torch.empty(SQ,H,HD,device="cuda",dtype=torch.bfloat16)
    cu_q=torch.tensor([0,SQ],dtype=torch.int32,device="cuda"); kvl=torch.tensor([SQ],dtype=torch.int32,device="cuda")
    btA=bt.view(1,NPHYS)
    def ait(): unified_attention(Q,Kk,Vv,out_ai,cu_seqlens_q=cu_q,seqused_k=kvl,max_seqlen_q=SQ,max_seqlen_k=SQ,softmax_scale=scale,causal=True,window_size=(-1,-1),block_table=btA,softcap=0.0,q_descale=None,k_descale=None,v_descale=None)
    ait(); torch.cuda.synchronize()
    d=(Cc-out_ai.float()).abs().max().item()
    msa=time_launches(ait,warmup=10,iters=50,stream=stream)
    print(f"AITER SQ={SQ} time={msa*1e3:.1f}us TF/s={flop/msa/1e12:.1f}  OURS/AITER={ms/msa:.2f}x (ours-vs-aiter max_abs={d:.2e})",flush=True)
except Exception as e: print("AITER skipped:",str(e).splitlines()[0][:120],flush=True)
synchronize_and_release(stream)
