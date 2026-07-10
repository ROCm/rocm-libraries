# Runtime kv-loop standard-QK attention (scf_for, acc in LDS): constant kernel size, scales to any Sq.
# grid over q-blocks[32]; causal work-skip (upper=qblk+1); D=256, f16, single head. Correctness + TF/s.
import sys, torch, math
import kernels.common.attention_unified as au
au._RESOLVED_ATTENTION_ARCH="gfx942"
from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType
from rocke.helpers.mfma_gemm_inner import (decode_mfma_lanes, mfma_atom_for_dtype,
    load_a_row_major_contiguous, load_b_col_strided_scalars, mfma_k_loop)
from rocke.helpers import SignatureBuilder
from rocke import compile_kernel
from rocke.runtime import KernelLauncher, LaunchConfig, time_launches, synchronize_and_release
D=256; LOG2E=1.4426950408889634
SQ=int(sys.argv[1]) if len(sys.argv)>1 else 512
SK=SQ; NQB=SQ//32
def build():
    at=mfma_atom_for_dtype("f16",32,32,prefer_packed_k=False); ND=D//32
    b=IRBuilder("stdqk_loop"); b.kernel.attrs["max_workgroup_size"]=64
    Q=b.param("Q",PtrType(F16,"global"),noalias=True,readonly=True,align=16)
    K=b.param("K",PtrType(F16,"global"),noalias=True,readonly=True,align=16)
    V=b.param("V",PtrType(F16,"global"),noalias=True,readonly=True,align=16)
    O=b.param("O",PtrType(F32,"global"),noalias=True,writeonly=True,align=16)
    lane=b.thread_id_x(); ld=decode_mfma_lanes(b,at,lane); zb=b.const_i32(0)
    qblk=b.block_id_x(); qbase=b.mul(qblk,b.const_i32(32))
    S_lds=b.smem_alloc(F32,[32,32],name_hint="Slds")
    acc_lds=b.smem_alloc(F32,[32*D],name_hint="acclds")
    m_lds=b.smem_alloc(F32,[32],name_hint="mlds"); l_lds=b.smem_alloc(F32,[32],name_hint="llds"); c_lds=b.smem_alloc(F32,[32],name_hint="clds")
    scale=b.const_f32(1.0/math.sqrt(D)); l2e=b.const_f32(LOG2E); ninf=b.const_f32(-1e30)
    row=b.mod(lane,b.const_i32(32)); qabs=b.add(qbase,row)
    b.smem_store_vN(m_lds,[row],ninf,1); b.smem_store_vN(l_lds,[row],b.const_f32(0.0),1)
    with b.scf_for(lane,b.const_i32(32*D),b.const_i32(64),iv_name="zi") as zi:
        b.smem_store_vN(acc_lds,[zi],b.const_f32(0.0),1)
    b.sync()
    with b.scf_for(zb,b.add(qblk,b.const_i32(1)),b.const_i32(1),iv_name="kv") as kv:
        kvb=b.mul(kv,b.const_i32(32))
        def la(bb,kt): return load_a_row_major_contiguous(bb,A=Q,atom=at,lane_decode=ld,m_tile_base=qbase,k_tile_base=bb.mul(kt,bb.const_i32(at.k)),K=D)
        def lb(bb,kt): return load_a_row_major_contiguous(bb,A=K,atom=at,lane_decode=ld,m_tile_base=kvb,k_tile_base=bb.mul(kt,bb.const_i32(at.k)),K=D)
        s_acc=mfma_k_loop(b,K=D,atom=at,load_a=la,load_b=lb,iv_name="ktqk",acc_name="accqk")
        for i in range(at.c_per_lane):
            r,c=at.lane_to_output(b,lane,i); b.smem_store_vN(S_lds,[r,c],b.fmul(b.vec_extract(s_acc,i),scale),1)
        b.sync()
        m_old=b.vec_extract(b.smem_load_vN(m_lds,row,dtype=F32,n=1),0); l_old=b.vec_extract(b.smem_load_vN(l_lds,row,dtype=F32,n=1),0)
        mx=m_old
        for j in range(32):
            kvabs=b.add(kvb,b.const_i32(j)); v=b.vec_extract(b.smem_load_vN(S_lds,row,b.const_i32(j),dtype=F32,n=1),0)
            v=b.select(b.cmp_gt(kvabs,qabs),ninf,v); mx=b.fmax(v,mx)
        corr=b.exp2(b.fmul(b.fsub(m_old,mx),l2e)); ssum=b.const_f32(0.0)
        for j in range(32):
            kvabs=b.add(kvb,b.const_i32(j)); v=b.vec_extract(b.smem_load_vN(S_lds,row,b.const_i32(j),dtype=F32,n=1),0)
            v=b.select(b.cmp_gt(kvabs,qabs),ninf,v); p=b.exp2(b.fmul(b.fsub(v,mx),l2e))
            b.smem_store_vN(S_lds,[row,b.const_i32(j)],p,1); ssum=b.fadd(ssum,p)
        b.smem_store_vN(m_lds,[row],mx,1); b.smem_store_vN(l_lds,[row],b.fadd(b.fmul(l_old,corr),ssum),1); b.smem_store_vN(c_lds,[row],corr,1)
        b.sync()
        for nt in range(ND):
            nbase=b.const_i32(nt*32)
            def lpa(bb,kt):
                qn=bb.add(zb,ld.m_in_atom); kbase=bb.add(bb.mul(kt,bb.const_i32(at.k)),bb.mul(ld.k_blk,bb.const_i32(at.a_per_lane)))
                el=[bb.cast_f32_to(bb.vec_extract(bb.smem_load_vN(S_lds,qn,bb.add(kbase,bb.const_i32(j)),dtype=F32,n=1),0),F16) for j in range(at.a_per_lane)]
                return bb.vec_pack(el,F16)
            def lvb(bb,kt,nbase=nbase): return load_b_col_strided_scalars(bb,B=V,atom=at,lane_decode=ld,n_tile_base=nbase,k_tile_base=bb.add(kvb,bb.mul(kt,bb.const_i32(at.k))),N=D)
            pv=mfma_k_loop(b,K=32,atom=at,load_a=lpa,load_b=lvb,iv_name=f"ktpv{nt}",acc_name=f"accpv{nt}")
            for i in range(at.c_per_lane):
                r,c=at.lane_to_output(b,lane,i); cr=b.vec_extract(b.smem_load_vN(c_lds,r,dtype=F32,n=1),0)
                idx=b.add(b.mul(r,b.const_i32(D)),b.add(nbase,c)); old=b.vec_extract(b.smem_load_vN(acc_lds,idx,dtype=F32,n=1),0)
                b.smem_store_vN(acc_lds,[idx],b.fadd(b.fmul(old,cr),b.vec_extract(pv,i)),1)
        b.sync()
    for nt in range(ND):
        nbase=b.const_i32(nt*32)
        for i in range(at.c_per_lane):
            r,c=at.lane_to_output(b,lane,i); lv=b.vec_extract(b.smem_load_vN(l_lds,r,dtype=F32,n=1),0)
            idx=b.add(b.mul(r,b.const_i32(D)),b.add(nbase,c)); ov=b.vec_extract(b.smem_load_vN(acc_lds,idx,dtype=F32,n=1),0)
            addr=b.add(b.mul(b.add(qbase,r),b.const_i32(D)),b.add(nbase,c)); b.global_store(O,addr,b.fdiv(ov,lv),align=4)
    b.ret(); return b.kernel
art=compile_kernel(build(),arch="gfx942"); print("built",flush=True)
torch.manual_seed(0)
q=torch.randn(SQ,D,device="cuda",dtype=torch.float16);k=torch.randn(SK,D,device="cuda",dtype=torch.float16);v=torch.randn(SK,D,device="cuda",dtype=torch.float16)
o=torch.zeros(SQ,D,device="cuda",dtype=torch.float32)
sig=SignatureBuilder().ptr("Q","f16").ptr("K","f16").ptr("V","f16").ptr("O","f32").build()
L=KernelLauncher(hsaco=art.hsaco,kernel_name=art.kernel_name,signature=sig)
hs=torch.cuda.current_stream().cuda_stream; cfg=LaunchConfig(grid=(NQB,1,1),block=(64,1,1),stream=hs)
L({"Q":q,"K":k,"V":v,"O":o},config=cfg); torch.cuda.synchronize()
ref=torch.nn.functional.scaled_dot_product_attention(q.float()[None,None],k.float()[None,None],v.float()[None,None],is_causal=True)[0,0]
err=(o-ref).abs().max().item()
print(f"LOOP-CAUSAL SQ={SQ} NQB={NQB} max_abs_err={err:.4e}  {'CORRECT' if err<0.05 else 'WRONG'}")
def once(): L({"Q":q,"K":k,"V":v,"O":o},config=cfg)
ms=time_launches(once,warmup=10,iters=50,stream=hs); synchronize_and_release(hs)
flop=2.0*(2.0*SQ*SK*D)*0.5  # causal ~half
print(f"time={ms*1e3:.1f}us  TF/s(causal)={flop/(ms*1e-3)/1e12:.2f}")
