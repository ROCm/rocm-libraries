# BLOCK_N=64: process 64 kv/iter (2 QK N-fragments, softmax over 64, PV contraction 64).
# Halves #kv-iters -> halves per-tile rescale + syncs. bf16 + GQA 16/2 + causal. +b.fma rescale.
import sys, torch, math
import kernels.common.attention_unified as au
au._RESOLVED_ATTENTION_ARCH="gfx942"
from rocke.core.ir import F16, BF16, F32, I32, IRBuilder, PtrType
from rocke.helpers.atoms import MfmaAtom
from rocke.helpers.mfma_gemm_inner import (decode_mfma_lanes, load_a_row_major_contiguous, load_b_col_strided_scalars, mfma_k_loop)
from rocke.helpers import SignatureBuilder
from rocke import compile_kernel
from rocke.runtime import KernelLauncher, LaunchConfig, time_launches, synchronize_and_release
D=256; LOG2E=1.4426950408889634; BN=64
SQ=int(sys.argv[1]) if len(sys.argv)>1 else 512
SK=SQ; NQB=SQ//32; H=int(sys.argv[2]) if len(sys.argv)>2 else 16; HKV=2; GQ=H//HKV
def build():
    at=MfmaAtom.bf16_32x32x8(); ND=D//32; CPL=at.c_per_lane; NF=BN//32
    b=IRBuilder("stdqk_bn64"); b.kernel.attrs["max_workgroup_size"]=64
    Q=b.param("Q",PtrType(BF16,"global"),noalias=True,readonly=True,align=16)
    K=b.param("K",PtrType(BF16,"global"),noalias=True,readonly=True,align=16)
    V=b.param("V",PtrType(BF16,"global"),noalias=True,readonly=True,align=16)
    O=b.param("O",PtrType(F32,"global"),noalias=True,writeonly=True,align=16)
    lane=b.thread_id_x(); ld=decode_mfma_lanes(b,at,lane); zb=b.const_i32(0)
    qblk=b.block_id_x(); qbase=b.mul(qblk,b.const_i32(32))
    head=b.block_id_y(); hoq=b.mul(head,b.const_i32(SQ)); hok=b.mul(b.div(head,b.const_i32(GQ)),b.const_i32(SK))
    S_lds=b.smem_alloc(F32,[32,BN+1],name_hint="Slds")  # [32,65] padded
    m_lds=b.smem_alloc(F32,[32],name_hint="mlds"); l_lds=b.smem_alloc(F32,[32],name_hint="llds"); c_lds=b.smem_alloc(F32,[32],name_hint="clds")
    scale=b.const_f32(1.0/math.sqrt(D)); l2e=b.const_f32(LOG2E); ninf=b.const_f32(-1e30)
    row=b.mod(lane,b.const_i32(32)); qabs=b.add(qbase,row)
    b.smem_store_vN(m_lds,[row],ninf,1); b.smem_store_vN(l_lds,[row],b.const_f32(0.0),1); b.sync()
    iter_args=[(f"acc{n}",b.const_f32(0.0)) for n in range(ND*CPL)]
    upper=b.add(b.div(b.add(qbase,b.const_i32(31)),b.const_i32(BN)),b.const_i32(1))  # causal 64-tiles
    kvloop=b.scf_for_iter(zb,upper,b.const_i32(1),iter_args,iv_name="kv")
    with kvloop as (kv,carry):
        kvb=b.mul(kv,b.const_i32(BN))
        def la(bb,kt): return load_a_row_major_contiguous(bb,A=Q,atom=at,lane_decode=ld,m_tile_base=bb.add(qbase,hoq),k_tile_base=bb.mul(kt,bb.const_i32(at.k)),K=D)
        for nf in range(NF):
            kvbf=b.add(kvb,b.const_i32(nf*32))
            def lb(bb,kt,kvbf=kvbf): return load_a_row_major_contiguous(bb,A=K,atom=at,lane_decode=ld,m_tile_base=bb.add(kvbf,hok),k_tile_base=bb.mul(kt,bb.const_i32(at.k)),K=D)
            s_acc=mfma_k_loop(b,K=D,atom=at,load_a=la,load_b=lb,iv_name=f"ktqk{nf}",acc_name=f"accqk{nf}")
            for i in range(CPL):
                r,c=at.lane_to_output(b,lane,i); col=b.add(c,b.const_i32(nf*32))
                b.smem_store_vN(S_lds,[r,col],b.fmul(b.vec_extract(s_acc,i),scale),1)
        b.sync()
        with b.scf_if(b.cmp_lt(lane,b.const_i32(32))):
            m_old=b.vec_extract(b.smem_load_vN(m_lds,row,dtype=F32,n=1),0); l_old=b.vec_extract(b.smem_load_vN(l_lds,row,dtype=F32,n=1),0)
            mx=m_old
            for jj in range(BN//4):
                v4=b.smem_load_vN(S_lds,row,b.const_i32(4*jj),dtype=F32,n=4)
                for e in range(4):
                    j=4*jj+e; v=b.select(b.cmp_gt(b.add(kvb,b.const_i32(j)),qabs),ninf,b.vec_extract(v4,e)); mx=b.fmax(v,mx)
            corr=b.exp2(b.fmul(b.fsub(m_old,mx),l2e)); ssum=b.const_f32(0.0)
            for jj in range(BN//4):
                v4=b.smem_load_vN(S_lds,row,b.const_i32(4*jj),dtype=F32,n=4); ps=[]
                for e in range(4):
                    j=4*jj+e; v=b.select(b.cmp_gt(b.add(kvb,b.const_i32(j)),qabs),ninf,b.vec_extract(v4,e))
                    p=b.exp2(b.fmul(b.fsub(v,mx),l2e)); ps.append(p); ssum=b.fadd(ssum,p)
                b.smem_store_vN(S_lds,[row,b.const_i32(4*jj)],b.vec_pack(ps,F32),4)
            b.smem_store_vN(m_lds,[row],mx,1); b.smem_store_vN(l_lds,[row],b.fadd(b.fmul(l_old,corr),ssum),1); b.smem_store_vN(c_lds,[row],corr,1)
        b.sync()
        new_acc=[None]*(ND*CPL)
        for nt in range(ND):
            nbase=b.const_i32(nt*32)
            def lpa(bb,kt):
                qn=bb.add(zb,ld.m_in_atom); kbase=bb.add(bb.mul(kt,bb.const_i32(at.k)),bb.mul(ld.k_blk,bb.const_i32(at.a_per_lane)))
                el=[bb.cast_f32_to(bb.vec_extract(bb.smem_load_vN(S_lds,qn,bb.add(kbase,bb.const_i32(j)),dtype=F32,n=1),0),BF16) for j in range(at.a_per_lane)]
                return bb.vec_pack(el,BF16)
            def lvb(bb,kt,nbase=nbase): return load_b_col_strided_scalars(bb,B=V,atom=at,lane_decode=ld,n_tile_base=nbase,k_tile_base=bb.add(bb.add(hok,kvb),bb.mul(kt,bb.const_i32(at.k))),N=D)
            pv=mfma_k_loop(b,K=BN,atom=at,load_a=lpa,load_b=lvb,iv_name=f"ktpv{nt}",acc_name=f"accpv{nt}")
            for i in range(CPL):
                r,c=at.lane_to_output(b,lane,i); cr=b.vec_extract(b.smem_load_vN(c_lds,r,dtype=F32,n=1),0)
                new_acc[nt*CPL+i]=b.fma(carry[nt*CPL+i],cr,b.vec_extract(pv,i))
        b.scf_yield(*new_acc)
    final=kvloop.results
    for nt in range(ND):
        nbase=b.const_i32(nt*32)
        for i in range(CPL):
            r,c=at.lane_to_output(b,lane,i); lv=b.vec_extract(b.smem_load_vN(l_lds,r,dtype=F32,n=1),0)
            addr=b.add(b.mul(b.add(b.add(hoq,qbase),r),b.const_i32(D)),b.add(nbase,c)); b.global_store(O,addr,b.fdiv(final[nt*CPL+i],lv),align=4)
    b.ret(); return b.kernel
art=compile_kernel(build(),arch="gfx942"); print("built",flush=True)
torch.manual_seed(0)
q=torch.randn(H,SQ,D,device="cuda",dtype=torch.bfloat16);k=torch.randn(HKV,SK,D,device="cuda",dtype=torch.bfloat16);v=torch.randn(HKV,SK,D,device="cuda",dtype=torch.bfloat16)
o=torch.zeros(H,SQ,D,device="cuda",dtype=torch.float32)
sig=SignatureBuilder().ptr("Q","bf16").ptr("K","bf16").ptr("V","bf16").ptr("O","f32").build()
L=KernelLauncher(hsaco=art.hsaco,kernel_name=art.kernel_name,signature=sig)
hs=torch.cuda.current_stream().cuda_stream; cfg=LaunchConfig(grid=(NQB,H,1),block=(64,1,1),stream=hs)
L({"Q":q,"K":k,"V":v,"O":o},config=cfg); torch.cuda.synchronize()
ke=k.repeat_interleave(GQ,dim=0); ve=v.repeat_interleave(GQ,dim=0)
ref=torch.nn.functional.scaled_dot_product_attention(q.float()[None],ke.float()[None],ve.float()[None],is_causal=True)[0]
err=(o-ref).abs().max().item()
print(f"BN64-CAUSAL SQ={SQ} H={H} blocks={NQB*H} max_abs_err={err:.4e}  {'CORRECT' if err<0.2 else 'WRONG'}")
def once(): L({"Q":q,"K":k,"V":v,"O":o},config=cfg)
ms=time_launches(once,warmup=10,iters=50,stream=hs); synchronize_and_release(hs)
flop=2.0*(2.0*SQ*SK*D)*0.5*H
print(f"time={ms*1e3:.1f}us  TF/s(causal)={flop/(ms*1e-3)/1e12:.2f}")
try:
    def aot(): torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None],is_causal=True,enable_gqa=True)
    aot(); torch.cuda.synchronize(); ms_a=time_launches(aot,warmup=10,iters=50,stream=hs); synchronize_and_release(hs)
    print(f"AOTriton time={ms_a*1e3:.1f}us TF/s={flop/(ms_a*1e-3)/1e12:.2f}  ratio={ms_a/ms:.3f}x ({'WIN' if ms<ms_a else 'trail'})")
except Exception as e: print("aot failed",str(e)[:60])
