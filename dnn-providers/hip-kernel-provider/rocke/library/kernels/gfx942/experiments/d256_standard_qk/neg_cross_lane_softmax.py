# Cross-lane REGISTER softmax standard-QK attention: butterfly (ds_swizzle_xor) row-reduce in registers.
# m/l/corr carried in regs (scf_for_iter: 128 acc + 16 m + 16 l). Only 1 LDS write (P-reshape C->A) + PV.
# multi-head grid, causal, D=256, f16. Correctness + TF/s.
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
SQ=int(sys.argv[1]) if len(sys.argv)>1 else 4096
H=int(sys.argv[2]) if len(sys.argv)>2 else 16
SK=SQ; NQB=SQ//32
def build():
    at=mfma_atom_for_dtype("f16",32,32,prefer_packed_k=False); ND=D//32; CPL=at.c_per_lane
    b=IRBuilder("stdqk_xlane"); b.kernel.attrs["max_workgroup_size"]=64
    Q=b.param("Q",PtrType(F16,"global"),noalias=True,readonly=True,align=16)
    K=b.param("K",PtrType(F16,"global"),noalias=True,readonly=True,align=16)
    V=b.param("V",PtrType(F16,"global"),noalias=True,readonly=True,align=16)
    O=b.param("O",PtrType(F32,"global"),noalias=True,writeonly=True,align=16)
    lane=b.thread_id_x(); ld=decode_mfma_lanes(b,at,lane); zb=b.const_i32(0)
    qblk=b.block_id_x(); qbase=b.mul(qblk,b.const_i32(32))
    head=b.block_id_y(); hoq=b.mul(head,b.const_i32(SQ)); hok=b.mul(head,b.const_i32(SK))
    S_lds=b.smem_alloc(F32,[32,32],name_hint="Slds")
    scale=b.const_f32(1.0/math.sqrt(D)); l2e=b.const_f32(LOG2E); ninf=b.const_f32(-1e30)
    # per-element (r_i,c) precompute
    rc=[at.lane_to_output(b,lane,i) for i in range(CPL)]  # (row_i, col) ; col == lane%32 for all i
    def bfly(v,is_max):
        for mask in (1,2,4,8,16):
            sw=b.bitcast(b.ds_swizzle_xor(b.bitcast(v,I32),mask),F32)
            v=b.fmax(v,sw) if is_max else b.fadd(v,sw)
        return v
    iter_args=[(f"acc{n}",b.const_f32(0.0)) for n in range(ND*CPL)]+\
              [(f"m{n}",ninf) for n in range(CPL)]+[(f"l{n}",b.const_f32(0.0)) for n in range(CPL)]
    kvloop=b.scf_for_iter(zb,b.add(qblk,b.const_i32(1)),b.const_i32(1),iter_args,iv_name="kv")
    with kvloop as (kv,carry):
        acc=carry[:ND*CPL]; m_old=carry[ND*CPL:ND*CPL+CPL]; l_old=carry[ND*CPL+CPL:]
        kvb=b.mul(kv,b.const_i32(32))
        def la(bb,kt): return load_a_row_major_contiguous(bb,A=Q,atom=at,lane_decode=ld,m_tile_base=bb.add(qbase,hoq),k_tile_base=bb.mul(kt,bb.const_i32(at.k)),K=D)
        def lb(bb,kt): return load_a_row_major_contiguous(bb,A=K,atom=at,lane_decode=ld,m_tile_base=bb.add(kvb,hok),k_tile_base=bb.mul(kt,bb.const_i32(at.k)),K=D)
        s_acc=mfma_k_loop(b,K=D,atom=at,load_a=la,load_b=lb,iv_name="ktqk",acc_name="accqk")
        # scale + causal mask (register)
        s=[None]*CPL
        for i in range(CPL):
            r_i,c=rc[i]; qi=b.add(qbase,r_i); kvi=b.add(kvb,c)
            v=b.fmul(b.vec_extract(s_acc,i),scale)
            s[i]=b.select(b.cmp_gt(kvi,qi),ninf,v)
        # register row-max (butterfly over 32 kv-lanes)
        rmax=[bfly(s[i],True) for i in range(CPL)]
        m_new=[b.fmax(m_old[i],rmax[i]) for i in range(CPL)]
        corr=[b.exp2(b.fmul(b.fsub(m_old[i],m_new[i]),l2e)) for i in range(CPL)]
        p=[b.exp2(b.fmul(b.fsub(s[i],m_new[i]),l2e)) for i in range(CPL)]
        rsum=[bfly(p[i],False) for i in range(CPL)]
        l_new=[b.fadd(b.fmul(l_old[i],corr[i]),rsum[i]) for i in range(CPL)]
        # write P to S_lds (C-layout) for reshape C->A
        for i in range(CPL):
            r_i,c=rc[i]; b.smem_store_vN(S_lds,[r_i,c],p[i],1)
        b.sync()
        new_acc=[None]*(ND*CPL)
        for nt in range(ND):
            nbase=b.const_i32(nt*32)
            def lpa(bb,kt):
                qn=bb.add(zb,ld.m_in_atom); kbase=bb.add(bb.mul(kt,bb.const_i32(at.k)),bb.mul(ld.k_blk,bb.const_i32(at.a_per_lane)))
                el=[bb.cast_f32_to(bb.vec_extract(bb.smem_load_vN(S_lds,qn,bb.add(kbase,bb.const_i32(j)),dtype=F32,n=1),0),F16) for j in range(at.a_per_lane)]
                return bb.vec_pack(el,F16)
            def lvb(bb,kt,nbase=nbase): return load_b_col_strided_scalars(bb,B=V,atom=at,lane_decode=ld,n_tile_base=nbase,k_tile_base=bb.add(bb.add(hok,kvb),bb.mul(kt,bb.const_i32(at.k))),N=D)
            pv=mfma_k_loop(b,K=32,atom=at,load_a=lpa,load_b=lvb,iv_name=f"ktpv{nt}",acc_name=f"accpv{nt}")
            for i in range(CPL):
                new_acc[nt*CPL+i]=b.fadd(b.fmul(acc[nt*CPL+i],corr[i]),b.vec_extract(pv,i))
        b.sync()
        b.scf_yield(*new_acc,*m_new,*l_new)
    final=kvloop.results; accf=final[:ND*CPL]; lf=final[ND*CPL+CPL:]
    for nt in range(ND):
        nbase=b.const_i32(nt*32)
        for i in range(CPL):
            r_i,c=rc[i]; addr=b.add(b.mul(b.add(b.add(hoq,qbase),r_i),b.const_i32(D)),b.add(nbase,c))
            b.global_store(O,addr,b.fdiv(accf[nt*CPL+i],lf[i]),align=4)
    b.ret(); return b.kernel
art=compile_kernel(build(),arch="gfx942"); print("built",flush=True)
torch.manual_seed(0)
q=torch.randn(H,SQ,D,device="cuda",dtype=torch.float16);k=torch.randn(H,SK,D,device="cuda",dtype=torch.float16);v=torch.randn(H,SK,D,device="cuda",dtype=torch.float16)
o=torch.zeros(H,SQ,D,device="cuda",dtype=torch.float32)
sig=SignatureBuilder().ptr("Q","f16").ptr("K","f16").ptr("V","f16").ptr("O","f32").build()
L=KernelLauncher(hsaco=art.hsaco,kernel_name=art.kernel_name,signature=sig)
hs=torch.cuda.current_stream().cuda_stream; cfg=LaunchConfig(grid=(NQB,H,1),block=(64,1,1),stream=hs)
L({"Q":q,"K":k,"V":v,"O":o},config=cfg); torch.cuda.synchronize()
ref=torch.nn.functional.scaled_dot_product_attention(q.float()[None],k.float()[None],v.float()[None],is_causal=True)[0]
err=(o-ref).abs().max().item()
print(f"XLANE SQ={SQ} H={H} blocks={NQB*H} max_abs_err={err:.4e}  {'CORRECT' if err<0.05 else 'WRONG'}")
def once(): L({"Q":q,"K":k,"V":v,"O":o},config=cfg)
ms=time_launches(once,warmup=10,iters=50,stream=hs); synchronize_and_release(hs)
flop=2.0*(2.0*SQ*SK*D)*0.5*H
print(f"time={ms*1e3:.1f}us  TF/s(causal)={flop/(ms*1e-3)/1e12:.2f}")
