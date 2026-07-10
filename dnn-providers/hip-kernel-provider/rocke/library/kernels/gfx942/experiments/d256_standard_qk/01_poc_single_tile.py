# Minimal end-to-end standard-QK attention: 1 q-tile[32] x 1 kv-tile[32], D=256, non-causal.
# QK -> LDS bridge -> row-wise softmax in LDS -> PV (V col-strided) -> /l epilogue. Verify vs torch.
import sys, torch, math
import kernels.common.attention_unified as au
au._RESOLVED_ATTENTION_ARCH="gfx942"
from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType
from rocke.helpers.mfma_gemm_inner import (decode_mfma_lanes, mfma_atom_for_dtype,
    load_a_row_major_contiguous, load_b_col_strided_scalars, mfma_k_loop, store_acc_to_global)
from rocke.helpers import SignatureBuilder
from rocke import compile_kernel
from rocke.runtime import KernelLauncher, LaunchConfig
SQ, SK, D = 32, 32, 256
def build():
    at=mfma_atom_for_dtype("f16",32,32,prefer_packed_k=False)  # 32x32x8
    b=IRBuilder("stdqk_attn"); b.kernel.attrs["max_workgroup_size"]=64
    Q=b.param("Q",PtrType(F16,"global"),noalias=True,readonly=True,align=16)
    K=b.param("K",PtrType(F16,"global"),noalias=True,readonly=True,align=16)
    V=b.param("V",PtrType(F16,"global"),noalias=True,readonly=True,align=16)
    O=b.param("O",PtrType(F32,"global"),noalias=True,writeonly=True,align=16)
    lane=b.thread_id_x(); ld=decode_mfma_lanes(b,at,lane)
    zb=b.const_i32(0)
    S_lds=b.smem_alloc(F32,[SQ,SK],name_hint="Slds")
    l_lds=b.smem_alloc(F32,[SQ],name_hint="llds")
    # --- QK: S=Q@K^T ---
    def la(bb,kt): return load_a_row_major_contiguous(bb,A=Q,atom=at,lane_decode=ld,m_tile_base=zb,k_tile_base=bb.mul(kt,bb.const_i32(at.k)),K=D)
    def lb(bb,kt): return load_a_row_major_contiguous(bb,A=K,atom=at,lane_decode=ld,m_tile_base=zb,k_tile_base=bb.mul(kt,bb.const_i32(at.k)),K=D)
    s_acc=mfma_k_loop(b,K=D,atom=at,load_a=la,load_b=lb,iv_name="ktqk",acc_name="accqk")
    scale=b.const_f32(1.0/math.sqrt(D))
    # write S (C-layout) to LDS[q,kv], scaled
    for i in range(at.c_per_lane):
        r,c=at.lane_to_output(b,lane,i)
        b.smem_store_vN(S_lds,[r,c], b.fmul(b.vec_extract(s_acc,i),scale), 1)
    b.sync()
    # --- row-wise softmax in LDS: lane%32 handles q-row ---
    row=b.mod(lane,b.const_i32(32))
    mx=b.const_f32(-1e30)
    for j in range(SK):
        v=b.vec_extract(b.smem_load_vN(S_lds,row,b.const_i32(j),dtype=F32,n=1),0)
        mx=b.fmax(v,mx)
    ssum=b.const_f32(0.0); ps=[]
    for j in range(SK):
        v=b.vec_extract(b.smem_load_vN(S_lds,row,b.const_i32(j),dtype=F32,n=1),0)
        p=b.exp2(b.fmul(b.fsub(v,mx),b.const_f32(1.4426950408889634)))
        ps.append(p); ssum=b.fadd(ssum,p)
    for j in range(SK):
        b.smem_store_vN(S_lds,[row,b.const_i32(j)], b.cast_f32_to(ps[j],F16) if False else ps[j], 1)
    b.smem_store_vN(l_lds,[row], ssum, 1)
    b.sync()
    # --- PV: acc[q,d]=P@V, 8 d-tiles, contraction kv=32 ---
    def lpa(bb,kt): # P A-operand from LDS[q,kv] (row=q=m_in_atom, contiguous kv)
        qн=bb.add(zb,ld.m_in_atom); kbase=bb.add(bb.mul(kt,bb.const_i32(at.k)),bb.mul(ld.k_blk,bb.const_i32(at.a_per_lane)))
        el=[bb.cast_f32_to(bb.vec_extract(bb.smem_load_vN(S_lds,qн,bb.add(kbase,bb.const_i32(j)),dtype=F32,n=1),0),F16) for j in range(at.a_per_lane)]
        return bb.vec_pack(el,F16)
    for nt in range(D//32):
        nbase=b.const_i32(nt*32)
        def lvb(bb,kt,nbase=nbase): return load_b_col_strided_scalars(bb,B=V,atom=at,lane_decode=ld,n_tile_base=nbase,k_tile_base=bb.mul(kt,bb.const_i32(at.k)),N=D)
        pv=mfma_k_loop(b,K=SK,atom=at,load_a=lpa,load_b=lvb,iv_name=f"ktpv{nt}",acc_name=f"accpv{nt}")
        # epilogue: /l per q-row, store to O[q, nt*32+..]
        for i in range(at.c_per_lane):
            r,c=at.lane_to_output(b,lane,i)
            lv=b.vec_extract(b.smem_load_vN(l_lds,r,dtype=F32,n=1),0)
            o=b.fdiv(b.vec_extract(pv,i),lv)
            addr=b.add(b.mul(r,b.const_i32(D)),b.add(nbase,c))
            b.global_store(O,addr,o,align=4)
    b.ret(); return b.kernel
art=compile_kernel(build(),arch="gfx942"); print("built",flush=True)
torch.manual_seed(0)
q=torch.randn(SQ,D,device="cuda",dtype=torch.float16);k=torch.randn(SK,D,device="cuda",dtype=torch.float16);v=torch.randn(SK,D,device="cuda",dtype=torch.float16)
o=torch.zeros(SQ,D,device="cuda",dtype=torch.float32)
sig=SignatureBuilder().ptr("Q","f16").ptr("K","f16").ptr("V","f16").ptr("O","f32").build()
L=KernelLauncher(hsaco=art.hsaco,kernel_name=art.kernel_name,signature=sig)
hs=torch.cuda.current_stream().cuda_stream
L({"Q":q,"K":k,"V":v,"O":o},config=LaunchConfig(grid=(1,1,1),block=(64,1,1),stream=hs))
torch.cuda.synchronize()
ref=torch.nn.functional.scaled_dot_product_attention(q.float()[None,None],k.float()[None,None],v.float()[None,None])[0,0]
err=(o-ref).abs().max().item()
print(f"ATTN max_abs_err={err:.4e} ref[0,:3]={ref[0,:3].tolist()} got[0,:3]={o[0,:3].tolist()}")
print("ATTN CORRECT" if err<0.05 else "ATTN WRONG")
