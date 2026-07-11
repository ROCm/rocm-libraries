import sys, math; sys.path.insert(0,"/home/AMD/avirgoel/wk")
from rocke.core.ir import BF16, F32, I32, IRBuilder, PtrType
from rocke.helpers.atoms import MfmaAtom
from rocke.helpers.mfma_gemm_inner import decode_mfma_lanes
from rocke import compile_kernel
from rocke.helpers import SignatureBuilder
from rocke.runtime import KernelLauncher, LaunchConfig
import torch
at=MfmaAtom.bf16_32x32x8(); APL,BPL,CPL,K=at.a_per_lane,at.b_per_lane,at.c_per_lane,at.k
HD=256; BN=64; NKEYT=BN//32; NK=HD//K; NDdim=HD//32; NKpv=BN//K; NUM_KV=8
def build():
    b=IRBuilder("e2eT3"); b.kernel.attrs["max_workgroup_size"]=64
    Q=b.param("Q",PtrType(BF16,"global"),noalias=True,readonly=True,align=16)
    Kp=b.param("K",PtrType(BF16,"global"),noalias=True,readonly=True,align=16)
    Vp=b.param("V",PtrType(BF16,"global"),noalias=True,readonly=True,align=16)
    C=b.param("C",PtrType(F32,"global"),noalias=True,writeonly=True,align=16)
    lane=b.thread_id_x(); ld=decode_mfma_lanes(b,at,lane)
    V_lds=b.smem_alloc(BF16,[64,256],name_hint="Vlds")
    qr=b.buffer_rsrc(Q,b.const_i32(32*HD*2)); kr=b.buffer_rsrc(Kp,b.const_i32(NUM_KV*BN*HD*2)); vr=b.buffer_rsrc(Vp,b.const_i32(NUM_KV*BN*HD*2))
    sc=b.const_f32((1.0/math.sqrt(HD))*1.4426950408889634); ninf=b.const_f32(-1e30); zf=b.const_f32(0.0)
    def bperm(v):
        partner=b.mul(b.xor(lane,b.const_i32(32)),b.const_i32(4)); return b.bitcast(b.ds_bpermute(partner,b.bitcast(v,I32)),F32)
    iters=[("m",ninf),("l",zf)]+[(f"a{nt}",at.zero_acc(b)) for nt in range(NDdim)]
    loop=b.scf_for_iter(b.const_i32(0),b.const_i32(NUM_KV),b.const_i32(1),iters,iv_name="kv")
    with loop as (kv, carry):
        m_old=carry[0]; l_old=carry[1]; accs=list(carry[2:])
        kelem=b.mul(kv,b.const_i32(BN)); vbyte=b.mul(kv,b.const_i32(BN*HD*2))
        for c in range(32):  # stage V tile [key,dim]
            lin=b.add(b.mul(lane,b.const_i32(256)),b.const_i32(c*8)); key=b.div(lin,b.const_i32(256)); hd=b.mod(lin,b.const_i32(256))
            b.smem_store_vN(V_lds,[key,hd],b.buffer_load_vN(vr,b.add(vbyte,b.mul(lin,b.const_i32(2))),b.const_i32(0),BF16,8),8)
        b.sync()
        S_T=[at.zero_acc(b) for _ in range(NKEYT)]
        for h in range(NK):
            koff=b.add(b.mul(b.const_i32(h),b.const_i32(K)),b.mul(ld.k_blk,b.const_i32(APL)))
            q=b.buffer_load_vN(qr,b.mul(b.add(b.mul(ld.n_in_atom,b.const_i32(HD)),koff),b.const_i32(2)),b.const_i32(0),BF16,BPL)
            for kt in range(NKEYT):
                key=b.add(kelem,b.add(b.mul(b.const_i32(kt),b.const_i32(32)),ld.m_in_atom))
                kf=b.buffer_load_vN(kr,b.mul(b.add(b.mul(key,b.const_i32(HD)),koff),b.const_i32(2)),b.const_i32(0),BF16,APL)
                S_T[kt]=at.emit(b,kf,q,S_T[kt])
        local=ninf
        for kt in range(NKEYT):
            for i in range(CPL): local=b.fmax(local,b.fmul(b.vec_extract(S_T[kt],i),sc))
        m_new=b.fmax(m_old,b.fmax(local,bperm(local)))
        alpha=b.exp2(b.fsub(m_old,m_new))
        P=[[None]*CPL for _ in range(NKEYT)]; lsum=zf
        for kt in range(NKEYT):
            for i in range(CPL):
                p=b.exp2(b.fsub(b.fmul(b.vec_extract(S_T[kt],i),sc),m_new)); lsum=b.fadd(lsum,p); P[kt][i]=b.cast_f32_to(p,BF16)
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
            b.global_store(C,b.add(b.mul(c,b.const_i32(HD)),dim),b.fmul(b.vec_extract(accs_f[nt],i),recip),align=4)
    b.ret(); return b.kernel
art=compile_kernel(build(),arch="gfx942"); open("/home/AMD/avirgoel/wk/e2eT3.hsaco","wb").write(art.hsaco); print("BUILT",art.kernel_name,flush=True)
torch.manual_seed(0)
Q=torch.randn(32,HD,device="cuda",dtype=torch.bfloat16)*0.3; Kk=torch.randn(NUM_KV*BN,HD,device="cuda",dtype=torch.bfloat16)*0.3; Vv=torch.randn(NUM_KV*BN,HD,device="cuda",dtype=torch.bfloat16)*0.3
Cc=torch.zeros(32,HD,device="cuda",dtype=torch.float32)
L=KernelLauncher(hsaco=art.hsaco,kernel_name=art.kernel_name,signature=SignatureBuilder().ptr("Q","bf16").ptr("K","bf16").ptr("V","bf16").ptr("C","f32").build())
L({"Q":Q,"K":Kk,"V":Vv,"C":Cc},config=LaunchConfig(grid=(1,1,1),block=(64,1,1),stream=torch.cuda.current_stream().cuda_stream)); torch.cuda.synchronize()
S=(Q.float()@Kk.float().T)*(1.0/math.sqrt(HD)); O=(torch.softmax(S,dim=-1)@Vv.float())
err=(Cc-O).abs().max().item(); print(f"NUM_KV={NUM_KV} max_abs={err:.4e} {'CORRECT' if err<1e-1 else 'WRONG'}",flush=True)
