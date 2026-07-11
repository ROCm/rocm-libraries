import sys, torch
sys.path.insert(0,"/home/AMD/avirgoel/wk")
import gqa_kernel_mc as G
from rocke.runtime import LaunchConfig, time_launches, synchronize_and_release
H,HKV,NPHYS,BS,HD,NUM_KV,BN,NSEQ=G.H,G.HKV,G.NPHYS,G.BS,G.HD,G.NUM_KV,G.BN,G.NSEQ
SQ=128; SK=NUM_KV*BN; scale=HD**-0.5; stream=torch.cuda.current_stream().cuda_stream
torch.manual_seed(1)
key_cache=torch.randn(NSEQ*NPHYS,BS,HKV,HD,device="cuda",dtype=torch.bfloat16)*0.3
value_cache=torch.randn(NSEQ*NPHYS,BS,HKV,HD,device="cuda",dtype=torch.bfloat16)*0.3
query=torch.randn(NSEQ*SQ,H,HD,device="cuda",dtype=torch.bfloat16)*0.3
bt=torch.stack([torch.randperm(NPHYS)+s*NPHYS for s in range(NSEQ)]).to(torch.int32).cuda()  # [NSEQ,NPHYS] distinct
Cc=torch.zeros(NSEQ*SQ,H,HD,device="cuda",dtype=torch.float32); klt=torch.tensor([SK],dtype=torch.int32,device="cuda")
kc4=key_cache.view(NSEQ*NPHYS,BS,HKV,HD); vc4=value_cache.view(NSEQ*NPHYS,BS,HKV,HD)
def ref_seq(sq):
    b0=bt[sq].long(); Kl=kc4[b0].reshape(NPHYS*BS,HKV,HD); Vl=vc4[b0].reshape(NPHYS*BS,HKV,HD)
    qpos=(SK-SQ)+torch.arange(SQ,device="cuda"); kpos=torch.arange(SK,device="cuda"); mask=kpos[None,:]>qpos[:,None]
    o=torch.zeros(SQ,H,HD,device="cuda")
    for h in range(H):
        kv=h//(H//HKV); sc=(query[sq*SQ:(sq+1)*SQ,h,:].float()@Kl[:,kv,:].float().T)*scale
        o[:,h,:]=torch.softmax(sc.masked_fill(mask,float("-inf")),-1)@Vl[:,kv,:].float()
    return o
def ours(): G.L({"Q":query,"K":key_cache,"V":value_cache,"C":Cc,"BT":bt.reshape(-1),"KL":klt},config=LaunchConfig(grid=(H*NSEQ,1,1),block=(256,1,1),stream=stream))
ours(); torch.cuda.synchronize()
print("Cc[seq0,q0,:4]=",Cc[0,0,:4].tolist(),flush=True)
print("Cc[seq1,q0,:4]=",Cc[128,0,:4].tolist(),flush=True)
print("Cc[seq63,q0,:4]=",Cc[63*128,0,:4].tolist(),flush=True)
print("Cc[seq1] absmax=",Cc[128:256].abs().max().item(),flush=True)
for sq in [0,1,63]:
    r=ref_seq(sq); e=(Cc[sq*SQ:(sq+1)*SQ]-r).abs().max().item(); print(f"OURS seq{sq} vs ref max_abs={e:.2e}",flush=True)
ms_ours=time_launches(ours,warmup=20,iters=100,stream=stream)
flop=2.0*(2.0*SQ*SK*HD)*0.5*H*NSEQ  # causal ~0.5
print(f"OURS  time={ms_ours*1e3:.1f}us  TF/s={flop/(ms_ours)/1e12:.1f}  (NSEQ={NSEQ}, {H*NSEQ} CTAs)",flush=True)
try:
    from aiter.ops.triton.attention.unified_attention import unified_attention
    out_ai=torch.empty(NSEQ*SQ,H,HD,device="cuda",dtype=torch.bfloat16)
    cu_q=torch.arange(0,(NSEQ+1)*SQ,SQ,dtype=torch.int32,device="cuda"); kvl=torch.full((NSEQ,),SK,dtype=torch.int32,device="cuda")
    def ait(): unified_attention(query,key_cache,value_cache,out_ai,cu_seqlens_q=cu_q,seqused_k=kvl,max_seqlen_q=SQ,max_seqlen_k=SK,softmax_scale=scale,causal=True,window_size=(-1,-1),block_table=bt,softcap=0.0,q_descale=None,k_descale=None,v_descale=None)
    ait(); torch.cuda.synchronize()
    ms_ai=time_launches(ait,warmup=20,iters=100,stream=stream)
    for sq in [0,1,63]:
        r=ref_seq(sq); e=(out_ai[sq*SQ:(sq+1)*SQ].float()-r).abs().max().item(); print(f"AITER seq{sq} vs ref max_abs={e:.2e}",flush=True)
    d=(Cc-out_ai.float()).abs().max().item()
    print(f"AITER time={ms_ai*1e3:.1f}us  TF/s={flop/(ms_ai)/1e12:.1f}   OURS/AITER={ms_ours/ms_ai:.2f}x  (ours-vs-aiter max_abs={d:.2e})",flush=True)
except Exception as e: print("AITER skipped:",str(e).splitlines()[0][:120],flush=True)
synchronize_and_release(stream)
