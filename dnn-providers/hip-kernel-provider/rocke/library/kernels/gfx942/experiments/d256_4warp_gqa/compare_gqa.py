import sys, math, torch
sys.path.insert(0,"/home/AMD/avirgoel/wk")
import gqa_kernel as G
from rocke.runtime import LaunchConfig, time_launches, synchronize_and_release
H,HKV,NPHYS,BS,HD,NUM_KV,BN=G.H,G.HKV,G.NPHYS,G.BS,G.HD,G.NUM_KV,G.BN
SQ=128; SK=NUM_KV*BN  # 128 queries (tail), 512 keys
torch.manual_seed(1)
key_cache=torch.randn(NPHYS,BS,HKV,HD,device="cuda",dtype=torch.bfloat16)*0.3
value_cache=torch.randn(NPHYS,BS,HKV,HD,device="cuda",dtype=torch.bfloat16)*0.3
query=torch.randn(SQ,H,HD,device="cuda",dtype=torch.bfloat16)*0.3
bt=torch.randperm(NPHYS,dtype=torch.int32,device="cuda")
scale=HD**-0.5; stream=torch.cuda.current_stream().cuda_stream
# reference (causal, tail-aligned, GQA)
Klog=key_cache.view(NPHYS,BS,HKV,HD)[bt.long()].reshape(NPHYS*BS,HKV,HD)
Vlog=value_cache.view(NPHYS,BS,HKV,HD)[bt.long()].reshape(NPHYS*BS,HKV,HD)
qpos=(SK-SQ)+torch.arange(SQ,device="cuda"); kpos=torch.arange(SK,device="cuda")
mask=kpos[None,:]>qpos[:,None]
ref=torch.zeros(SQ,H,HD,device="cuda")
for h in range(H):
    kv=h//(H//HKV); s=(query[:,h,:].float()@Klog[:,kv,:].float().T)*scale
    ref[:,h,:]=torch.softmax(s.masked_fill(mask,float("-inf")),-1)@Vlog[:,kv,:].float()
# --- ours ---
Cc=torch.zeros(SQ,H,HD,device="cuda",dtype=torch.float32); klt=torch.tensor([SK],dtype=torch.int32,device="cuda")
def ours(): G.L({"Q":query,"K":key_cache,"V":value_cache,"C":Cc,"BT":bt,"KL":klt},config=LaunchConfig(grid=(H,1,1),block=(256,1,1),stream=stream))
ours(); torch.cuda.synchronize()
err_ours=(Cc-ref).abs().max().item()
ms_ours=time_launches(ours,warmup=20,iters=200,stream=stream)
print(f"OURS  max_abs={err_ours:.3e} time={ms_ours*1e3:.2f}us",flush=True)
# --- AITER ---
try:
    from aiter.ops.triton.attention.unified_attention import unified_attention
    out_ai=torch.empty(SQ,H,HD,device="cuda",dtype=torch.bfloat16)
    cu_q=torch.tensor([0,SQ],dtype=torch.int32,device="cuda"); kvl=torch.tensor([SK],dtype=torch.int32,device="cuda")
    btA=bt.view(1,NPHYS)
    def ait(): unified_attention(query,key_cache,value_cache,out_ai,cu_seqlens_q=cu_q,seqused_k=kvl,max_seqlen_q=SQ,max_seqlen_k=SK,softmax_scale=scale,causal=True,window_size=(-1,-1),block_table=btA,softcap=0.0,q_descale=None,k_descale=None,v_descale=None)
    ait(); torch.cuda.synchronize()
    err_ai=(out_ai.float()-ref).abs().max().item()
    ms_ai=time_launches(ait,warmup=20,iters=200,stream=stream)
    print(f"AITER max_abs={err_ai:.3e} time={ms_ai*1e3:.2f}us  OURS/AITER={ms_ours/ms_ai:.2f}x",flush=True)
except Exception as e: print("AITER skipped:",str(e).splitlines()[0][:120],flush=True)
synchronize_and_release(stream)
