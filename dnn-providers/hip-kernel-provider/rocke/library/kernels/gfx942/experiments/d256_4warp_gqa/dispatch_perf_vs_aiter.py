import math, torch
from kernels.common.attention_unified import UnifiedAttentionProblem, run_unified_attention_torch, _d256_gfx942_fast
from rocke.runtime import time_launches
H,HKV,HD,BS=16,2,256,16; GQAG=H//HKV
for SQ in (4096,8192):
    NB=(SQ+BS-1)//BS+2
    kc=torch.randn(NB,BS,HKV,HD,device="cuda",dtype=torch.bfloat16)*0.3
    vc=torch.randn(NB,BS,HKV,HD,device="cuda",dtype=torch.bfloat16)*0.3
    q=torch.randn(SQ,H,HD,device="cuda",dtype=torch.bfloat16)*0.3
    out=torch.empty(SQ,H,HD,device="cuda",dtype=torch.bfloat16)
    cuq=torch.tensor([0,SQ],dtype=torch.int32,device="cuda"); sk=torch.tensor([SQ],dtype=torch.int32,device="cuda")
    bt=torch.arange(NB,dtype=torch.int32,device="cuda").view(1,NB)
    scale=1.0/math.sqrt(HD); st=torch.cuda.current_stream().cuda_stream
    prob=UnifiedAttentionProblem(total_q=SQ,num_seqs=1,num_query_heads=H,num_kv_heads=HKV,head_size=HD,block_size=BS,max_seqlen_q=SQ,max_seqlen_k=SQ,dtype="bf16",sliding_window=0,softcap=0.0,use_sinks=False,use_alibi=False,use_qq_bias=False,use_fp8=False,num_kv_blocks=NB)
    assert _d256_gfx942_fast(prob)
    def ours(): run_unified_attention_torch(problem=prob,q=q,k=kc,v=vc,out=out,cu_seqlens_q=cuq,seqused_k=sk,softmax_scale=scale,block_table=bt,softcap=0.0,backend="auto")
    ours(); torch.cuda.synchronize()
    mo=time_launches(ours,warmup=10,iters=50,stream=st)
    r=""
    try:
        from aiter.ops.triton.attention.unified_attention import unified_attention
        oa=torch.empty(SQ,H,HD,device="cuda",dtype=torch.bfloat16)
        def ait(): unified_attention(q,kc,vc,oa,cu_seqlens_q=cuq,seqused_k=sk,max_seqlen_q=SQ,max_seqlen_k=SQ,softmax_scale=scale,causal=True,window_size=(-1,-1),block_table=bt,softcap=0.0,q_descale=None,k_descale=None,v_descale=None)
        ait(); torch.cuda.synchronize()
        ma=time_launches(ait,warmup=10,iters=50,stream=st)
        r=f" | AITER {ma*1e3:.1f}us | OURS/AITER={mo/ma:.2f}x (SAME-RUN)"
    except Exception as e: r=f" | AITER skipped: {str(e).splitlines()[0][:80]}"
    print(f"DISPATCH SQ={SQ} bs=16 OURS {mo*1e3:.1f}us{r}",flush=True)
