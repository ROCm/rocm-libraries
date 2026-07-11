import math, torch
import torch.nn.functional as Fnn
from kernels.common.attention_unified import UnifiedAttentionProblem, run_unified_attention_torch, _d256_gfx942_fast

H, HKV, HD = 16, 2, 256
GQAG = H // HKV
qlens = [300, 1, 100, 260]
klens = [300, 500, 500, 260]
NSEQ = len(qlens)
total_q = sum(qlens)
scale = 1.0 / math.sqrt(HD)
cu_q = [0]
for q in qlens:
    cu_q.append(cu_q[-1] + q)


def run_bs(BS):
    torch.manual_seed(0)
    blocks_per_seq = [(kl + BS - 1) // BS for kl in klens]
    max_bps = max(blocks_per_seq)
    num_blocks = sum(blocks_per_seq) + 4
    perm = torch.randperm(num_blocks)  # scattered (real paging)
    block_table = torch.zeros(NSEQ, max_bps, dtype=torch.int32)
    p = 0
    for i, nb in enumerate(blocks_per_seq):
        for j in range(nb):
            block_table[i, j] = perm[p]; p += 1
    key_cache = torch.randn(num_blocks, BS, HKV, HD, device="cuda", dtype=torch.bfloat16) * 0.3
    value_cache = torch.randn(num_blocks, BS, HKV, HD, device="cuda", dtype=torch.bfloat16) * 0.3
    q = torch.randn(total_q, H, HD, device="cuda", dtype=torch.bfloat16) * 0.3
    out = torch.empty(total_q, H, HD, device="cuda", dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor(cu_q, dtype=torch.int32, device="cuda")
    seqused_k = torch.tensor(klens, dtype=torch.int32, device="cuda")
    bt = block_table.cuda()
    problem = UnifiedAttentionProblem(
        total_q=total_q, num_seqs=NSEQ, num_query_heads=H, num_kv_heads=HKV, head_size=HD,
        block_size=BS, max_seqlen_q=max(qlens), max_seqlen_k=max(klens), dtype="bf16",
        sliding_window=0, softcap=0.0, use_sinks=False, use_alibi=False, use_qq_bias=False,
        use_fp8=False, num_kv_blocks=num_blocks,
    )
    assert _d256_gfx942_fast(problem), f"BS={BS} did not route to 4-warp GQA!"
    run_unified_attention_torch(
        problem=problem, q=q, k=key_cache, v=value_cache, out=out,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, softmax_scale=scale,
        block_table=bt, softcap=0.0, backend="auto",
    )
    torch.cuda.synchronize()
    bt_cpu = block_table.tolist()
    worst = 0.0
    for i in range(NSEQ):
        q0, q1 = cu_q[i], cu_q[i + 1]; ql, kl, nb = qlens[i], klens[i], blocks_per_seq[i]
        bidx = torch.tensor(bt_cpu[i][:nb], device="cuda")
        kseq = key_cache[bidx].reshape(-1, HKV, HD)[:kl].float()
        vseq = value_cache[bidx].reshape(-1, HKV, HD)[:kl].float()
        Qi = q[q0:q1].float()
        br = (torch.arange(kl, device="cuda")[None, :] <= ((kl - ql) + torch.arange(ql, device="cuda"))[:, None])
        for h in range(H):
            kvh = h // GQAG
            rs = Fnn.scaled_dot_product_attention(
                Qi[:, h, :].unsqueeze(0), kseq[:, kvh, :].unsqueeze(0), vseq[:, kvh, :].unsqueeze(0),
                attn_mask=br.unsqueeze(0), scale=scale).squeeze(0)
            worst = max(worst, (out[q0:q1, h, :].float() - rs).abs().max().item())
    print(f"BS={BS}: routed=4wgqa, ragged {NSEQ}seqs qlens={qlens} klens={klens} worst_max_abs={worst:.2e}  {'PASS' if worst < 5e-2 else 'FAIL'}", flush=True)
    return worst


for BS in (16, 32):
    run_bs(BS)
