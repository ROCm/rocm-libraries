# Filter to match the aiter test_batch_prefill.py run:
#   -s 8000 -p 1 -q 4 -k 1 --head_dim 256 -c true -d bf16 --input_dtype fp8 \
#   --quant_method pertensor --kv_layout linear -t sglang -l 0.0 --return_lse false
#
# data_type (fp8bf16), mask (top_left), and qscale (pertensor) are already
# scoped by batch_prefill_fp8_pertensor.json. This file pins the remaining
# batch_prefill feature axes that the JSON trait_config cannot express, so the
# sweep only varies the TILE shape (to find the fastest instance).


def filter_config(c):
    return (
        c.page_size == 1                       # -p 1
        and c.kv_memory_layout == "linear"     # --kv_layout linear (forced by page_size=1)
        and c.kv_lookup_table == "sglang"      # -t sglang
        and not c.logits                       # -l 0.0  (no logits soft-cap)
        and not c.lse                          # --return_lse false (already f for fp8)
    )
