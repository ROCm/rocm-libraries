# Pin exactly TWO tiles for a head-to-head across batch sizes, on the deployable
# aiter serving config (page_size=1, linear, sglang, no logits, no lse):
#
#   candidate    : 128x64x64  bpc=2   (FmhaFwdTileSize(128, 64, 64, 256,32,256, ... 2))
#   aiter default: 128x128x32 bpc=-1  (FmhaFwdTileSize(128,128,32, 256,32,256, ... -1))
#
# For hdim=256 the (tile_m0, tile_n0, tile_k0) triple uniquely identifies the
# tile (n1=256, k1=32, k0max=256 are fixed), so pinning those three + block_per_cu
# is sufficient. The JSON's block_per_cu sweep must include both 2 and -1.


def _features_ok(c):
    return (
        c.page_size == 1                       # -p 1
        and c.kv_memory_layout == "linear"     # --kv_layout linear
        and c.kv_lookup_table == "sglang"      # -t sglang
        and not c.logits                       # -l 0.0
        and not c.lse                          # --return_lse false
    )


def filter_config(c):
    if not _features_ok(c):
        return False
    candidate = (
        c.tile_m0 == 128 and c.tile_n0 == 64 and c.tile_k0 == 64 and c.block_per_cu == 2
    )
    aiter_default = (
        c.tile_m0 == 128
        and c.tile_n0 == 128
        and c.tile_k0 == 32
        and c.block_per_cu == -1
    )
    return candidate or aiter_default
