docker pull rocm/fw-bringup:gfx1250-atom--flydsl-mxfp8-gemm-20260909
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --entrypoint "" \
  rocm/fw-bringup:gfx1250-atom--flydsl-mxfp8-gemm-20260909 \
  bash -lc 'cd /app/aiter &&
    ENABLE_CK=0 \
    python3 op_tests/test_gemm_a8w8_blockscale.py \
      --flydsl \
      --ck_preshuffle True \
      -m 512 \
      --apre True \
      --init random \
      -nk 6144,7168 7168,3072 8192,1536 2048,7168 65536,1536 7168,16384'