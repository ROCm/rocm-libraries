# RUN: %stinkytofu-opt --arch gfx1250 %s --MemTokenConsistencyCheckPass --emit-asm --from-label token_start --to-label token_end 2>&1
# XFAIL
#
# Error case: one ds_load has a "// st.token:0" annotation but the other does
# not.  MemTokenConsistencyCheckPass requires all ds_load/ds_store/
# tensor_load instructions in a block to be consistently annotated — either all
# have MemTokenData or none do.  The pass prints an error to stderr (captured
# via 2>&1) and aborts.  XFAIL marks the non-zero exit as expected.
#
# CHECK: [MemTokenConsistencyCheck] ERROR: BB "entry" has inconsistent memory tokens
# CHECK: [has token]
# CHECK: [NO TOKEN]

token_start:
    ds_load_b128 v[0:3], v16 offset:0  // st.token:0
    ds_load_b128 v[4:7], v20 offset:0
    s_barrier_signal -1                // st.token:0
    s_barrier_wait -1                  // st.token:0
token_end:
