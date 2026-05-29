# RUN: %stinkytofu-opt --arch gfx1250 %s -O0 --emit-asm --from-label token_start --to-label token_end 2>&1
# XFAIL
#
# Verify that the O0 pipeline catches inconsistent token annotations.
# Previously, the consistency check only ran inside StinkyBuildImplicitDependencyPass
# at O3, so O0 silently accepted bad input.
#
# CHECK: [MemTokenConsistencyCheck] ERROR: BB {{.*}} has inconsistent memory tokens
# CHECK: [has token]
# CHECK: [NO TOKEN]

token_start:
    ds_load_b128 v[0:3], v16 offset:0  // st.token:0
    ds_load_b128 v[4:7], v20 offset:0
    s_barrier_signal -1                // st.token:0
    s_barrier_wait -1                  // st.token:0
token_end:
