# RUN: %stinkytofu-opt --arch gfx1250 %s --emit-asm --from-label token_start --to-label token_end
#
# At O0 (no StinkyBuildImplicitDependencyPass), inconsistent token annotations
# should NOT trigger an abort.  The instructions are emitted verbatim.
#
# CHECK: ds_load_b128 v[0:3], v16
# CHECK: ds_load_b128 v[4:7], v20
# CHECK-NOT: ERROR

token_start:
    ds_load_b128 v[0:3], v16 offset:0  // st.token:0
    ds_load_b128 v[4:7], v20 offset:0
    s_barrier_signal -1                // st.token:0
    s_barrier_wait -1                  // st.token:0
token_end:
