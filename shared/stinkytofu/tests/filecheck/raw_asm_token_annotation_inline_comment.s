# RUN: %stinkytofu-opt --arch gfx1250 %s --StinkyBuildImplicitDependencyPass --emit-asm --from-label token_start --to-label token_end
#
# Verify that "// st.token:N" is recognized even when it follows other comment
# text on the same line (e.g. "// load A // st.token:0").
# All three token-annotated instructions should pick up token 0 consistently,
# so StinkyBuildImplicitDependencyPass must not error.
#
# CHECK: ds_load_b128
# CHECK: s_barrier_signal
# CHECK: s_barrier_wait

token_start:
    ds_load_b128 v[0:3], v16 offset:0  // load matrix A tile // st.token:0
    s_barrier_signal -1                // all-wave barrier st.token:0
    s_barrier_wait -1                  // wait for signal st.token:0
token_end:
