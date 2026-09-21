# RUN: %stinkytofu-opt --arch gfx1250 %s --StinkyBuildImplicitDependencyPass --StinkyDAGSchedulerPass --emit-asm --from-label token_start --to-label token_end
#
# Two independent memory-token groups annotated via "// st.token:N" comments in
# raw assembly.  The assembler ignores the comments; StinkyTofu reads them to
# build use-def chains through LDS pseudo-registers.
#
# Group 0: ds_load → signal → wait → ds_load (post-barrier)
# Group 1: ds_load → signal → wait → ds_load (post-barrier)
#
# After DAG scheduling the relative ordering within each group must be preserved:
#   - each group's pre-barrier ds_load must precede its signal
#   - signal must precede wait
#   - wait must precede post-barrier ds_load
#
# Group 1 signal/wait are before group 0 signal/wait in the source; the scheduler
# must not let group 0's wait slip past group 1's post-barrier load.
#
# Pre-barrier loads appear before signals (within-group ordering preserved)
# CHECK: ds_load_b128 v[0:3], v16
# CHECK: ds_load_b128 v[4:7], v20
# Both signals appear (one per group, scheduler may interleave)
# CHECK-COUNT-2: s_barrier_signal
# Both waits appear
# CHECK-COUNT-2: s_barrier_wait
# Post-barrier loads appear somewhere after the first wait (exact interleaving is scheduler-chosen)
# CHECK-DAG: ds_load_b128 v[8:11], v24
# CHECK-DAG: ds_load_b128 v[12:15], v28

token_start:
    ds_load_b128 v[0:3], v16 offset:0    // st.token:0  pre-barrier load, group 0
    ds_load_b128 v[4:7], v20 offset:0    // st.token:1  pre-barrier load, group 1
    v_wmma_f32_16x16x32_bf16 a[0:7], v[0:7], v[0:7], a[0:7]
    s_barrier_signal -1                  // st.token:1
    s_barrier_signal -1                  // st.token:0
    s_barrier_wait -1                    // st.token:1
    s_barrier_wait -1                    // st.token:0
    ds_load_b128 v[8:11], v24 offset:0   // st.token:0  post-barrier load, group 0
    ds_load_b128 v[12:15], v28 offset:0  // st.token:1  post-barrier load, group 1
token_end:
