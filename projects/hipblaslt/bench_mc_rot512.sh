# pure
# ./build/release/clients/hipblaslt-bench --yaml bbs_tn_ootb_bench_40_flat_rot384_pure.yaml
# fused
# ./build/release/clients/hipblaslt-bench --yaml bbs_tn_nk_pairs_m_sweep.yaml
# Out of the box hipblaslt-bench picks up /opt/rocm/lib/libhipblaslt.so.1 and the ROCm-installed
# device library, so it never sees locally tuned kernels. Three things have to be pinned:
#
#  - libhipblaslt.so, from build/release/library.
#  - libtensilelite-host.so, from build/release/tensilelite. A tuning run leaves its own copy in
#    tensilelite/build_tmp/tensilelite on LD_LIBRARY_PATH, which overrides our RUNPATH; pairing
#    that one with our libhipblaslt fails every load ("Could not load ...TensileLibrary_lazy").
#    Both entries must come first for the same reason.
#  - the device library, via HIPBLASLT_TENSILE_LIBPATH. It is a literal directory, so it names
#    the arch subdir, not its parent.
#
# The device library has to come from our build as a whole: ROCm's .dat files still carry the
# predicate name FusedGemmA2A, which this tree renamed to FusedA2A, so our host lib rejects any
# of them ("Unknown subclass type FusedGemmA2A", visible under TENSILE_DB=0x1000). That rules
# out mixing a tuned file into a copy of the installed library. Our build is logic-filtered and
# only covers Alik_Bljk BBS fused, which is every problem in the yaml below but nothing else.
export LD_LIBRARY_PATH="$PWD/build/release/library:$PWD/build/release/tensilelite:$LD_LIBRARY_PATH"
export HIPBLASLT_TENSILE_LIBPATH="$PWD/build/release/Tensile/library/gfx1250"
export HIP_VISIBLE_DEVICES=0

# fused v2
# ./build/release/clients/hipblaslt-bench --yaml bbs_tn_mk_pairs_n_sweep_v2.yaml > bbs_tn_mk_pairs_n_sweep_v2.log 2>&1

# flush on and off
HIP_VISIBLE_DEVICES=0 ./build/release/clients/hipblaslt-bench --yaml bbs_tn_mk_pairs_n_sweep_flush_on.yaml > bbs_tn_mk_pairs_n_sweep_flush_on.log 2>&1 &
HIP_VISIBLE_DEVICES=1 ./build/release/clients/hipblaslt-bench --yaml bbs_tn_mk_pairs_n_sweep_flush_off.yaml > bbs_tn_mk_pairs_n_sweep_flush_off.log 2>&1 &

# test
# ./build/release/clients/hipblaslt-bench --yaml bbs_tn_mk_pairs_n_sweep_flush_on.yaml