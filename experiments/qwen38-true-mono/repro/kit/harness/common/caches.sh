# Sourced by the q38 harness job scripts (container mode). Job-local compile caches (ROADMAP T0):
# comgr, FlyDSL, Triton and inductor under /job/cache, fresh for every job, so concurrent jobs never
# share an entry and no job reuses another's compile. Q38_SHARED_CACHE=1 opts FlyDSL / Triton /
# inductor back into the node's shared /work/cache (warm starts for quick iterations; those caches
# are keyed by sources and build parameters); comgr stays per job either way.
mkdir -p /job/cache/comgr 2> /dev/null && export AMD_COMGR_CACHE_DIR=/job/cache/comgr
if [ "${Q38_SHARED_CACHE:-0}" != 1 ]; then
  export FLYDSL_RUNTIME_CACHE_DIR=/job/cache/flydsl TRITON_CACHE_DIR=/job/cache/triton TORCHINDUCTOR_CACHE_DIR=/job/cache/inductor
  mkdir -p "$FLYDSL_RUNTIME_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
fi
echo "q38: compile caches comgr=$AMD_COMGR_CACHE_DIR flydsl=${FLYDSL_RUNTIME_CACHE_DIR:-<image default>} triton=${TRITON_CACHE_DIR:-<image default>}"
