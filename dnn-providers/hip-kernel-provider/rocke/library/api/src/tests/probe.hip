// Trivial GPU kernel compiled cross-arch at build time to generate the
// rocKE-client integration-test probe bundle.
// Built by the generate_test_bundle.py CMake custom-command via:
//   hipcc --genco --offload-arch=<arch> probe.hip
//   clang-offload-bundler --unbundle -> probe_<arch>.hsaco
//   generate_test_bundle.py --hsaco probe_<arch>.hsaco
// Symbol: rocke_test_probe
// toc_key: rocke/test/probe/rocke_test_probe

extern "C" __global__ void rocke_test_probe(float* p)
{
    if(p)
        *p = 1.0f;
}
