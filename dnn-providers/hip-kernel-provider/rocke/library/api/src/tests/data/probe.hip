extern "C" __global__ void rocke_test_probe(float* p)
{
    if(p) *p = 1.0f;
}
