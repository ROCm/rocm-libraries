/*
 * Q5 prototype for RFC 0001 (docs/rfcs/0001_HipdnnForwardingWrapper_phase1_investigation.md).
 *
 * Tiny consumer that exercises the public C API. Built against the *installed*
 * MIOpen headers (never the in-tree src/ tree) and linked against libMIOpen.so.
 * The matching shell harness (check_consumer_smoke.sh) asserts that:
 *   1. The translation unit compiles without seeing miopen_private_rename.h.
 *   2. The resulting binary has no undefined references to miopen*_impl
 *      symbols, proving the rename header did not leak.
 *
 * Intentionally does not call into a GPU — it should be runnable on
 * machines without a working accelerator, since this is a build-time test.
 */

#include <miopen/miopen.h>
#include <stdio.h>

int main(void)
{
    miopenHandle_t handle = NULL;
    miopenStatus_t s      = miopenCreate(&handle);
    if(s != miopenStatusSuccess)
    {
        /* Not a real failure for this build-time smoke: a missing GPU is fine. */
        fprintf(stderr, "miopenCreate returned %d (acceptable for build-time smoke)\n", s);
        return 0;
    }
    miopenDestroy(handle);
    return 0;
}
