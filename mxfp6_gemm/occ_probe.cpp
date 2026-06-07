// occ2 occupancy probe — authoritative gate. Queries the runtime for actual
// max-active-blocks/CU for each occ2 kernel instance (VGPR+AGPR+LDS combined).
// blocks/CU == 2 -> occ2 achieved; == 1 -> NO-GO. NOT a production file.
#include <cstdio>
#include "mxfp6_asm_utils.hpp"
#include "mxfp6_preprocess.hpp"
#include "mxfp6_reference.hpp"
#include "mxfp6_types.hpp"
#include "mxfp6_lds.hpp"
#include "mxfp6_lds_occ2.hpp"
using namespace mxfp6;

template <int MT, int NT, int KT, int WM, int WN, int OCC, int SWZ, bool DB, typename OutT>
static void probe(const char* tag) {
    int lds = (DB ? 2 : 1) * (MT * (KT * 6 / 8) + NT * (KT * 6 / 8));
    auto k = lds_gemm_db<MT, NT, KT, WM, WN, OCC, SWZ, DB, OutT>;
    int blocks = -1;
    hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (const void*)k, 256, (size_t)lds);
    int dynblocks = -1;
    // also probe with 0 dynamic LDS to isolate the register-only ceiling
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&dynblocks, (const void*)k, 256, 0);
    printf("  %-30s LDS=%6dB  blocks/CU(LDS)=%d  blocks/CU(reg-only)=%d %s%s\n",
           tag, lds, blocks, dynblocks,
           (e != hipSuccess ? " ERR:" : ""), (e != hipSuccess ? hipGetErrorString(e) : ""));
}

int main() {
    int dev = 0; hipDeviceProp_t p; hipGetDeviceProperties(&p, dev);
    printf("Device: %s  CUs=%d  sharedMemPerBlock=%zu  regsPerBlock=%d\n",
           p.name, p.multiProcessorCount, p.sharedMemPerBlock, p.regsPerBlock);
    printf("=== occ2 max-active-blocks/CU probe (2 => occ2 OK, 1 => NO-GO) ===\n");
    // occ2-DEEP 128x256 KT256 single
    probe<128,256,256,2,2,2, 0,false,float >("DEEP F32  SWZ0");
    probe<128,256,256,2,2,2, 0,false,__half>("DEEP F16  SWZ0");
    // occ2-DB 128x256 KT128 double
    probe<128,256,128,2,2,2, 0,true, float >("DB   F32  SWZ0");
    probe<128,256,128,2,2,2, 0,true, __half>("DB   F16  SWZ0");
    // Reference: production occ1 config (expect 1)
    probe<256,256,192,2,2,1,16,true, __half>("PROD occ1 256x256 KT192 F16");

    // Phase 2 drip kernel (lds_gemm_occ2) — must keep occ=2.
    printf("=== Phase 2 drip kernel lds_gemm_occ2 (must stay 2) ===\n");
    {
        auto probe_drip = [](const char* tag, auto k, int lds) {
            int b1 = -1, b2 = -1;
            hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(&b1, (const void*)k, 256, (size_t)lds);
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&b2, (const void*)k, 256, 0);
            printf("  %-30s LDS=%6dB  blocks/CU(LDS)=%d  blocks/CU(reg-only)=%d %s%s\n",
                   tag, lds, b1, b2, (e != hipSuccess ? " ERR:" : ""),
                   (e != hipSuccess ? hipGetErrorString(e) : ""));
        };
        int lds = 2 * (128 * (128 * 6 / 8) + 256 * (128 * 6 / 8));  // DB KT128
        probe_drip("DRIP F32 SWZ0", lds_gemm_occ2<128,256,128,2,2,2,0,true,float>, lds);
        probe_drip("DRIP F16 SWZ0", lds_gemm_occ2<128,256,128,2,2,2,0,true,__half>, lds);
    }
    return 0;
}
