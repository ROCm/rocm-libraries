#include "GEMMProblem.hpp"

GEMMProblem setup_GEMMF8_NT()
{
    GEMMProblem gemm;

    // 4x2 jamming
    uint wavesPerWGX = 16;
    uint wavesPerWGY = 2;

    gemm.waveM = 16;
    gemm.waveN = 16;
    gemm.waveK = 32;

    gemm.macM = wavesPerWGX * gemm.waveM;
    gemm.macN = wavesPerWGY * gemm.waveN;
    gemm.macK = 2 * gemm.waveK;

    gemm.loadLDSA = true;
    gemm.loadLDSB = true;

    gemm.workgroupSizeX = 256;
    gemm.workgroupSizeY = 1;

    gemm.m = 33 * gemm.macM;
    gemm.n = 17 * gemm.macN;
    gemm.k = 4 * gemm.macK;

    gemm.alpha = 2.1;
    gemm.beta  = 0.75;

    gemm.transA = "N";
    gemm.transB = "T";

    return gemm;
}

GEMMProblem setup_GEMMF8F6F4_NT(int waveM, int waveN, int waveK)
{
    GEMMProblem gemm;

    // 2x2 jamming
    uint wavesPerWGX = 4;
    uint wavesPerWGY = 4;

    gemm.waveM = waveM;
    gemm.waveN = waveN;
    gemm.waveK = waveK;

    gemm.macM = wavesPerWGX * gemm.waveM;
    gemm.macN = wavesPerWGY * gemm.waveN;
    gemm.macK = 2 * gemm.waveK;

    gemm.loadLDSA = true;
    gemm.loadLDSB = true;

    gemm.workgroupSizeX = 256;
    gemm.workgroupSizeY = 1;

    gemm.m = 2 * gemm.macM;
    gemm.n = 3 * gemm.macN;
    gemm.k = 4 * gemm.macK;

    gemm.alpha = 2.1;
    gemm.beta  = 0.75;

    gemm.transA = "N";
    gemm.transB = "T";

    return gemm;
}

GEMMProblem setup_GEMMF8_TN()
{
    GEMMProblem gemm;

    // 1x1 jamming
    uint wavesPerWGX = 4;
    uint wavesPerWGY = 1;

    gemm.waveM = 16;
    gemm.waveN = 16;
    gemm.waveK = 32;

    gemm.macM = wavesPerWGX * gemm.waveM;
    gemm.macN = wavesPerWGY * gemm.waveN;
    gemm.macK = 2 * gemm.waveK;

    gemm.loadLDSA  = true;
    gemm.loadLDSB  = true;
    gemm.storeLDSD = false;

    gemm.workgroupSizeX = 256;
    gemm.workgroupSizeY = 1;

    gemm.m = 33 * gemm.macM;
    gemm.n = 17 * gemm.macN;
    gemm.k = 4 * gemm.macK;

    gemm.alpha = 2.1;
    gemm.beta  = 0.75;

    gemm.transA = "T";
    gemm.transB = "N";

    return gemm;
}

GEMMProblem setup_GEMMF8F6F4_TN(int waveM, int waveN, int waveK)
{
    GEMMProblem gemm;

    // 2x2 jamming
    uint wavesPerWGX = 4;
    uint wavesPerWGY = 4;

    gemm.waveM = waveM;
    gemm.waveN = waveN;
    gemm.waveK = waveK;

    gemm.macM = wavesPerWGX * gemm.waveM;
    gemm.macN = wavesPerWGY * gemm.waveN;
    gemm.macK = 2 * gemm.waveK;

    gemm.loadLDSA  = true;
    gemm.loadLDSB  = true;
    gemm.storeLDSD = false;

    gemm.workgroupSizeX = 256;
    gemm.workgroupSizeY = 1;

    gemm.m = 2 * gemm.macM;
    gemm.n = 3 * gemm.macN;
    gemm.k = 4 * gemm.macK;

    gemm.alpha = 2.1;
    gemm.beta  = 0.75;

    gemm.transA = "T";
    gemm.transB = "N";

    return gemm;
}
