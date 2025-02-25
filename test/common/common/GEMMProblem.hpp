struct GEMMProblem
{
    // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
    int   m     = 512;
    int   n     = 512;
    int   k     = 128;
    float alpha = 2.0f;
    float beta  = 0.5f;

    // output macro tile size
    int macM = 64;
    int macN = 64;
    int macK = 64;

    // Wave tile size
    int waveM = 32;
    int waveN = 32;
    int waveK = 2;
    int waveB = 1;

    // Workgroup size
    uint wavefrontSize  = 64;
    uint workgroupSizeX = 2 * wavefrontSize;
    uint workgroupSizeY = 2;

    uint numCUs = 0;

    std::string transA = "N";
    std::string transB = "T";

    // Unroll Sizes
    unsigned int unrollK = 0;

    bool loadLDSA  = true;
    bool loadLDSB  = true;
    bool storeLDSD = true;

    bool fuseLoops                 = true;
    bool allowAmbiguousMemoryNodes = false;
    bool betaInFma                 = true;
    bool literalStrides            = true;

    bool prefetch          = false;
    int  prefetchInFlight  = 1;
    int  prefetchLDSFactor = 0;
    bool prefetchMixMemOps = false;

    bool packMultipleElementsInto1VGPR = true;

    bool loopOverTiles  = false;
    bool streamK        = false;
    bool streamKTwoTile = false;

    bool splitStoreTileIntoWaveBlocks = false;

    uint8_t scaleA = 127;
    uint8_t scaleB = 127;
};
