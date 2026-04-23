////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) kernels — configs 5-9 are Fprop
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<5>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<6>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<7>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<8>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward4CFp16Kernel<9>, SIGNATURE>);

/////////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) kernels — Fprop configs
/////////////////////////////////////////////////////////////////////////////

// V3 instances (native CK tile distribution implementation)
// Group 0: no swizzle, direct DRAM epilogue — Fprop indices 5-9
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<5, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<6, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<7, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<8, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<9, direct_conv::Version::v3>, SIGNATURE>);
// Group 1: no swizzle, LDS-staged epilogue — Fprop indices 15-19
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<15, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<16, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<17, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<18, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<19, direct_conv::Version::v3>, SIGNATURE>);
// Group 2: XOR swizzle, direct DRAM epilogue — Fprop indices 25-29
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<25, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<26, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<27, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<28, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<29, direct_conv::Version::v3>, SIGNATURE>);
// Group 3: XOR swizzle, LDS-staged epilogue — Fprop indices 35-39
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<35, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<36, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<37, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<38, direct_conv::Version::v3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward4CFp16Kernel<39, direct_conv::Version::v3>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct HIP convolution (streaming) 16c kernels — configs 9-17 are Fprop
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<9>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<10>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<11>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<12>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<13>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<14>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<15>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvForward16CFp16Kernel<17>, SIGNATURE>);

////////////////////////////////////////////////////////////////////////
// Direct CK Tile convolution (streaming) 16c kernels — Fprop configs 9-17
////////////////////////////////////////////////////////////////////////
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<9>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<10>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<11>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<12>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<13>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<14>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<15>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<16>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvForward16CFp16Kernel<17>, SIGNATURE>);
