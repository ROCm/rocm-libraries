// Direct HIP convolution (streaming) kernels — configs 0-4 are Dgrad
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<0>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectHipConvBwdData4CFp16Kernel<4>, SIGNATURE>);

// Direct CK Tile convolution (streaming) kernels — configs 0-4 are Dgrad
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<0>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<1>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<2>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<3>, SIGNATURE>);
run_alg(run_direct_conv<direct_conv::DirectTileConvBwdData4CFp16Kernel<4>, SIGNATURE>);
