// Direct convolution (streaming) kernels — configs 0-4 are Dgrad
run_alg(run_direct_conv_4c_fp16<SIGNATURE, 0>);
run_alg(run_direct_conv_4c_fp16<SIGNATURE, 1>);
run_alg(run_direct_conv_4c_fp16<SIGNATURE, 2>);
run_alg(run_direct_conv_4c_fp16<SIGNATURE, 3>);
run_alg(run_direct_conv_4c_fp16<SIGNATURE, 4>);
