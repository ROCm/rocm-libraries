// Direct convolution (streaming) kernels — configs 5-9 are Fprop
run_alg(run_direct_conv_4c_fp16<SIGNATURE, 5>);
run_alg(run_direct_conv_4c_fp16<SIGNATURE, 6>);
run_alg(run_direct_conv_4c_fp16<SIGNATURE, 7>);
run_alg(run_direct_conv_4c_fp16<SIGNATURE, 8>);
run_alg(run_direct_conv_4c_fp16<SIGNATURE, 9>);
