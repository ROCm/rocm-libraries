# ONLY XDL_KERNELS
add_instance_library(device_grouped_gemm_fastgelu_instance
   device_grouped_gemm_fastgelu_xdl_f16_f16_f16_mk_kn_mn_instance.cpp
   device_grouped_gemm_fastgelu_xdl_f16_f16_f16_mk_nk_mn_instance.cpp
   device_grouped_gemm_fastgelu_xdl_f16_f16_f16_km_kn_mn_instance.cpp
   device_grouped_gemm_fastgelu_xdl_f16_f16_f16_km_nk_mn_instance.cpp
)
