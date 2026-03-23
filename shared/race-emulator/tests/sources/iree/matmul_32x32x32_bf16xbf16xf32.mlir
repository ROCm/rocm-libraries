// 32x32x32 matmul: bf16 inputs, f32 output.
//
// To compile for gfx942:
//   iree-compile matmul_32x32x32_bf16xbf16xf32.mlir \
//     --iree-hal-target-backends=rocm \
//     --iree-rocm-target=gfx942 \
//     --iree-hal-dump-executable-intermediates-to=/tmp/iree_dump \
//     -o /dev/null
//
// The .rocmasm file in the dump directory is the compiled assembly.

func.func @matmul(%a: tensor<32x32xbf16>, %b: tensor<32x32xbf16>) -> tensor<32x32xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<32x32xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<32x32xf32>) -> tensor<32x32xf32>
  %result = linalg.matmul ins(%a, %b : tensor<32x32xbf16>, tensor<32x32xbf16>)
                          outs(%fill : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}
