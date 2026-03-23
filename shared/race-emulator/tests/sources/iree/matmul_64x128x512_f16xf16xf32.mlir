// 64x128x512 matmul: f16 inputs, f32 output.
//
// To compile for gfx942:
//   iree-compile matmul_64x128x512_f16xf16xf32.mlir \
//     --iree-hal-target-backends=rocm \
//     --iree-rocm-target=gfx942 \
//     --iree-hal-dump-executable-intermediates-to=/tmp/iree_dump \
//     -o /dev/null
//
// To compile for gfx1151:
//   iree-compile matmul_64x128x512_f16xf16xf32.mlir \
//     --iree-hal-target-backends=rocm \
//     --iree-rocm-target=gfx1151 \
//     --iree-hal-dump-executable-intermediates-to=/tmp/iree_dump \
//     -o /dev/null
//
// The .rocmasm file in the dump directory is the compiled assembly.

func.func @matmul(%a: tensor<64x512xf16>, %b: tensor<512x128xf16>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<64x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x128xf32>) -> tensor<64x128xf32>
  %result = linalg.matmul ins(%a, %b : tensor<64x512xf16>, tensor<512x128xf16>)
                          outs(%fill : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %result : tensor<64x128xf32>
}
