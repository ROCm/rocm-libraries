func.func @matmul(%a: tensor<16x512xbf16>, %b: tensor<512x16xbf16>) -> tensor<16x16xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
  %result = linalg.matmul ins(%a, %b : tensor<16x512xbf16>, tensor<512x16xbf16>)
                          outs(%fill : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}
