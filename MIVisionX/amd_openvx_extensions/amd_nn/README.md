# OpenVX Neural Network Extension Library (vx_nn)

vx_nn is an OpenVX Neural Network extension module. This implementation supports only floating-point tensor data type and does not support 8-bit and 16-bit fixed-point data types specified in the OpenVX specification.

### List of supported tensor and neural network layers:

| Layer name | Function | Kernel name |
| ---------- |--------- | ----------- |
| Activation|vxActivationLayer|org.khronos.nn_extension.activation_layer |
| Argmax|vxArgmaxLayerNode|com.amd.nn_extension.argmax_layer |
| Batch Normalization|vxBatchNormalizationLayer|com.amd.nn_extension.batch_normalization_layer |
| Cast|vxCastLayer|com.amd.nn_extension.cast_layer|
| Concat|vxConcatLayer|com.amd.nn_extension.concat_layer |
| Convolution|vxConvolutionLayer|org.khronos.nn_extension.convolution_layer |
| Crop|vxCropLayer|com.amd.nn_extension.crop_layer |
| CropAndResize|vxCropAndResizeLayer|com.amd.nn_extension.crop_and_resize_layer |
| Deconvolution|vxDeconvolutionLayer|org.khronos.nn_extension.deconvolution_layer |
| Detection Output|vxDetectionOutputLayer|com.amd.nn_extension.detection_output |
| Fully Connected|vxFullyConnectedLayer|org.khronos.nn_extension.fully_connected_layer |
| Gather|vxGatherLayer|com.amd.nn_extension.gather_layer |
| Local Response Normalization|vxNormalizationLayer|org.khronos.nn_extension.normalization_layer |
| Non Max Suppression|vxNMSLayer|com.amd.nn_extension.nms_layer |
| Permute|vxPermuteLayer|com.amd.nn_extension.permute_layer |
| Pooling|vxPoolingLayer|org.khronos.nn_extension.pooling_layer |
| Prior Box|vxPriorBoxLayer|com.amd.nn_extension.prior_box_layer|
| Reduce Min|vxReduceMinLayer|com.amd.nn_extension.reduce_min_layer|
| ROI Pooling|vxROIPoolingLayer|org.khronos.nn_extension.roi_pooling_layer |
| Scale|vxScaleLayer|com.amd.nn_extension.scale_layer |
| Slice|vxSliceLayer|com.amd.nn_extension.slice_layer |
| Softmax|vxSoftmaxLayer|org.khronos.nn_extension.softmax_layer |
| Tensor Add|vxTensorAddNode|org.khronos.openvx.tensor_add |
| Tensor Compare|vxTensorCompareNode|com.amd.nn_extension.tensor_compare |
| Tensor Convert Depth|vxTensorConvertDepthNode|org.khronos.openvx.tensor_convert_depth |
| Tensor Convert from Image|vxConvertImageToTensorNode|com.amd.nn_extension.convert_image_to_tensor |
| Tensor Convert to Image|vxConvertTensorToImageNode|com.amd.nn_extension.convert_tensor_to_image |
| Tensor Exponential|vxTensorExpNode|com.amd.nn_extension.tensor_exp |
| Tensor Log|vxTensorLogNode|com.amd.nn_extension.tensor_log |
| Tensor Matrix Multiply|vxTensorMatrixMultiplyNode|org.khronos.openvx.tensor_matrix_multiply |
| Tensor Max|vxTensorMaxNode|com.amd.nn_extension.tensor_max |
| Tensor Min|vxTensorMinNode|com.amd.nn_extension.tensor_min |
| Tensor Multiply|vxTensorMultiplyNode|org.khronos.openvx.tensor_multiply |
| Tensor Subtract|vxTensorSubtractNode|org.khronos.openvx.tensor_subtract |
| Tile|vxTileLayer|com.amd.nn_extension.tile_layer |
| TopK|vxTopKLayer|com.amd.nn_extension.topk_layer|
| Upsample Nearest Neighborhood|vxUpsampleNearestLayer|com.amd.nn_extension.upsample_nearest_layer |

### Example 1: Convert an image to a tensor of type float32

Use the below GDF with RunVX.

``` 
import vx_nn

data input = image:32,32,RGB2
data output = tensor:4,{32,32,3,1},VX_TYPE_FLOAT32,0
data a = scalar:FLOAT32,1.0
data b = scalar:FLOAT32,0.0
data reverse_channel_order = scalar:BOOL,0
read input input.png
node com.amd.nn_extension.convert_image_to_tensor input output a b reverse_channel_order
write output input.f32
```

### Example 2: 2x2 Upsample a tensor of type float32

Use the below GDF with RunVX.

``` 
import vx_nn

data input = tensor:4,{80,80,3,1},VX_TYPE_FLOAT32,0
data output = tensor:4,{160,160,3,1},VX_TYPE_FLOAT32,0

read input tensor.f32
node com.amd.nn_extension.upsample_nearest_layer input output
write output upsample.f32
```
