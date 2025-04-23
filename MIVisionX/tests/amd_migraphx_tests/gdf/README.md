* To run the mnist_migraphx GDF, 
* Download the mnist model to this directory, then run the sample.
```
wget https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/model/mnist-8.onnx?raw=true
/opt/rocm/bin/runvx mnist_migraphx.gdf
```

* To change the input image, use either image_0.jpg, image_1.jpg or image_4.jpg each corresponding to digits 0, 1 and 4 respectively.
* The output file out_mnist.f32 is for digit image_4.jpg
