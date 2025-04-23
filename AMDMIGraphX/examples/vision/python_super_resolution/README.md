# Super Resolution with AMD MIGraphX

This example is based on [ONNX run_super_resolution_model notebook](https://github.com/onnx/models/blob/master/vision/super_resolution/sub_pixel_cnn_2016/dependencies/Run_Super_Resolution_Model.ipynb) and modified for MIGraphX.

## Description
Given an input image, this application resizes the image to 224x224 and then scales it to 672x672, thus it is useful for upscaling low-resolution images.

### Model Utilized
> "Super Resolution uses efficient  [Sub-pixel convolutional layer](https://arxiv.org/abs/1609.05158) described for increasing spatial resolution within network tasks. By increasing pixel count, images are then clarified, sharpened, and upscaled without losing the input image’s content and characteristics." [[Reference]](https://github.com/onnx/models/blob/master/vision/super_resolution/sub_pixel_cnn_2016/README.md)

Model in PyTorch definitions:
```
self.relu = nn.ReLU(inplace=inplace)
self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
```
## How-to
If you have jupyter installed, you can simply use the notebook given. Otherwise please follow the step-by-step guide.
### Jupyter Notebook
Run Jupyter notebook server on a ROCm and MIGraphX installed system, and run `Run_Super_Resolution_Model.ipynb`

### Step by Step
1) Upgrade pip3. You may skip this stage if you already have latest pip3. This step is needed for OpenCV installation.
```
pip3 install --upgrade pip
```
2) Install requirements.
```
pip3 install -r requirements.txt
```
3) Import required libraries.
```
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from resizeimage import resizeimage
```
4) Download ONNX model.
```
wget -nc https://github.com/onnx/models/raw/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx
```
5) Preprocess the sample image `cat.jpg`.
```
orig_img = Image.open("./cat.jpg")
print(orig_img.size)
img = resizeimage.resize_cover(orig_img, [224,224], validate=False)
img_ycbcr = img.convert('YCbCr')
img_y_0, img_cb, img_cr = img_ycbcr.split()
img_ndarray = np.asarray(img_y_0)

img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
img_5 = img_4.astype(np.float32) / 255.0
```
6) Import MIGraphX, parse & compile the ONNX model with MIGraphX. Print the model.
```
model = migraphx.parse_onnx("super-resolution-10.onnx")
model.compile(migraphx.get_target("gpu"))
model.print()
```
7) You can check the model inputs and outputs with the following functions.
```
print(model.get_parameter_names())
print(model.get_parameter_shapes())
print(model.get_output_shapes())
```
8) Run the image throgh model and get the output data.
```
result = model.run({
         "input": img_5
     })

data = np.array(result[0])[0]
```
9) Post processing image. If matplotlib is installed correctly, it should show up the image. The output image will be stored with filename `output.jpg`.
```
img_out_y = Image.fromarray(np.uint8((data* 255.0).clip(0, 255)[0]), mode='L')
# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
final_img.save("output.jpg")
print(final_img.size)
```
10) Measure the improvement in terms of PSNR and show the both input and super-resolution image:
```
import cv2

imgIN = cv2.imread('cat.jpg')
imgOUT = cv2.imread('output.jpg')
imgIN = cv2.cvtColor(imgIN, cv2.COLOR_BGR2RGB) #BGR to RGB
imgOUT = cv2.cvtColor(imgOUT, cv2.COLOR_BGR2RGB)

imgIN_resized = cv2.resize(imgIN, (672,672)) #Resizing input to 672

psnr = cv2.PSNR(imgIN_resized, imgOUT) #dimensions need to be same
print("PSNR Value = %.3f db"%psnr)

fig = plt.figure(figsize=(16, 16))
sp1 = fig.add_subplot(1, 2, 1)
sp1.title.set_text('Output Super Resolution Image (%sx%s)'%(imgOUT.shape[0], imgOUT.shape[1]))
plt.imshow(imgOUT)

sp2 = fig.add_subplot(1, 2, 2)
sp2.title.set_text('Input Image (%sx%s)'%(imgIN.shape[0], imgIN.shape[1]))
plt.imshow(imgIN)
plt.show()
```
