# AMD OpenVX&trade; OpenCV Extension Function Tests

## Script to run vision tests

```
python runOpenCVTests.py --help
```


usage:

```
runOpenCVTests.py [--runvx_directory RUNVX_DIRECTORY] -- required
                  [--list_tests LIST_TESTS]
                  [--num_frames NUM_FRAMES]

Arguments:
  -h, --help        show this help message and exit
  --runvx_directory RunVX Executable Directory - required
  --list_tests      List OpenCV GDF Tests - optional (default:no [options:no/yes])
  --num_frames      Run Test for X number of frames - optional (default:1000 [range:1 - N])
```

## Vision Functionality Test GDFs

```
absdiff.gdf
adaptiveThreshold.gdf
add.gdf
addWeighted.gdf
bilateralFilter.gdf
bitwise_and.gdf
bitwise_not.gdf
bitwise_or.gdf
bitwise_xor.gdf
blur.gdf
boxFilter.gdf
canny.gdf
compare.gdf
convertScaleAbs.gdf
cornerMinEigenVal.gdf
cornerharris.gdf
cvBuildPyramid.gdf
cvtColor.gdf
dilate.gdf
distanceTransform.gdf
divide.gdf
erode.gdf
fastNlMeansDenoising.gdf
fastNlMeansDenoisingColored.gdf
filter2D.gdf
flip.gdf
gaussianBlur.gdf
goodFeatures.gdf
inputs
laplacian.gdf
medianBlur.gdf
morphologyEX.gdf
multiply.gdf
pyrDown.gdf
pyrUp.gdf
resize.gdf
scharr.gdf
sepFilter2D.gdf
simple_blob.gdf
sobel.gdf
subtract.gdf
threshold.gdf
transpose.gdf
warpAffine.gdf
warpPerspective.gdf
BRISK_Compute.gdf
BRISK_Detect.gdf
FAST.gdf
MSER_Detect.gdf
ORB_Compute.gdf
ORB_Detect.gdf
star_Detect.gdf
SIFT_Compute.gdf
SIFT_Detect.gdf
SURF_Compute.gdf
SURF_Detect.gdf
```

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
