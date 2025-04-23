# AMD OpenCV Extension

The AMD OpenCV (vx_opencv) is an OpenVX module that implements a mechanism to access OpenCV functionality as OpenVX kernels. These kernels can be accessed from within OpenVX framework using OpenVX API call [vxLoadKernels](https://www.khronos.org/registry/vx/specs/1.0.1/html/da/d83/group__group__user__kernels.html#gae00b6343fbb0126e3bf0f587b09393a3)(context, "vx_opencv").

## List of OpenCV-interop kernels

The following is a list of OpenCV functions that have been included in the vx_opencv module.

    absDiff                     	org.opencv.absdiff                              
    adaptiveThreshold           	org.opencv.adaptivethreshold                                                          
    add                         	org.opencv.add 
    addWeighted                 	org.opencv.addweighted                          
    bilateralFilter             	org.opencv.bilateralfilter
    bitwise_and                 	org.opencv.bitwise_and
    bitwise_not                 	org.opencv.bitwise_not
    bitwise_or                  	org.opencv.bitwise_or
    bitwise_xor                 	org.opencv.bitwise_xor
    blur                        	org.opencv.blur
    boxfilter                   	org.opencv.boxfilter
    BRISK                       	org.opencv.brisk_detect
    BRISK_Compute               	org.opencv.brisk_compute 
    buildOpticalFlowPyramid     	org.opencv.buildopticalflowpyramid
    buildPyramid                	org.opencv.buildpyramid
    canny                       	org.opencv.canny  
    compare                     	org.opencv.compare
    convert_scale_abs           	org.opencv.convertscaleabs                      
    cvtcolor                    	org.opencv.cvtcolor                          
    dilate                      	org.opencv.dilate 
    distanceTransform           	org.opencv.distancetransform                                           
    divide                      	org.opencv.divide  
    erode                       	org.opencv.erode 
    FAST                        	org.opencv.fast
    fastNlMeansDenoising        	org.opencv.fastnlmeansdenoising
    fastNlMeansDenoisingColored 	org.opencv.fastnlmeansdenoisingcolored 
    filter2D                    	org.opencv.filter2d
    flip                        	org.opencv.flip 
    gaussianBlur                	org.opencv.gaussianblur
    goodFeature_detector        	org.opencv.good_features_to_track
    laplacian                   	org.opencv.laplacian
    medianBlur                  	org.opencv.medianblur
    morphologyEx                	org.opencv.morphologyex
    MSER                        	org.opencv.mser_detect 
    multiply                    	org.opencv.multiply    
    ORB                         	org.opencv.orb_detect
    ORB_Compute                 	org.opencv.orb_compute   
    pyrDown                     	org.opencv.pyrdown
    pyrUp                       	org.opencv.pyrup
    resize                      	org.opencv.resize
    scharr                      	org.opencv.scharr
    sepFilter2D                 	org.opencv.sepfilter2d
    SIFT_Compute                	org.opencv.sift_compute                         
    SIFT_Detect                 	org.opencv.sift_detect 
    simpleBlobDetector          	org.opencv.simple_blob_detect                   
    simpleBlobDetector_Init     	org.opencv.simple_blob_detect_initialize 
    sobel                       	org.opencv.sobel
    STAR_FEATURE_Detector       	org.opencv.star_detect  
    subtract                    	org.opencv.subtract
    SURF_Compute                	org.opencv.surf_compute
    SURF_Detect                 	org.opencv.surf_detect
    threshold                   	org.opencv.threshold  
    transpose                   	org.opencv.transpose                            
    warpAffine                  	org.opencv.warpaffine 
    warpPerspective             	org.opencv.warpperspective  

**NOTE** - For list of OpenVX API calls for OpenCV-interop refer include/[vx_ext_opencv.h](include/vx_ext_opencv.h)

## Build Instructions

### Pre-requisites

* AMD OpenVX&trade; library
* CMake `3.0` or later
* OpenCV `3`/`4` `with`/`without` **contrib**
    - **Note** For pre-built library: OpenCV_DIR environment variable should point to OpenCV/build folder

### Build using `Visual Studio 2019` on 64-bit Windows `10` / `11`

* Use amd_openvx_extensions/amd_opencv/amd_opencv.sln to build for x64 platform

### Build using CMake on Linux

* Use CMake to configure and generate Makefile

**NOTE:** OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
