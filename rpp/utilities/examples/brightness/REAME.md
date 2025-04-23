# Sample example to run brightness augmentation 

*   To run this example
```none
mkdir build
cd build
cmake ../
make -j
./brightness_hip <src1 folder> <src2 folder> <dst folder> <u8 = 0/ f16 = 1/ f32 = 2/ u8->f16 = 3/ u8->f32 = 4/ i8 = 5/ u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0/ pkd->pln = 1)> <layout type (pkd3 - 0/pln3 - 1/pln1 - 2)> <decoder type (TurboJPEG - 0/ OpenCV - 1)>
```

*   Note: src2 folder -place same as src1 folder for single image functionalities
*   Example:
```none
./brightness_hip ../../../test_suite/TEST_IMAGES/three_images_mixed_src1/ ../../../test_suite/TEST_IMAGES/three_images_mixed_src2/ output_images/ 0 0 0 0
```
