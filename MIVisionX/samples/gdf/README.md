## GDF - Graph Description Format

MIVisionX samples using [RunVX](https://github.com/ROCm/MIVisionX/tree/master/utilities/runvx#amd-runvx)

**Note:** 

* To run the samples we need to put MIVisionX executables and libraries into the system path

``` 
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
```

* To get help on RunVX, use `-h` option

``` 
runvx -h
```

### skintonedetect.gdf

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/skinToneDetect_image.PNG" /></p>

usage:

``` 
runvx skintonedetect.gdf
```

### canny.gdf

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/samples/images/canny_image.PNG" /></p>

usage:

``` 
runvx canny.gdf
```

### skintonedetect-LIVE.gdf

Using a live camera

usage:

``` 
runvx -frames:live skintonedetect-LIVE.gdf
```

### canny-LIVE.gdf

Using a live camera

usage:

``` 
runvx -frames:live canny-LIVE.gdf
```

### OpenCV_orb-LIVE.gdf

Using a live camera

usage:

``` 
runvx -frames:live OpenCV_orb-LIVE.gdf
```
