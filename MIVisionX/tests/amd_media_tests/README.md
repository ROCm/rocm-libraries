# AMD Media Tests

## Decoder Test - `gdf/decoder.gdf`

```
runvx -affinity:GPU -frames:0,eof file gdf/decoder.gdf
```

**NOTE:** Make sure to edit `gdf/decoder.gdf` and add the full path to `${FULL_PATH}/MIVisionX/data/videos/AMD_driving_virtual_20.mp4`

**Sample expected output:**
```
INFO: reading 1920x1080 into slice#0 from XXXXXXXXX/MIVisionX/data/videos/AMD_driving_virtual_20.mp4
csv,HEADER ,STATUS, COUNT,cur-ms,avg-ms,min-ms,clenqueue-ms,clwait-ms,clwrite-ms,clread-ms
csv,OVERALL,  PASS,     0,      ,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00 (median 0.000)
> total elapsed time:   0.00 sec<br>
```

## Encoder Test - `gdf/encoder.gdf`

```
runvx -affinity:GPU file gdf/encoder.gdf
```

**Sample Expected output:**
```
INFO: writing 1280x720 4.00mbps 30.00fps gopsize=15 bframes=0 video into output.264
csv,HEADER ,STATUS, COUNT,cur-ms,avg-ms,min-ms,clenqueue-ms,clwait-ms,clwrite-ms,clread-ms
csv,OVERALL,  PASS,     1,      ,  6.26,  6.26,  0.00,  0.00,  0.00,  0.00 (median 6.256)
```