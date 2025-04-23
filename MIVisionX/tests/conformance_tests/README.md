# MIVisionX - OpenVX 1.3 Conformance Test

## Script to run tests

```
python runConformanceTests.py  --help
```

usage: 

```
runConformanceTests.py [--directory CTS_Build_Directory]
                       [--backend_type MIVisionX_Backend]

Arguments:
  -h, --help        Show this help message and exit
  --directory       Conformance build directory - optional (default:~/)
  --backend_type    Backend type - optional (default:ALL [options:ALL/HOST/HIP/OCL])
```