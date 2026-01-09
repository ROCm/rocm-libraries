# Formocast: A hardware simulation-based prediction model for tensilelite GEMM algorithms

Formocast is a prediction model which forecast the performance of tensilelite solutions.
An example is in path: /rocm-libraries_p/projects/hipblaslt/tensilelite/formocast/example/BBSTN.yaml
The parameter PredictionThreshold is to control the number of solutions to be run when tuning.

PredictionThreshold is from 0.0~1.0

**Usage**

### build the tensilelite client and run with BBSTN.yaml. Check the log and the forecast result.
1. Go to the tensilelite folder.
2. Run command "invoke build-client"
3. Run command "./build_tmp/Tensile.sh formocast/example/BBSTN.yaml build"
4. Check the log. It should show the selected solution only.

### To run an tf32 example with hipblaslt-bench,
1. Go to the hipblaslt folder.
2. Run command "./install.sh -c -a gfx950 --skip_rocroller --logic-yaml-filter *gfx950/**/*_S_MX_*"
3. Run command "TENSILE_PREDICTION_ALGO=1 TENSILE_PREDICTION_LIB=1 ./hipblaslt-bench --yaml ./tensilelite/tests/tf32_tt_test.yaml"

Note that this 2 examples are the functionality purpose. For better performance, we need to investigate more about the gfx950 platform, find a better solution pool, and optimize the selection time.