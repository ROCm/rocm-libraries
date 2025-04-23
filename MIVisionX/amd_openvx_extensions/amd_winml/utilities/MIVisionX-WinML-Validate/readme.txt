*************************************************************************************************************************************

                                              MIVisionX ONNX Model Validation - 1.0.0

*************************************************************************************************************************************

Usage:

        MIVisionX-WinML-Validate.exe [options]  --m <ONNX.model full path>
                                                --i <model input tensor name>
                                                --o <model output tensor name>
                                                --s <output tensor size in (n,c,h,w)>
                                                --l <label.txt full path>
                                                --f <image frame full path>
                                                --d <Learning Model Device Kind <DirectXHighPerformance>> [optional]

MIVisionX ONNX Model Validation Parameters

        --m/--model                     -- onnx model full path [required]
        --i/--inputName                 -- model input tensor name [required]
        --o/--outputName                -- model output tensor name [required]
        --s/--outputSize                -- model output tensor size <n,c,h,w> [required]
        --l/--label                     -- label.txt file full path [required]
        --f/--imageFrame                -- imageFrame.png file full path [required]
        --d/--deviceKind                -- Learning Model Device Kind <0-4> [optional]
                                         0 - Default
                                         1 - Cpu
                                         2 - DirectX
	                                     3 - DirectXHighPerformance
                                         4 - DirectXMinPower

MIVisionX ONNX Model Validation Options

        --h/--help      -- Show full help