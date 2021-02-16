# Openvino Midas v2.1

## Installation

Install Intel's OpenVino ToolKit 2021.1.  
  
The model provided in the models_IR directory is in FP16 (optimized for NCS2).  
If you need a different data type, use:  

```
python3 <Path to Openvino>/deployment_tools/model_optimizer/mo.py \
  --input_model model-small.onnx \
  --input 0 \
  --input_shape [1,3,256,256] \
  --data_type FP32
```

## Usage 
  
* On CPU:  
`python3 midas_v21s.py -m models_IR/midas-small_256.xml -i demo.mp4`  
  
* On the NCS2 (MyriadX):  
`python3 midas_v21s.py -m models_IR/midas-small_256.xml -i demo.mp4 -d MYRIAD` 

#### Command line arguments:

    -h : Help
    -m : Required. Path to an .xml file with a trained model.
    -i : Required. Path to video file or image. 'cam' for capturing video stream from camera.
    -l : Optional. Required for CPU custom layers. Absolute path to a shared library with the kernels implementations.
    -d : Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable.
