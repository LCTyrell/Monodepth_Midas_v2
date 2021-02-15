# Openvino Midas v2.1

## Setup

The model need Intel's OpenVino ToolKit to be installed (version 2021.1).

## Demo
The models provided in the models_IR directory are in FP16 (optimized for NCS2)  
  
* On CPU:  
`python3 midas_v21s.py -m models_IR/midas-small_256.xml -i demo.mp4`  
  
* On the NCS2 (MyriadX):  
`python3 midas_v21s.py -m models_IR/midas-small_256.xml -i demo.mp4 -d MYRIAD`  
