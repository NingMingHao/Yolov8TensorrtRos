# Yolov8TensorrtRos

## Instruction for Applying Yolov8 on ROS with Tensorrt acceleration
### Get Yolov8 onnx model using [Ultralytics](https://github.com/ultralytics/ultralytics)
* Directly coverting from pt to engine may have some problems, so I choose to convert it to onnx, and then compile in C++.
* [Export the official pt model](https://docs.ultralytics.com/usage/cfg/#export)
  `yolo export model=yolov8s.pt format=onnx device=0 imgsz=640 simplify=True`  
* You may visualize the onnx model in [netron](https://netron.app/)
### Compile it in C++ node
* This part is modified based on [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api/tree/main)
* What I have done it to add a postprocess of the output (Nx84x8400) of the YOLOv8 
  * Remove unnecessary classes, only keeps the 0-16
    1: bicycle
    2: car
    3: motorcycle
    4: airplane
    5: bus
    6: train
    7: truck
    8: boat
    9: traffic light
    10: fire hydrant
    11: stop sign
    12: parking meter
    13: bench
    14: bird
    15: cat
    16: dog
  * The NMS for each class
  * Set the parameters in roslaunch
    * topics
    * conf_threshold
    * IOU_threshold
    * GPU_workspace (bytes)

## Current environment
Jetson Orin (Jetpack 5.0.2), CUDA, cuDNN, OPENCV are included
