# yolov5-onnxruntime

C++ YOLO v5 ONNXRuntime inference code. `Work in progress`...

Dependecies: CMake 3.16, OpenCV 4.5.2, ONNXRuntime 1.8.1. (Tested on Windows 10 and Ubuntu 20.04).

## Build
Before building the project you should manually change `ONNXRUNTIME_DIR` in `CMakeLists.txt` to your ONNXRuntime path, i.e. `"C:/onnxruntime-win-x64-1.8.1"` after that you should build with below command:

```bash
cmake --build .
```

## Run
To run the executable you should add OpenCV and ONNXRuntime dll's (.so) to your environment path `or` put all needed dll's (.so) near the executable.

Run from CLI:
```bash
# On Windows
# yolov5_ort.exe path_to_onnx_model path_to_class_names path_to_image
yolov5_ort.exe yolov5m.onnx coco.names bus.jpg

# On Linux
# ./yolov5_ort path_to_onnx_model path_to_class_names path_to_image
./yolov5_ort yolov5m.onnx coco.names bus.jpg
```

<p align="center">
  <a href="images/bus_result.jpg"><img src="images/bus_result.jpg" style="width:60%; height:60%;"/></a>
</p>


## TODO
- refactoring;
- add C++ letterbox implementation and scaling;
- ~~add Linux compatibility~~;
- ~~read class names from file~~;
- ~~better visualization with class names and boxes~~;
- ~~create YOLO class for easy deployment~~; 
- add Python implementation of the project.

## References
- YOLO v5 repo: https://github.com/ultralytics/yolov5
- YOLOv5 Runtime Stack repo: https://github.com/zhiqwang/yolov5-rt-stack
- ONNXRuntime Inference examples: https://github.com/microsoft/onnxruntime-inference-examples
