# yolov5-onnxruntime

C++ YOLO v5 ONNXRuntime inference code. `Work in progress`...

Dependecies: OpenCV 4.5.2, ONNXRuntime 1.8.1. (Tested only on Windows 10)

To run from CLI:
```bash
# yolov5_ort.exe path_to_onnx_model path_to_image
yolov5_ort.exe yolov5m.onnx bus.jpg
```

<a href="images/bus_result.jpg"><img src="images/bus_result.jpg" style="width:60%; height:60%;"/></a>

## TODO
- refactoring;
- add C++ letterbox implementation and scaling;
- read class names from file;
- better visualization with class names and boxes;
- create YOLO class for easy deployment; 
- add Python implementation of the project.

## References
- YOLO v5 repo: https://github.com/ultralytics/yolov5
