#include <iostream>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "detector.h"


int main(int argc, char* argv[])
{
    std::cout << "args: " << argc << std::endl;
    const float confThreshold = 0.5f;
    const float iouThreshold = 0.4f;
    std::wstring modelPath = L"yolov5m.onnx";
    std::string imagePath = "bus.jpg";
    std::string classNamesPath = "coco.names";

    if (argc == 4)
    {
        modelPath = utils::charToWstring(argv[1]);
        classNamesPath = argv[2];
        imagePath = argv[3];
    }

    std::vector<std::string> classNames = utils::loadNames(classNamesPath);

    Yolov5Detector detector(modelPath, "gpu", cv::Size(640, 640));

    cv::Mat image = cv::imread(imagePath);
    Detection result = detector.detect(image, confThreshold, iouThreshold);

    utils::visualizeDetection(image, result, classNames);

    cv::imshow("result", image);

    cv::imwrite("result.jpg", image);
    cv::waitKey(0);


    return 0;
}
