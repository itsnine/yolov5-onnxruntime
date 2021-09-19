#include <iostream>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "detector.h"


int main(int argc, char* argv[])
{
    std::cout << "args: " << argc << std::endl;
    const float conf_threshold = 0.5f;
    const float iou_threshold = 0.4f;
    std::wstring modelPath = L"yolov5m.onnx";
    std::string imagePath = "bus.jpg";
    std::string classNamesPath = "coco.names";

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();

    for (const std::string& provider : availableProviders)
        std::cout << provider << std::endl;

    if (argc == 4)
    {
        modelPath = utils::charToWstring(argv[1]);
        classNamesPath = argv[2];
        imagePath = argv[3];
    }

    std::vector<std::string> classNames = utils::loadNames(classNamesPath);

    Yolov5Detector detector(modelPath, "gpu", cv::Size(640, 640));

    cv::Mat image = cv::imread(imagePath);
    Detection result = detector.detect(image, conf_threshold, iou_threshold);

    utils::visualizeDetection(image, result, classNames);

    cv::imshow("result", image);

    cv::imwrite("result.jpg", image);
    cv::waitKey(0);

    //// Perfomance Test
    // int numTests{50};
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // for (int i = 0; i < numTests; i++) {
    //     session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
    //                 inputTensors.data(), 1, outputNames.data(),
    //                 outputTensors.data(), 1);
    // }
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Minimum Inference Latency: "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / static_cast<float>(numTests)
    //           << " ms" << std::endl;

    return 0;
}
