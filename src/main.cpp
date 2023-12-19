#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ostream>

#include "cmdline.h"
#include "detector.h"
#include "utils.h"

void Delay(int time) // time*1000为秒数
{
    clock_t now = clock();

    while (clock() - now < time)
        ;
}

int main(int argc, char *argv[]) {
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
    cmd.add<std::string>("image", 'i', "Image source to be detected.", false);
    cmd.add<std::string>("v4l2", 'v', "video dev node to be detected.", false);
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", true, "coco.names");
    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string imagePath = cmd.get<std::string>("image");
    const std::string videoPath = cmd.get<std::string>("v4l2");
    const std::string modelPath = cmd.get<std::string>("model_path");

    if (classNames.empty()) {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    if (imagePath.empty() && videoPath.empty()) {
        std::cerr << "At least give one source! jpg or /dev/videox"<<std::endl;
        return -1;
    }

    YOLODetector detector{nullptr};
    cv::Mat image;
    std::vector<Detection> result;

    try {
        detector = YOLODetector(modelPath, isGPU, cv::Size(640, 640));
        std::cout << "Model was initialized." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    if (!imagePath.empty()) {
        image = cv::imread(imagePath);
        result = detector.detect(image, confThreshold, iouThreshold);
        utils::visualizeDetection(image, result, classNames);
        cv::imshow("result", image);
        // cv::imwrite("result.jpg", image);
        cv::waitKey(0);
    } else if (!videoPath.empty()) {
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera." << std::endl;
            return -1;
        }

        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "Error: Could not read frame." << std::endl;
                continue;
            }

            auto start_time = std::chrono::high_resolution_clock::now();
            result = detector.detect(frame, confThreshold, iouThreshold);
            utils::visualizeDetection(frame, result, classNames);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Execution time: " << duration.count() << " ms." << std::endl;

            cv::imshow("result", frame);
            cv::waitKey(10);
            //	break;
        }
    }
    return 0;
}
