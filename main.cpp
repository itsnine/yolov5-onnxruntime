#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


size_t vectorProduct(std::vector<int64> vector)
{
    if (vector.empty())
        return -1;

    size_t product = 1;
    for (int64 element : vector)
        product *= element;

    return product;
}


int main(int argc, char* argv[])
{
    const float conf_threshold = 0.5f;
    const float iou_threshold = 0.4f;
    std::wstring modelPath = L"yolov5m.onnx";
    std::string imagePath = "./bus.jpg";
    int numClasses = 80;
    if (argc > 1)
        imagePath = argv[1];

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    Ort::SessionOptions sessionOptions;
    // sessionOptions.SetIntraOpNumThreads(4);

    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    const char* inputName = session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;

    for (int i = 0; i < numOutputNodes; i++)
    {
        std::cout << i << ": " << session.GetOutputName(i, allocator) << std::endl;
    }

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

    for (int64_t inputDim : inputDims)
        std::cout << "Input Dimensions: " << inputDim << std::endl;

    const char* outputName = session.GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    for (int64_t outputDim : outputDims)
        std::cout << "Output Dimensions: " << outputDim << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    cv::Mat image = cv::imread(imagePath);
    cv::Mat blob = cv::dnn::blobFromImage(image,
                                          1 / 255.,
                                          cv::Size(640, 640),
                                          cv::Scalar(0, 0, 0),
                                          true,
                                          false,
                                          CV_32F);
    std::cout << "blob size: " << blob.size << std::endl;

    size_t inputTensorSize = vectorProduct(inputDims);
    size_t outputTensorSize = vectorProduct(outputDims);
    std::cout << "inputTensorSize: " << inputTensorSize << std::endl;
    std::cout << "outputTensorSize: " << outputTensorSize << std::endl;

    std::vector<float> inputTensorValues(inputTensorSize);
    std::vector<float> outputTensorValues(outputTensorSize);
    inputTensorValues.assign(blob.begin<float>(),
                             blob.end<float>());

    // TODO: vector of blobs for batch_size > 1, mb blobFromImages

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputDims.data(), inputDims.size()
    ));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()
    ));

    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

    std::cout << "outputTensorValues.size: " << outputTensorValues.size() << std::endl;

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    float w_scale = image.cols / 640.f;
    float h_scale = image.rows / 640.f;

    auto getClassId = [numClasses](std::vector<float>::iterator it){
        int idxMax = 5;
        float maxConf = 0;

        for (int i = 5; i < numClasses + 5; i++)
        {
            if (it[i] > maxConf)
            {
                maxConf = it[i];
                idxMax = i - 5;
            }
        }
        return idxMax;
    };

    for (auto it = outputTensorValues.begin(); it != outputTensorValues.end(); it += (numClasses + 5))
    {
        float confidence = it[4];

        if (confidence > conf_threshold)
        {
            int centerX = (int)(it[0] * w_scale);
            int centerY = (int)(it[1] * h_scale);
            int width = (int)(it[2] * w_scale);
            int height = (int)(it[3] * h_scale);
            int left = centerX - width / 2;
            int top = centerY - height / 2;
            int classId = getClassId(it);

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, conf_threshold, iou_threshold, indices);
    std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    for (int idx : indices)
    {
        std::cout << "Conf: " << confs[idx] << "; classId: " << classIds[idx] << std::endl;
        cv::rectangle(image, boxes[idx], cv::Scalar(255, 255, 0), 2);
    }


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



    cv::Mat resized;
    cv::resize(image, resized, cv::Size(), 0.65, 0.65);

    cv::imshow("result", resized);
    cv::imwrite("result.jpg", image);
    cv::waitKey(0);

    return 0;
}
