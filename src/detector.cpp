#include "detector.h"

Yolov5Detector::Yolov5Detector(const std::wstring& modelPath, const std::string& device = "gpu", const cv::Size& inputSize = cv::Size(640, 640))
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();
    // sessionOptions.SetIntraOpNumThreads(4);
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);

    if (device == "gpu" || device == "GPU" || device == "cuda" || device == "CUDA")
    {
        // TODO
    }
    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    inputDims = inputTensorInfo.GetShape();

    for (int64_t inputDim : inputDims)
        std::cout << "Input Dimensions: " << inputDim << std::endl;

    const char* inputName = session.GetInputName(0, allocator);
    const char* outputName = session.GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    outputDims = outputTensorInfo.GetShape();
    this->numClasses = (int)(outputDims.back()) - 5;
    this->inputImageSize = inputSize;

    inputNames.push_back(inputName);
    outputNames.push_back(outputName);

}

cv::Mat Yolov5Detector::preprocessing(cv::Mat &image)
{
    cv::Mat blob = cv::dnn::blobFromImage(image,
                                          1 / 255.,
                                          this->inputImageSize,
                                          cv::Scalar(0, 0, 0),
                                          true,
                                          false,
                                          CV_32F);
    return  blob;
}

Detection Yolov5Detector::postprocessing(cv::Mat& image, std::vector<float> &outputTensorValues,
                                         float confThreshold, float iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    float w_scale = (float)image.cols / (float)this->inputImageSize.width;
    float h_scale = (float)image.rows / (float)this->inputImageSize.height;

    auto getClassId = [=](std::vector<float>::iterator it){
        int idxMax = 5;
        float maxConf = 0;

        for (int i = 5; i < this->numClasses + 5; i++)
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

        if (confidence > confThreshold)
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
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<cv::Rect> finalBoxes;
    std::vector<float> finalConfs;
    std::vector<int> finalClassIds;
    for (int idx : indices)
    {
        finalBoxes.emplace_back(boxes[idx]);
        finalConfs.emplace_back(confs[idx]);
        finalClassIds.emplace_back((classIds[idx]));
    }

    return {finalBoxes, finalConfs, finalClassIds};

}

Detection Yolov5Detector::detect(cv::Mat &image, float confThreshold, float iouThreshold)
{
    cv::Mat blob = this->preprocessing(image);

    size_t inputTensorSize = utils::vectorProduct(inputDims);
    size_t outputTensorSize = utils::vectorProduct(outputDims);
    std::cout << "inputTensorSize: " << inputTensorSize << std::endl;
    std::cout << "outputTensorSize: " << outputTensorSize << std::endl;

    std::vector<float> inputTensorValues(inputTensorSize);
    std::vector<float> outputTensorValues(outputTensorSize);
    inputTensorValues.assign(blob.begin<float>(),
                             blob.end<float>());

    // TODO: vector of blobs for batch_size > 1, mb blobFromImages
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

    this->session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                      inputTensors.data(), 1, outputNames.data(),
                      outputTensors.data(), 1);

    Detection result = this->postprocessing(image, outputTensorValues, confThreshold, iouThreshold);
    return result;
}
