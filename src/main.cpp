#include "engine.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <ros/ros.h>

typedef std::chrono::high_resolution_clock Clock;

int main() {
    // Initialize ROS
    ros::init(argc, argv, "yolov8_tensorrt_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    // Read parameters
    std::string onnxModelpath;
    std::bool normalize;
    std::bool halfPrecision;
    std::uint maxWorkspaceSize;
    std::bool doesSupportDynamicBatchSize;
    pnh.param<std::string>("onnx_model_path", onnxModelpath, "../models/yolov8s.onnx");
    pnh.param<bool>("normalize", normalize, true);
    pnh.param<bool>("half_precision", halfPrecision, true);
    pnh.param<std::uint>("max_workspace_size", maxWorkspaceSize, 2000000000); // 2GB
    pnh.param<bool>("does_support_dynamic_batch_size", doesSupportDynamicBatchSize, false);


    // Specify our GPU inference configuration options
    Options options;
    // TODO: If your model only supports a static batch size
    options.doesSupportDynamicBatchSize = doesSupportDynamicBatchSize;
    if (halfPrecision)
        options.precision = Precision::FP16; // Use fp16 precision for faster inference.
    options.maxWorkspaceSize = maxWorkspaceSize; 

    if (options.doesSupportDynamicBatchSize) {
        options.optBatchSize = 4;
        options.maxBatchSize = 16;
    } else {
        options.optBatchSize = 1;
        options.maxBatchSize = 1;
    }

    Engine engine(options);

    // Build the engine
    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // Let's use a batch size which matches that which we set the Options.optBatchSize option
    size_t batchSize = options.optBatchSize;

    const std::string inputImage = "../inputs/bus.png";
    auto cpuImg = cv::imread(inputImage);
    if (cpuImg.empty()) {
        throw std::runtime_error("Unable to read image at path: " + inputImage);
    }

    // The model expects RGB input
    cv::cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);

    // Upload to GPU memory
    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // Populate the input vectors
    const auto& inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    // TODO:
    // For the sake of the demo, we will be feeding the same image to all the inputs
    // You should populate your inputs appropriately.
    for (const auto & inputDim : inputDims) {
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize; ++j) {
            cv::cuda::GpuMat resized;
            // TODO:
            // You can choose to resize by scaling, adding padding, or a combination of the two in order to maintain the aspect ratio
            // You can use the Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while maintain the aspect ratio (adds padding where necessary to achieve this).
            // If you are running the sample code using the suggested model, then the input image already has the correct size.
            // The following resizes without maintaining aspect ratio so use carefully!
            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2], inputDim.d[1])); // TRT dims are (height, width) whereas OpenCV is (width, height)
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }


    // Discard the first inference time as it takes longer
    std::vector<std::vector<std::vector<float>>> featureVectors;
    succ = engine.runInference(inputs, featureVectors, normalize);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }
    // print the feature vectors shape
    std::cout << "Feature vectors shape: " << featureVectors.size() << " x " << featureVectors[0].size() << " x " << featureVectors[0][0].size() << std::endl;

    size_t numIterations = 10000;

    // Benchmark the inference time
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIterations; ++i) {
        featureVectors.clear();
        engine.runInference(inputs, featureVectors, normalize);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Success! Average time per inference: " << totalTime / numIterations / static_cast<float>(inputs[0].size()) <<
    " ms, for batch size of: " << inputs[0].size() << std::endl;

    // Print the feature vectors
    for (size_t batch = 0; batch < featureVectors.size(); ++batch) {
        for (size_t outputNum = 0; outputNum < featureVectors[batch].size(); ++outputNum) {
            std::cout << "Batch " << batch << ", " << "output " << outputNum << std::endl;
            int i = 0;
            for (const auto &e:  featureVectors[batch][outputNum]) {
                std::cout << e << " ";
                if (++i == 10) {
                    std::cout << "...";
                    break;
                }
            }
            std::cout << "\n" << std::endl;
        }
    }

    // TODO: If your model requires post processing (ex. convert feature vector into bounding boxes) then you would do so here.

    return 0;
}
