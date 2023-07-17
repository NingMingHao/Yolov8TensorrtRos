#include "engineRosWrapper.h"
#include <cv_bridge/cv_bridge.h>
#include <chrono>
#include <opencv2/dnn.hpp>
#include <ros/package.h>

typedef std::chrono::steady_clock Clock;

EngineRosWrapper::EngineRosWrapper(ros::NodeHandle &nh, ros::NodeHandle &pnh, const Options &options):engineTool_(options)
{
    // Read parameters
    // Get ROS parameters
    ros::NodeHandle pnh_("~"); // private node handle
    std::string onnxName;
    pnh.param<std::string>("onnx_model_name", onnxName, "yolov8s.onnx");
    std::string package_path = ros::package::getPath("tensorrt_yolov8");
    onnxModelpath_ = package_path + "/models/" + onnxName;
    pnh.param<bool>("normalize", normalize_, true);
    pnh.param<int>("wanted_class_nums", wantedClassNums_, 80);
    pnh.param<float>("confidence_threshold", confidenceThreshold_, 0.5);
    pnh.param<float>("nms_threshold", nmsThreshold_, 0.4);
    pnh.param<std::string>("input_topic", inputTopic_, "/pylon_camera_node_center/image_rect/compressed");
    pnh.param<std::string>("output_topic", outputTopic_, "/bbox_array_center");
    batchSize_ = options.optBatchSize;

    // judge useCompressedImage_ based on inputTopic_
    if (inputTopic_.find("compressed") != std::string::npos) {
        useCompressedImage_ = true;
    } else {
        useCompressedImage_ = false;
    }

    // Build the engine
    bool succ = engineTool_.build(onnxModelpath_);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }
    succ = engineTool_.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    // subscribe to input topic
    if (useCompressedImage_) {
        sub_compressedImage_ = nh.subscribe(inputTopic_, 1, &EngineRosWrapper::callback_compressedImage, this);
    } else {
        sub_image_ = nh.subscribe(inputTopic_, 1, &EngineRosWrapper::callback_image, this);
    }
    // advertise output topic
    pub_ = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>(outputTopic_, 1);

    cudaError_t err = cudaStreamCreate(&inferenceCudaStream_);
    if (err != 0) {
        throw std::runtime_error("Unable to create inference cuda stream.");
    }
}

EngineRosWrapper::~EngineRosWrapper() {
    cudaStreamDestroy(inferenceCudaStream_);
}

void EngineRosWrapper::callback_compressedImage(const sensor_msgs::CompressedImageConstPtr& msg) {
    ROS_INFO("callback_compressedImage");
    auto start_time = Clock::now();
    // convert to cv::Mat
    cv::Mat cvImg = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    //log the time
    auto img_decode_time = Clock::now();
    auto img_decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(img_decode_time - start_time).count();
    std::cout << "img_decode_time: " << img_decode_duration << " ms" << std::endl;
    
    // process
    jsk_recognition_msgs::BoundingBoxArray bboxarry = process(cvImg);
    bboxarry.header = msg->header;
    pub_.publish(bboxarry);
    auto end_time = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "all run time: " << duration << " ms" << std::endl;
}

void EngineRosWrapper::callback_image(const sensor_msgs::ImageConstPtr& msg) {
    ROS_INFO("callback_rawImage");
    auto start_time = Clock::now();
    // convert to cv::Mat
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    // process
    jsk_recognition_msgs::BoundingBoxArray bboxarry = process(cv_ptr->image);
    bboxarry.header = msg->header;
    pub_.publish(bboxarry);
    auto end_time = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "all run time: " << duration << " ms" << std::endl;
}

jsk_recognition_msgs::BoundingBoxArray EngineRosWrapper::process(const cv::Mat &cpuImg)
{
    auto start_time = Clock::now();
    // preprocess
    cv::cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);
    // convert to GPU Mat
    cv::cuda::GpuMat img;
    img.upload(cpuImg);
    // Populate the input vectors
    const auto &inputDims = engineTool_.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;
    for (const auto &inputDim : inputDims)
    {
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize_; ++j)
        {
            cv::cuda::GpuMat resized;
            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2], inputDim.d[1])); // TRT dims are (height, width) whereas OpenCV is (width, height)
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }
    // calculate the scale factor
    float height_scale = (float)cpuImg.rows / inputDims[0].d[2];
    float width_scale = (float)cpuImg.cols / inputDims[0].d[1];
    //log the time
    auto preprocess_time = Clock::now();
    auto preprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_time - start_time).count();
    std::cout << "preprocess time: " << preprocess_duration << " ms" << std::endl;

    // inference
    featureVectors_.clear();
    engineTool_.runInference(inputs, featureVectors_, normalize_, inferenceCudaStream_);
    //log the time
    auto inference_time = Clock::now();
    auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_time - preprocess_time).count();
    std::cout << "inference time: " << inference_duration << " ms" << std::endl;

    // post process NMS, the featureVectors_ is Batchx1x(84x8400)
    // only use the first batch, and convert to vector of cv::Mat
    // we will have 8400 cv::Mat, for for each Mat, we only keep the first 4+wantedClassNums_ elements
    int rows = 84;
    int cols = 8400;
    int wantedRows = 4 + wantedClassNums_;
    if (featureVectors_[0][0].size() != rows * cols || wantedRows > rows)
    {
        throw std::runtime_error("Invalid input feature vector dimensions or wanted class nums.");
    }
    
    cv::Mat detection_mat(rows, cols, CV_32FC1);
    std::memcpy(detection_mat.data, featureVectors_[0][0].data(), rows * cols * sizeof(float));

    // filter out the low confidence boxes
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < cols; i++)
    {   cv::Mat oneDetection = detection_mat.col(i);
        cv::Mat scores = oneDetection.rowRange(4, wantedRows);
        cv::Point classIdPoint;
        double classProb;
        cv::minMaxLoc(scores, 0, &classProb, 0, &classIdPoint);

        if (classProb > confidenceThreshold_)
        {
            int centerX = (int)(oneDetection.at<float>(0) * width_scale);
            int centerY = (int)(oneDetection.at<float>(1) * height_scale);
            int width = (int)(oneDetection.at<float>(2) * width_scale);
            int height = (int)(oneDetection.at<float>(3) * height_scale);
            int left = centerX - width / 2;
            int top = centerY - height / 2;
            classIds.push_back(classIdPoint.y);
            confidences.push_back((float)classProb);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }
    // NMS for each class
    std::map<int, std::vector<size_t>> class2indices;
    for (size_t i = 0; i < classIds.size(); i++)
    {
        class2indices[classIds[i]].push_back(i);
    }

    std::vector<cv::Rect> nmsBoxes;
    std::vector<float> nmsConfidences;
    std::vector<int> nmsClassIds;
    for (std::map<int, std::vector<size_t>>::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
    {
        std::vector<cv::Rect> localBoxes;
        std::vector<float> localConfidences;
        std::vector<size_t> classIndices = it->second;
        for (size_t i = 0; i < classIndices.size(); i++)
        {
            localBoxes.push_back(boxes[classIndices[i]]);
            localConfidences.push_back(confidences[classIndices[i]]);
        }
        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxes(localBoxes, localConfidences, confidenceThreshold_, nmsThreshold_, nmsIndices);
        for (size_t i = 0; i < nmsIndices.size(); i++)
        {
            size_t idx = nmsIndices[i];
            nmsBoxes.push_back(localBoxes[idx]);
            nmsConfidences.push_back(localConfidences[idx]);
            nmsClassIds.push_back(it->first);
        }
    }

    //create the bboxArray message
    jsk_recognition_msgs::BoundingBoxArray bboxArray;
    for (size_t i = 0; i < nmsBoxes.size(); i++)
    {
        jsk_recognition_msgs::BoundingBox bbox;
        bbox.pose.position.x = nmsBoxes[i].x;
        bbox.pose.position.y = nmsBoxes[i].y;
        bbox.dimensions.x = nmsBoxes[i].width;
        bbox.dimensions.y = nmsBoxes[i].height;
        bbox.value = nmsConfidences[i];
        bbox.label = nmsClassIds[i];
        bboxArray.boxes.push_back(bbox);
    }

    //log the time
    auto postprocess_time = Clock::now();
    auto postprocess_duration = std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_time - inference_time).count();
    std::cout << "postprocess time: " << postprocess_duration << " ms" << std::endl;
    return bboxArray;
}