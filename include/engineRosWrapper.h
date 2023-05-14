#include "engineTool.h"
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

class EngineRosWrapper {
public:
    EngineRosWrapper(ros::NodeHandle& nh, ros::NodeHandle& pnh, const Options& options);
    ~EngineRosWrapper();
    void callback_compressedImage(const sensor_msgs::CompressedImageConstPtr &msg);
    void callback_image(const sensor_msgs::ImageConstPtr& msg);
    jsk_recognition_msgs::BoundingBoxArray process(const cv::Mat &img);

private:
    EngineTool engineTool_;
    cudaStream_t inferenceCudaStream_;
    ros::Subscriber sub_compressedImage_;
    ros::Subscriber sub_image_;
    ros::Publisher pub_;
    // topics
    std::string inputTopic_;
    std::string outputTopic_;
    // parameters
    std::string onnxModelpath_;
    bool normalize_;
    int wantedClassNums_;
    float confidenceThreshold_;
    float nmsThreshold_;
    bool useCompressedImage_;
    size_t batchSize_;
    std::vector<std::vector<std::vector<float>>> featureVectors_;
};
