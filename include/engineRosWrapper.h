#include <fstream>
#include "engineTool.h"
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include "autoware_perception_msgs/DynamicObjectArray.h"
#include "autoware_perception_msgs/DynamicObjectWithFeatureArray.h"
#include "autoware_perception_msgs/Feature.h"
#include "autoware_perception_msgs/Semantic.h"

#define __APP_NAME__ "tensorrt_yolov8"

class EngineRosWrapper {
public:
    EngineRosWrapper(ros::NodeHandle& nh, ros::NodeHandle& pnh, const Options& options);
    ~EngineRosWrapper();
    void callback_compressedImage(const sensor_msgs::CompressedImageConstPtr &msg);
    void callback_image(const sensor_msgs::ImageConstPtr& msg);
    void timerCallback(const ros::TimerEvent&);
    autoware_perception_msgs::DynamicObjectWithFeatureArray process(const cv::Mat &img);
    bool readLabelFile(const std::string & filepath, std::vector<std::string> * labels);

private:
    std::vector<std::string> labels_;
    EngineTool engineTool_;
    cudaStream_t inferenceCudaStream_;
    ros::Subscriber sub_compressedImage_;
    ros::Subscriber sub_image_;
    ros::Subscriber sub_camera_info_;
    ros::Publisher publisher_obj_;
    ros::Publisher publisher_img_overlay_;
    // topics
    std::string inputTopic_;
    std::string outputTopic_;
    std::string labelFile_;
    std::string cameraInfoTopic_;
    // parameters
    std::string onnxModelpath_;
    bool normalize_;
    int wantedClassNums_;
    float confidenceThreshold_;
    float nmsThreshold_;
    double operateRate_;
    bool useCompressedImage_;
    size_t batchSize_;
    std::vector<std::vector<std::vector<float>>> featureVectors_;

    void IntrinsicsCallback(const sensor_msgs::CameraInfo& in_message);
    cv::Size                            image_size_;
    cv::Mat                             camera_intrinsics_;
    cv::Mat                             distortion_coefficients_;
    cv::Mat                             current_frame_;  
    bool                                camera_info_ok_;  

    ros::Timer timer_;
    cv::Mat latest_image_;
    ros::Time latest_image_time_;
    ros::Time last_processed_image_time_;    
};
