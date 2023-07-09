#include <fstream>
#include "engineTool.h"
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include "autoware_perception_msgs/DynamicObjectArray.h"
#include "autoware_perception_msgs/DynamicObjectWithFeatureArray.h"
#include "autoware_perception_msgs/Feature.h"
#include "autoware_perception_msgs/Semantic.h"

#include <memory>
#include <boost/circular_buffer.hpp>

#define __APP_NAME__ "tensorrt_yolov8"

class EngineRosWrapper {
public:
    EngineRosWrapper(ros::NodeHandle& nh, ros::NodeHandle& pnh, const Options& options);
    ~EngineRosWrapper();
    void timerCallback(const ros::TimerEvent&);
    autoware_perception_msgs::DynamicObjectWithFeatureArray process(const cv::Mat &img, const image_transport::Publisher &image_pub);
    bool readLabelFile(const std::string & filepath, std::vector<std::string> * labels);

private:
    std::vector<std::string> labels_;
    EngineTool engineTool_;
    cudaStream_t inferenceCudaStream_;
    // sub and pub
    std::vector<std::shared_ptr<ros::Subscriber> > v_camera_info_sub_;
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    std::vector<image_transport::Subscriber> image_subs_;
    std::vector<image_transport::Publisher> image_pubs_; 
    std::vector<ros::Publisher> publisher_obj_;
    // topics
    std::string labelFile_;
    // parameters
    std::string onnxModelpath_;
    bool normalize_;
    int wantedClassNums_;
    float confidenceThreshold_;
    float nmsThreshold_;
    double operateRate_;
    int camera_num_;
    size_t batchSize_;
    std::vector<std::vector<std::vector<float>>> featureVectors_;

    void IntrinsicsCallback(const sensor_msgs::CameraInfoConstPtr & in_message, const int id);
    void imageCallback(const sensor_msgs::ImageConstPtr & input_image_msg, const int id);
    std::vector<boost::circular_buffer<sensor_msgs::ImageConstPtr>> image_buffers_;
    std::map<int, sensor_msgs::CameraInfo> m_camera_info_;
    std::map<int, cv::Size> image_size_;
    std::map<int, cv::Mat> camera_intrinsics_;
    std::map<int, cv::Mat> distortion_coefficients_;
    std::map<int, bool> camera_info_ok_;

    ros::Timer timer_;
};
