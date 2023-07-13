#include "engineRosWrapper.h"


int main(int argc, char** argv)
{
    ros::init(argc, argv, "engine_ros_wrapper");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~"); // private node handle

    bool halfPrecision_;
    bool doesSupportDynamicBatchSize_;
    int maxWorkspaceSize_;

    pnh.param<bool>("half_precision", halfPrecision_, true);
    pnh.param<bool>("does_support_dynamic_batch_size", doesSupportDynamicBatchSize_, false);
    pnh.param<int>("max_workspace_size", maxWorkspaceSize_, 2000000000); // 2GB

    // Specify our GPU inference configuration options
    Options options;
    options.doesSupportDynamicBatchSize = doesSupportDynamicBatchSize_;
    if (halfPrecision_)
        options.precision = Precision::FP16; // Use fp16 precision for faster inference.
    else
        options.precision = Precision::FP32;
    options.maxWorkspaceSize = maxWorkspaceSize_;
    if (options.doesSupportDynamicBatchSize)
    {
        options.optBatchSize = 4;
        options.maxBatchSize = 16;
    }
    else
    {
        options.optBatchSize = 1;
        options.maxBatchSize = 1;
    }

    EngineRosWrapper engineRosWrapper(nh, pnh, options);
    ros::spin();
    return 0;
}