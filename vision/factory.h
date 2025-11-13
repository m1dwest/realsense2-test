#pragma once

#include "detector.h"

namespace vision {

enum class ModelType { YOLOv5, YOLOv8 };

ModelRuntime make_runtime(ModelType model_type, const std::string& model_path,
                          const std::string& labels_path, int input_w,
                          int input_h, cv::Scalar letterbox_color);

}  // namespace vision
