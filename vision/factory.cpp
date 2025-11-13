#include "factory.h"

#include <fstream>

#include "parsers/yolov5.h"
#include "parsers/yolov8.h"

namespace vision {

cv::dnn::Net load_model(const std::string& path) {
    auto net = cv::dnn::readNetFromONNX(path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return net;
}

std::vector<std::string> load_labels(const std::string& path) {
    auto ifs = std::ifstream{path};

    std::vector<std::string> result;
    std::string line;

    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            result.push_back(line);
        }
    }

    if (result.empty()) {
        throw std::runtime_error{
            "Failed to load class names from file: " + path + "\n"};
    }

    return result;
}

ModelRuntime make_runtime(ModelType model_type, const std::string& model_path,
                          const std::string& labels_path, int input_w,
                          int input_h, cv::Scalar letterbox_color) {
    const auto create_parser = [](auto type) -> std::unique_ptr<Parser> {
        switch (type) {
            case ModelType::YOLOv5:
                return std::make_unique<YOLOv5Parser>();
            case ModelType::YOLOv8:
                return std::make_unique<YOLOv8Parser>();
            default:
                throw std::runtime_error("Unknown model type");
        }
    };

    return ModelRuntime{.net = load_model(model_path),
                        .labels = load_labels(labels_path),
                        .parser = create_parser(model_type),
                        .input_w = input_w,
                        .input_h = input_h,
                        .letterbox_color = letterbox_color};
}

}  // namespace vision
