#include "backend.h"

#include <fstream>

namespace vision {

cv::dnn::Net DetectorBackend::load_model(const std::string& path) {
    auto net = cv::dnn::readNetFromONNX(path);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return net;
}

std::vector<std::string> DetectorBackend::load_names(const std::string& path) {
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

}  // namespace vision
