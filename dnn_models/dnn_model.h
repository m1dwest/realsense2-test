#pragma once

#include <fstream>
#include <optional>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

class DNNModel {
    cv::dnn::Net _net;
    std::optional<cv::Mat> _outputs;

   protected:
    struct Letterbox {
        float aspect_ratio;
        int image_y, image_x, image_w, image_h;
        int src_w, src_h;

        cv::Mat data;
    };
    Letterbox _letterbox;

    Letterbox get_letterbox(const cv::Mat& src, int new_w, int new_h,
                            const cv::Scalar& border_color) const {
        const int src_w = src.cols;
        const int src_h = src.rows;
        const float aspect_ratio = std::min(static_cast<float>(new_w) / src_w,
                                            static_cast<float>(new_h) / src_h);

        const int image_w = static_cast<int>(std::round(src_w * aspect_ratio));
        const int image_h = static_cast<int>(std::round(src_h * aspect_ratio));
        const int image_x = (new_w - image_w) / 2;
        const int image_y = (new_h - image_h) / 2;

        cv::Mat resized;
        cv::resize(src, resized, cv::Size(image_w, image_w));

        cv::Mat dst;
        cv::copyMakeBorder(resized, dst, image_y, new_h - image_h - image_y,
                           image_x, new_w - image_w - image_x,
                           cv::BORDER_CONSTANT, border_color);
        return {aspect_ratio, image_y, image_x, image_w,
                image_h,      src_w,   src_h,   dst};
    }

    cv::dnn::Net load_model(const std::string& path) const {
        auto net = cv::dnn::readNetFromONNX(path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        return net;
    }

    std::vector<std::string> load_names(const std::string& path) const {
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

    void input_preprocessed(cv::Mat blob) { _net.setInput(std::move(blob)); }

   public:
    struct Object {
        std::string class_name;
        float score;
        cv::Rect box;
    };

    explicit DNNModel(const std::string& model_path,
                      const std::string& names_path) {
        _net = this->load_model(model_path);
        _class_names = this->load_names(names_path);
    }

    virtual void input(const cv::Mat& color_bgr) = 0;

    void forward() { _outputs = _net.forward(); }
    bool has_outputs() const { return _outputs.has_value(); }
    const cv::Mat& outputs() const { return _outputs.value(); }

    virtual std::vector<Object> parse() const = 0;
    std::vector<std::string> _class_names;
};
