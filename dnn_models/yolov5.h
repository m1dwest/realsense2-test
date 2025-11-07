#pragma once

#include "dnn_model.h"

class YOLOv5 : public DNNModel {
   private:
    const int INPUT_W = 640;
    const int INPUT_H = 640;
    const cv::Scalar LB_COLOR = cv::Scalar(114, 114, 114);

   public:
    using DNNModel::DNNModel;

    virtual void input(const cv::Mat& color_bgr) override {
        letterbox = this->get_letterbox(color_bgr, INPUT_W, INPUT_H, LB_COLOR);

        const auto blob = cv::dnn::blobFromImage(
            letterbox.data, 1.0 / 255.0, cv::Size(INPUT_W, INPUT_H),
            cv::Scalar(), /*swapRB*/ true, /*crop*/ false);

        this->input_preprocessed(std::move(blob));
    }
};
