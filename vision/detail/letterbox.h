#pragma once

#include <opencv2/opencv.hpp>

struct Letterbox {
    float aspect_ratio;
    int img_y, img_x, img_w, img_h;
    int src_w, src_h;

    cv::Mat data;
};

Letterbox img_to_letterbox(const cv::Mat& src, int lb_w, int lb_h,
                           const cv::Scalar& fill_color);

std::optional<cv::Rect> box_from_letterbox(float lb_cx, float lb_cy, float lb_w,
                                           float lb_h,
                                           const Letterbox& letterbox);
std::optional<cv::Rect> box_from_letterbox(const cv::Rect& rect,
                                           const Letterbox& letterbox);
