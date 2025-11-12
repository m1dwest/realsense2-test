#include "letterbox.h"

Letterbox img_to_letterbox(const cv::Mat& src, int lb_w, int lb_h,
                           const cv::Scalar& fill_color) {
    const int src_w = src.cols;
    const int src_h = src.rows;
    const float aspect_ratio = std::min(static_cast<float>(lb_w) / src_w,
                                        static_cast<float>(lb_h) / src_h);

    const int lb_img_w = static_cast<int>(std::round(src_w * aspect_ratio));
    const int lb_img_h = static_cast<int>(std::round(src_h * aspect_ratio));
    const int lb_img_x = (lb_w - lb_img_w) / 2;
    const int lb_img_y = (lb_h - lb_img_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(lb_img_w, lb_img_w));

    cv::Mat dst;
    cv::copyMakeBorder(resized, dst, lb_img_y, lb_h - lb_img_h - lb_img_y,
                       lb_img_x, lb_w - lb_img_w - lb_img_x,
                       cv::BORDER_CONSTANT, fill_color);
    return {aspect_ratio, lb_img_y, lb_img_x, lb_img_w,
            lb_img_h,     src_w,    src_h,    dst};
}

std::optional<cv::Rect> box_from_letterbox(float lb_cx, float lb_cy, float lb_w,
                                           float lb_h,
                                           const Letterbox& letterbox) {
    // YOLOv8 returns coordinates of box center and it's dimensions
    // here we calculate box coordinates in letterbox coordinate system
    float lb_x = lb_cx - lb_w / 2.0f;
    float lb_y = lb_cy - lb_h / 2.0f;

    // Remove padding and scale back
    float x0 = (lb_x - letterbox.img_x) / letterbox.aspect_ratio;
    float y0 = (lb_y - letterbox.img_y) / letterbox.aspect_ratio;
    float x1 = (lb_x + lb_w - letterbox.img_x) / letterbox.aspect_ratio;
    float y1 = (lb_y + lb_h - letterbox.img_y) / letterbox.aspect_ratio;

    // CLip
    int ix = std::max(0, (int)std::round(x0));
    int iy = std::max(0, (int)std::round(y0));
    int iw = std::min(letterbox.src_w - ix, (int)std::round(x1 - x0));
    int ih = std::min(letterbox.src_h - iy, (int)std::round(y1 - y0));

    return (iw <= 0 || ih <= 0) ? std::nullopt
                                : std::make_optional(cv::Rect{ix, iy, iw, ih});
}

std::optional<cv::Rect> box_from_letterbox(const cv::Rect& rect,
                                           const Letterbox& letterbox) {
    return box_from_letterbox(rect.x, rect.y, rect.width, rect.height,
                              letterbox);
}
