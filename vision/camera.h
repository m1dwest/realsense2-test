#pragma once

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

namespace vision {

class Frames {
   public:
    Frames(rs2::video_frame color, rs2::depth_frame depth,
           rs2::video_frame depth_colorized);

    const cv::Mat& color() const;
    const cv::Mat& color_depth() const;
    const cv::Mat& depth() const;

    float get_distance(int x, int y) const;

   private:
    rs2::video_frame _color;
    rs2::depth_frame _depth;
    rs2::video_frame _depth_colorized;

    cv::Mat _color_bgr;
    cv::Mat _depth_z16;
    cv::Mat _depth_rgb;
};

class Camera {
   public:
    Camera(int width, int height, int fps);
    ~Camera();

    std::optional<Frames> wait_for_frames();
    float depth_scale() const;

   private:
    rs2::pipeline _pipe;
    rs2::align _align_to_color;
    rs2::colorizer _colorizer;
    float _depth_scale = 0.01f;
};

}  // namespace vision
