#include "camera.h"

#include <plog/Log.h>

#include <chrono>

namespace {

float get_depth_scale(const rs2::pipeline_profile& profile) {
    auto depth_scale = 0.f;

    const auto sensors = profile.get_device().query_sensors();
    for (const auto& s : sensors) {
        if (auto ds = s.as<rs2::depth_sensor>()) {
            depth_scale = ds.get_depth_scale();
            break;
        }
    }

    if (depth_scale <= 0.f) {
        LOG_WARNING << "Failed to get depth scale; defaulting to 0.001\n";
        depth_scale = 0.001f;
    }

    return depth_scale;
}

}  // namespace

namespace vision {

Frames::Frames(rs2::video_frame color, rs2::depth_frame depth,
               rs2::video_frame depth_colorized)
    : _color(std::move(color)),
      _depth(std::move(depth)),
      _depth_colorized(std::move(depth_colorized)) {
    _color_bgr = cv::Mat(_color.get_height(), _color.get_width(), CV_8UC3,
                         (void*)_color.get_data(), cv::Mat::AUTO_STEP);
    if (!_color_bgr.isContinuous()) {
        _color_bgr = _color_bgr.clone();
    }

    _depth_z16 = cv::Mat(_depth.get_height(), _depth.get_width(), CV_16U,
                         (void*)_depth.get_data(), cv::Mat::AUTO_STEP);
    if (!_depth_z16.isContinuous()) {
        _depth_z16 = _depth_z16.clone();
    }

    _depth_rgb = cv::Mat(
        _depth_colorized.get_height(), _depth_colorized.get_width(), CV_8UC3,
        (void*)_depth_colorized.get_data(), cv::Mat::AUTO_STEP);
    if (!_depth_rgb.isContinuous()) {
        _depth_rgb = _depth_rgb.clone();
    }
}

const cv::Mat& Frames::color() const { return _color_bgr; }

const cv::Mat& Frames::color_depth() const { return _depth_rgb; }

const cv::Mat& Frames::depth() const { return _depth_z16; }

float Frames::get_distance(int x, int y) const {
    return _depth.get_distance(x, y);
}

Camera::Camera(int width, int height, int fps)
    : _align_to_color(RS2_STREAM_COLOR) {
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
    cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
    // cfg.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8,
    //                   fps);
    auto profile = _pipe.start(cfg);

    _depth_scale = get_depth_scale(profile);
}

Camera::~Camera() { _pipe.stop(); }

std::optional<Frames> Camera::wait_for_frames() {
    const auto frames = _pipe.wait_for_frames();
    const auto aligned_frames = _align_to_color.process(frames);

    auto color = aligned_frames.get_color_frame();
    auto depth = aligned_frames.get_depth_frame();

    if (!color || !depth) {
        return std::nullopt;
    }

    auto depth_colorized = _colorizer.process(depth);

    return Frames{std::move(color), std::move(depth),
                  std::move(depth_colorized)};
}

float Camera::depth_scale() const { return _depth_scale; }

}  // namespace vision
